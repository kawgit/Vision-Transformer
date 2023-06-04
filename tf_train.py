import numpy as np
import random
import time
import os
import glob
import time
import math

print("importing tensorflow...")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
print("tensorflow", tf.__version__, "imported.")

print(tf.test.gpu_device_name())

block_size = 32
batch_size = 16
num_batches = 100
num_embd = 32
optimizer = tf.keras.optimizers.Adam(learning_rate=.001)
loss_fn = tf.keras.losses.CategoricalCrossentropy()
eval_itrs = 32
key_size = 32
query_size = key_size

with open('tinyshakespeare.txt', 'r') as file:
    text = file.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {c:i for i, c in enumerate(chars)}
itos = {i:c for i, c in enumerate(chars)}

def encode(txt):
    return [stoi[c] for c in txt]

def decode(tokens):
    return ''.join([itos[token] for token in tokens])

tokens = np.array(encode(text), dtype=np.uint8)

train_data = tokens[:int(len(tokens) * .9)]
val_data = tokens[int(len(tokens) * .9):]

def get_batch(split='val', size=batch_size):
    data = train_data if split == 'train' else val_data
    indexes = np.random.randint(0, len(data) - block_size, (size,))
    xs = np.array([data[i : i + block_size] for i in indexes], dtype=np.uint8)
    ys = np.array([data[i + 1 : i + block_size + 1] for i in indexes], dtype=np.uint8)
    return xs, ys

def load_embdmaker(path=None):
    if (path != None and os.path.isdir(path)):
        path += "/embdmaker"
        assert(os.path.isdir(path))
        return tf.keras.models.load_model(path)

    truths = tf.keras.Input(shape=(None, vocab_size))
    embd = tf.keras.layers.Dense(num_embd, use_bias=False)(truths)
    model = tf.keras.Model(inputs=truths, outputs=embd, name="embdmaker")
    return model

def load_position_encoder(path=None):
    if (path != None and os.path.isdir(path)):
        path += "/position_encoder"
        if os.path.isdir(path):
            return tf.keras.models.load_model(path)

    truths = tf.keras.Input(shape=(None, vocab_size))
    embd = tf.keras.layers.Dense(num_embd, use_bias=False, kernel_initializer=tf.keras.initializers.Zeros())(truths)
    model = tf.keras.Model(inputs=truths, outputs=embd, name="position_encoder")
    return model

def load_querymaker(path=None):
    if (path != None and os.path.isdir(path)):
        path += "/querymaker"
        assert(os.path.isdir(path))
        return tf.keras.models.load_model(path)

    embds = tf.keras.Input(shape=(None, vocab_size))
    query = tf.keras.layers.Dense(query_size, use_bias=False)(embds)
    model = tf.keras.Model(inputs=embds, outputs=query, name="querymaker")
    return model

def load_keymaker(path=None):
    if (path != None and os.path.isdir(path)):
        path += "/keymaker"
        assert(os.path.isdir(path))
        return tf.keras.models.load_model(path)

    embds = tf.keras.Input(shape=(None, vocab_size))
    key = tf.keras.layers.Dense(key_size, use_bias=False)(embds)
    model = tf.keras.Model(inputs=embds, outputs=key, name="keymaker")
    return model

def load_softy(path=None):
    if (path != None and os.path.isdir(path)):
        path += "/softy"
        assert(os.path.isdir(path))
        return tf.keras.models.load_model(path)

    embds = tf.keras.Input(shape=(None, num_embd))
    x = tf.keras.layers.Dense(vocab_size, activation='sigmoid')(embds)
    x = tf.keras.layers.Dense(vocab_size, activation='sigmoid')(x)
    x = tf.keras.layers.Dense(vocab_size, activation='sigmoid')(x)
    logits = tf.keras.layers.Dense(vocab_size, activation='softmax')(x)
    model = tf.keras.Model(inputs=embds, outputs=logits, name="softy")
    return model

def np_to_categorical(y):
    return np.eye(vocab_size)[y]

class Model:

    def __init__(self, path='__latest__'):
        if path == '__latest__':
            all_subdirs = ['checkpoints/' + d for d in os.listdir('checkpoints') if os.path.isdir('checkpoints/' + d)] # * means all if need specific format then *.csv
            if (len(all_subdirs) == 0):
                print("no checkpoints found")
                path = None
            else:
                latest_dir = max(all_subdirs, key=os.path.getctime)
                print("loading latest checkpoint:", latest_dir)
                path = latest_dir

        self.embdmaker = load_embdmaker(path)
        self.querymaker = load_querymaker(path)
        self.keymaker = load_keymaker(path)
        self.softy = load_softy(path)
        self.position_encoder = load_position_encoder(path)

        block = tf.keras.Input(shape=(block_size, vocab_size))
        embds = tf.math.add(self.embdmaker(block), self.position_encoder(block))
        queries = self.querymaker(block)
        keys = self.keymaker(block)

        # print(block.shape)
        # print(embds.shape)
        # print(queries.shape)
        # print(keys.shape)

        # exit()

        weights = tf.matmul(queries, keys, transpose_b=True) #(Tq, Tk)

        tril = tf.constant(np.tril(np.ones((block_size, block_size))), dtype=tf.float32)
        weights = tf.math.multiply(weights, tril)

        negtril = tf.constant((np.tril(np.ones((block_size, block_size))) - 1) * (1.7976931348623157e+308), dtype=tf.float32)
        weights = tf.math.add(weights, negtril)

        # weights = tf.math.exp(weights)
        # sums = tf.math.reduce_sum(weights, axis=-1)
        # sums = tf.repeat(tf.reshape(sums, (tf.shape(sums)[0], block_size, 1)), block_size, axis=-1)

        # weights = tf.divide(weights, sums)

        weights = tf.nn.softmax(weights, axis=-1)

        weighted_embds = tf.matmul(weights, embds)

        logits = self.softy(weighted_embds)
        
        self.training_model = tf.keras.Model(inputs=block, outputs=logits, name="embdmaker")
    
    def save_checkpoint(self, loss=0):
        model_name = 'checkpoints/' + str(round(time.time())) + "_" + str(round(float(loss), 2))
        self.embdmaker.save(model_name + '/embdmaker')
        self.position_encoder.save(model_name + '/position_encoder')
        self.querymaker.save(model_name + '/querymaker')
        self.keymaker.save(model_name + '/keymaker')
        self.softy.save(model_name + '/softy')
    
    def train(self, epochs):
        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))
            for i in range(num_batches):

                xs, ys = get_batch(split='train')
                xs = tf.keras.utils.to_categorical(xs, num_classes=vocab_size)
                ys = tf.keras.utils.to_categorical(ys, num_classes=vocab_size)

                with tf.GradientTape() as tape:
                    logits = self.training_model(xs)
                    loss_value = loss_fn(ys, logits)
                    
                grads = tape.gradient(loss_value, self.training_model.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.training_model.trainable_weights))

            loss_est = loss_value
            print("Val loss at epoch", epoch, "is", float(loss_est))

            if epoch % 10 == 0:
                self.save_checkpoint(loss_est)
    
    def generate(self, seed="What a strange drowsiness possesses them! \n Wha", num_chars=100):
        x = list(encode(seed))
        for i in range(num_chars):
            probs = self.training_model(np_to_categorical(np.array([x[-block_size:]])), training=False).numpy()[0][-1]
            # print(list(reversed(sorted(probs)))[:5])
            choice = np.random.choice(np.arange(vocab_size), p=probs)
            x.append(choice)
        return decode(np.array(x)).replace("\n", "\\n")


"""

def estimate_loss(model):
    xs, ys = get_batch(split='val', size=eval_itrs)
    ys = tf.keras.utils.to_categorical(ys, num_classes=vocab_size)
    logits = model(xs, training=False).reshape(eval_itrs, vocab_size)
    losses = loss_fn(ys, logits)
    return float(losses)
"""

model = Model()

# model.train(10000)
# train_model(model, 10)
print(model.generate(num_chars=10))
print(model.generate(num_chars=10))
print(model.generate(num_chars=10))
print(model.generate(num_chars=10))
# print(generate(model, num_chars=1000))