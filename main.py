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

block_size = 128
batch_size = 64
num_batches = 16
optimizer = tf.keras.optimizers.Adam(learning_rate=.0003)
loss_fn = tf.keras.losses.CategoricalCrossentropy()
eval_itrs = 32
num_heads = 6
num_sablocks = 6
dropout_rate = .2
head_size = 32
num_embds = num_heads * head_size


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

def get_batches(split='val', num=1, size=batch_size):
    data = train_data if split == 'train' else val_data
    indexes = np.random.randint(0, len(data) - block_size, (size * num,))
    xs = np.array([data[i : i + block_size] for i in indexes], dtype=np.uint8)
    ys = np.array([data[i + 1 : i + block_size + 1] for i in indexes], dtype=np.uint8)
    ys = tf.keras.utils.to_categorical(ys, num_classes=vocab_size)
    return xs, ys

def np_to_categorical(y):
    return np.eye(vocab_size)[y]

class Head:
    def __init__(self, name):
        embds = tf.keras.Input(shape=(None, num_embds))
        
        queries = tf.keras.layers.Dense(head_size, name='queries_maker')(embds)
        keys = tf.keras.layers.Dense(head_size, name='keys_maker')(embds)
        values = tf.keras.layers.Dense(num_embds, name='values_maker')(embds)

        weights = tf.matmul(queries, keys, transpose_b=True) #(Tq, Tk)
        tril = tf.constant(np.tril(np.ones((block_size, block_size))) * (head_size**-.5), dtype=tf.float32)
        weights = tf.math.multiply(weights, tril)
        negtril = tf.constant((np.tril(np.ones((block_size, block_size))) - 1) * (1.7976931348623157e+308), dtype=tf.float32)
        weights = tf.math.add(weights, negtril)
        weights = tf.nn.softmax(weights, axis=-1)
        weights = tf.keras.layers.Dropout(dropout_rate)(weights)

        weighted_embds = tf.matmul(weights, values)

        self.model = tf.keras.Model(inputs=embds, outputs=weighted_embds, name=name)
        tf.keras.utils.plot_model(self.model, to_file='head.png')
    

class SABlock:
    def __init__(self, name):

        self.heads = [Head("head_" + str(i)) for i in range(num_heads)]

        embds = tf.keras.Input(shape=(None, num_embds))
        out = embds

        #MHA
        out = tf.keras.layers.LayerNormalization()(out)
        weighted_embds = [head.model(embds) for head in self.heads]
        weighted_embds = tf.concat(weighted_embds, 2)
        weighted_embds = tf.keras.layers.Dropout(dropout_rate)(weighted_embds)
        proj = tf.keras.layers.Dense(num_embds, use_bias=False, kernel_initializer=tf.keras.initializers.Zeros())(weighted_embds)
        out = tf.math.add(out, proj)

        #FF
        out = tf.keras.layers.LayerNormalization()(out)
        x = tf.keras.layers.Dense(num_embds * 4, activation='relu')(out)
        x = tf.keras.layers.Dense(num_embds, activation='relu', kernel_initializer=tf.keras.initializers.Zeros(), bias_initializer=tf.keras.initializers.Zeros())(x)
        out = tf.math.add(out, x)

        self.model = tf.keras.Model(inputs=embds, outputs=out, name=name)
        tf.keras.utils.plot_model(self.model, to_file='sablock.png')


class Model:
    def __init__(self, path='__latest__'):
        self.sablocks = [SABlock("sablock_" + str(i)) for i in range(num_sablocks)]

        block = tf.keras.Input(shape=(None,))
        embds = tf.keras.layers.Embedding(vocab_size, num_embds)(block)

        for sablock in self.sablocks:
            embds = sablock.model(embds)

        # logits = tf.keras.layers.Dense(vocab_size, activation='relu')(embds)
        # logits = tf.keras.layers.Dense(vocab_size, activation='sigmoid')(logits)
        logits = tf.keras.layers.Dense(vocab_size, activation='softmax')(embds)
        
        self.model = tf.keras.Model(inputs=block, outputs=logits, name="transformer")
        self.model.compile(optimizer=optimizer, loss=loss_fn)
        tf.keras.utils.plot_model(self.model, to_file='model.png')

        if path == '__latest__':
            all_subdirs = ['checkpoints/' + d for d in os.listdir('checkpoints') if os.path.isdir('checkpoints/' + d)] # * means all if need specific format then *.csv
            if (len(all_subdirs) == 0):
                path = None
                print("no checkpoints found")
            else:
                path = max(all_subdirs, key=os.path.getctime)
                print("loading latest checkpoint:", path)
                self.model.load_weights(path)
    
    def save_checkpoint(self, loss=0):
        path = 'checkpoints/' + str(round(time.time())) + "_" + str(round(float(loss), 2))
        self.model.save_weights(path)
        for sablock in self.sablocks:
            sablock_path = path + "/" + sablock.model.name 
            for head in sablock.heads:
                head_path = sablock_path + "/" + head.model.name
                head.model.save_weights(head_path) 

    def train(self, epochs):
        for epoch in range(epochs // (5 * 100)):
            print("generating batches...")
            xs, ys = get_batches(split='train', num=num_batches)
            print("finished generating batches.")
            self.model.fit(x=xs, y=ys, batch_size=batch_size, epochs=10)

            loss = self.model.evaluate(*get_batches(split='val', num=num_batches))
            print("val_loss:", str(loss))

            if epoch % 10 == 0:
                self.save_checkpoint(loss)
                print("checkpoint saved.")
        self.save_checkpoint(loss)
        print("checkpoint saved.")
    
    def generate(self, seed="Nor I; my spirits are nimble.\nThey fell together all, as by consent;\nThey dropp'd, as by a thunder-stroke. What might,\nWorthy Sebastian? O, what might?--No more:--\nAnd yet me thinks I see it in thy face,\nWhat thou shouldst be: the occasion speaks thee, and\n", num_chars=100):
        x = list(encode(seed))
        for i in range(num_chars):
            inp = np.array([x[-block_size:]])
            pad_len = block_size - inp.shape[1]
            inp = np.pad(inp, ((0, 0), (0, pad_len)))
            probs = self.model(inp).numpy()[0][-pad_len - 1]
            choice = np.random.choice(np.arange(vocab_size), p=probs)
            x.append(choice)
            print("itr", i)
            print(decode(x))
        return decode(np.array(x))


model = Model()

# model.train(1000000)

print(model.generate(seed="hello world!", num_chars=10000))