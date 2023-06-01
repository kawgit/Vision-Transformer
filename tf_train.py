import numpy as np
import random
import time
import os
import glob
import time

print("importing tensorflow...")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
print("tensorflow imported.")

block_size = 8
batch_size = 64
num_batches = 100
num_embd = 32
optimizer = tf.keras.optimizers.Adam(learning_rate=.001)
loss_fn = tf.keras.losses.CategoricalCrossentropy()
eval_itrs = 32

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
    ys = np.array([data[i + block_size + 1] for i in indexes], dtype=np.uint8)
    return xs, ys

def load_model(path='__latest__'):
    if path == '__latest__':
        all_subdirs = ['checkpoints/' + d for d in os.listdir('checkpoints') if os.path.isdir('checkpoints/' + d)] # * means all if need specific format then *.csv
        if (len(all_subdirs) == 0):
            print("no checkpoints found")
            path = None
        else:
            latest_dir = max(all_subdirs, key=os.path.getctime)
            print("loading latest checkpoint:", latest_dir)
            path = latest_dir

    if (path != None and os.path.isdir(path)):
        return tf.keras.models.load_model(path)
    else:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(vocab_size, num_embd, input_length=block_size))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(vocab_size, activation='sigmoid'))
        model.add(tf.keras.layers.Dense(vocab_size, activation='sigmoid'))
        model.add(tf.keras.layers.Dense(vocab_size, activation='softmax'))
        model.compile('adam', loss='sparse_categorical_crossentropy')
        return model

def save_checkpoint(model, loss):
    model.save('checkpoints/' + str(round(time.time())) + "_" + str(round(loss, 2)))

def estimate_loss(model):
    xs, ys = get_batch(split='val', size=eval_itrs)
    ys = tf.keras.utils.to_categorical(ys, num_classes=vocab_size)
    logits = model(xs, training=False).reshape(eval_itrs, vocab_size)
    losses = loss_fn(ys, logits)
    return float(losses)

def train_model(model, epochs):
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        for i in range(num_batches):

            xs, ys = get_batch(split='train')
            ys = tf.keras.utils.to_categorical(ys, num_classes=vocab_size)

            with tf.GradientTape() as tape:
                logits = model(xs, training=True).reshape((batch_size, vocab_size))
                loss_value = loss_fn(ys, logits)
                
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

        loss_est = estimate_loss(model)
        print("Val loss at epoch", epoch, "is", loss_est)
        save_checkpoint(model, loss_est)

def generate(model, seed="There has", num_chars=100):
    x = list(encode(seed))
    for i in range(num_chars):
        probs = model(np.array([x[-8:]]), training=False).numpy().reshape(vocab_size)
        choice = np.random.choice(np.arange(vocab_size), p=probs)
        x.append(choice)
    return decode(np.array(x))
        

model = load_model('__latest__')

train_model(model, 10)

print(generate(model, num_chars=1000))