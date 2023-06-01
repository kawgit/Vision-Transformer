import numpy as np
import random
import time

block_size = 8
batch_size = 512

l1_size = 32

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

def get_batch(split = 'val'):
    data = train_data if split == 'train' else val_data
    indexes = np.random.randint(0, len(data) - block_size, (batch_size,))
    xs = np.array([data[i : i + block_size] for i in indexes], dtype=np.uint8)
    ys = np.array([data[i + 1 : i + block_size + 1] for i in indexes], dtype=np.uint8)
    return xs, ys

def softmax(xs):
    return np.array([np.exp(x) / np.sum(np.exp(x), axis=0) for x in xs])

def categorize(blocks):
    result = np.zeros((*blocks.shape, vocab_size))
    for i, block in enumerate(blocks):
        for j, idx in enumerate(block):
            result[i][j][idx] = 1
    return result

def select_idx(block):
    r = random.uniform(0, 1)
    for idx, prob in enumerate(block):
        r -= block[idx]
        if r < 0:
            return idx
    return len(block) - 1

def select_idxs(blocks):
    return np.array([select_idx(block) for block in blocks])

class Model:
    def __init__(self):
        self.in_l1_weights = np.random.uniform(0, 1, (vocab_size, l1_size)) #embedding lookup layer
        self.l1_out_weights = np.random.uniform(0, 1, (l1_size, vocab_size))
        self.l1 = np.zeros((l1_size,))
        self.l2 = np.zeros((vocab_size,))
    
    def forward(self, xs, ys = None):
        self.l1 = np.array([self.in_l1_weights[block[-1]] for block in xs])
        self.l2 = np.dot(self.l1, self.l1_out_weights)
        return softmax(self.l2)

    def generate(self, xs, ys = None, num = 1):
        result = xs
        for i in range(num):
            probs = self.forward(result)
            token = select_idxs(probs).reshape(len(xs), 1)
            result = np.append(result, token, 1)
        
        if ys is None:
            return result
        return result, None
        ys_probs = categorize(ys[:, -1:]).reshape((batch_size, vocab_size))
        loss = -np.sum(ys_probs * np.log(probs), axis=-1)
        return result, loss
    
    def train(self, epochs, rate):
        for epoch in range(epochs):
            batch = get_batch('train')
            grad_w1 = np.zeros(self.l1_out_weights.shape)
            grad_w0 = np.zeros(self.in_l1_weights.shape)
            total_loss = 0
            for i in range(1, block_size):
                xs, ys = batch[0][:, :i], batch[1][:, :i]

                probs = self.forward(xs)
                expected_probs = categorize(ys[:, -1:]).reshape((batch_size, vocab_size))

                loss = -np.sum(expected_probs * np.log(probs), axis=-1)
                # print(loss)

                dLdp = -expected_probs / probs
                explogits = np.exp(self.l2)
                S = np.repeat(np.sum(explogits, axis=-1).reshape((batch_size,1)), vocab_size, axis=-1)
                k = S - explogits
                dpdl1 = (explogits * k) / np.power(explogits + k, 2)
                dLdl1 = dLdp * dpdl1

                dLdw1 = np.repeat(dLdl1.reshape(batch_size, 1, vocab_size), l1_size, axis=-2) * self.in_l1_weights[xs[:,-1]].reshape(batch_size, l1_size, 1)
                dLdw0 = np.dot(dLdl1, self.l1_out_weights.T)

                # print(dLdw0)
                # print(dLdw1)

                for b, block in enumerate(xs):
                    grad_w1 += dLdw1[b]
                    grad_w0[xs[b][-1]] += dLdw0[b]
                total_loss += np.sum(loss) / batch_size

            grad_w1 /= batch_size
            grad_w0 /= batch_size

            print("epoch", epoch, "mean loss", total_loss / (block_size + 1))
            print(np.min(grad_w0), np.max(grad_w0))
            print(np.min(grad_w1), np.max(grad_w1))
            self.in_l1_weights -= grad_w0 * rate
            self.l1_out_weights -= grad_w1 * rate
            # time.sleep(2)
            


def print_encoded(blocks, losses = None):
    print([decode(block) + ("" if losses is None else " " + str(losses[i])) for i, block in enumerate(blocks)])

model = Model()
batch = get_batch('val')
print_encoded(batch[1])
print_encoded(*model.generate(*batch, num=10))

model.train(50, .01)

print_encoded(model.generate(xs = np.array([encode('the')]), ys = None, num = 200))