import numpy as np
import random

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

block_size = 8
batch_size = 128

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
        self.weights = np.random.uniform(0, 1, (vocab_size, vocab_size))
    
    def forward(self, xs, ys = None):
        return np.array([[self.weights[idx] for idx in block] for block in xs])

    def generate(self, xs, ys = None, num = 1):
        result = xs
        for i in range(num):
            logits = self.forward(result)[:, -1]
            probs = softmax(logits)
            token = select_idxs(probs).reshape(len(xs), 1)
            result = np.append(result, token, 1)
        
        if ys is None:
            return result
        return result, 1
        ys_probs = categorize(ys[:, -1:]).reshape((batch_size, vocab_size))
        loss = -np.sum(ys_probs * np.log(probs), axis=-1)
        return result, loss
    
    def train(self, epochs, rate):
        for epoch in range(epochs):
            batch = get_batch('train')
            grad = np.zeros(self.weights.shape)
            total_loss = 0
            for i in range(1, block_size):
                xs, ys = (batch[0])[:, :i], (batch[1])[:, :i]
                logits = self.forward(xs)[:, -1]
                # print(logits)
                probs = softmax(logits)
                token = np.argmax(probs, axis=-1)
                ys_probs = categorize(ys[:, -1:]).reshape((batch_size, vocab_size))
                loss = -np.sum(ys_probs * np.log(probs), axis=-1)
                dLdp = -ys_probs / probs
                explogits = np.exp(logits)
                S = np.repeat(np.sum(explogits, axis=-1).reshape((batch_size,1)), vocab_size, axis=-1)
                k = S - explogits
                dpdw = (explogits * k) / np.power(explogits + k, 2)
                dLdw = dLdp * dpdw
                
                # print(dLdw)
                for b, block in enumerate(xs):
                    grad[xs[b][-1]] += dLdw[b]
                # grad[xs[-1]] += dLdw
                # print(grad[0])

                total_loss += np.sum(loss) / batch_size

            print("epoch", epoch, "mean loss", total_loss / (block_size - 1))
            self.weights -= grad * rate
            


def print_encoded(blocks, losses = None):
    print([decode(block) + ("" if losses is None else " " + str(losses[i])) for i, block in enumerate(blocks)])

model = Model()
# batch = get_batch('val')
# print_encoded(batch[1])
# print_encoded(*model.generate(*batch))

model.train(100, .01)

print_encoded(model.generate(xs = np.array([encode('the')]), ys = None, num = 200))