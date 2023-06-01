import numpy as np

with open('tinyshakespeare.txt', 'r') as file:
    text = file.read()


chars = sorted(list(set(text)))

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
batch_size = 4

def get_batch(split = 'val'):
    data = train_data if split == 'train' else val_data
    indexes = np.random.randint(0, len(data) - block_size, (batch_size,))
    xs = np.array([data[i : i + block_size] for i in indexes], dtype=np.uint8)
    ys = np.array([data[i + 1 : i + block_size + 1] for i in indexes], dtype=np.uint8)
    return xs, ys

class Model:
    def __init__(self, vocab_size):
        self.weights = np.random.uniform(0, 1, (vocab_size, vocab_size))
    
    def forward(self, xs, ys = None):
        return np.array([self.weights[x] for x in xs])

    def generate(self, xs, ys = None):
        print(xs.shape)
        logits = self.forward(xs)[:, -1]
        print(logits.shape)
        token = np.argmax(logits, axis=-1)
        result = np.reshape(np.append(xs.T, token.T).T, (batch_size, block_size + 1))
        print(result.shape)
        print(result)
        return result

model = Model(len(chars))
print([decode(block) for block in model.generate(*get_batch('val'))])