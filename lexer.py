from collections import Counter
import re
import numpy as np

with open('mediocre.txt', 'r', encoding="utf8") as file:
    filetext = file.read()

words = Counter(filter(lambda x: len(x) > 1, re.split("\s|\n|\.|\:|“|”|,|\"|/|\?|\!|@|\-", filetext)))
customs = np.array(words.most_common(500))[:, 0]

temp = filetext
for token in customs:
    temp.replace(token, "")
standards = sorted(list(set(temp)))

vocab_size = len(customs) + len(standards)
num_customs = len(customs)
num_standards = len(standards)

stoi = {c:i for i, c in enumerate(standards)}
itos = {i:c for i, c in enumerate(standards)}

for idx, token in enumerate(customs):
    stoi[token] = idx + num_standards
    itos[idx + num_standards] = token

def encode(text):
    tokens = []
    while(len(text) > 0):
        found = False
        for idx, token in enumerate(customs):
            if text.startswith(token):
                tokens.append(stoi[token])
                text = text[len(token):]
                found = True
        
        if not found:
            tokens.append(stoi[text[0]])
            text = text[1:]
    return tokens

def decode(tokens):
    result = ""
    for token in tokens:
        result += itos[token]
    return result

print(vocab_size)