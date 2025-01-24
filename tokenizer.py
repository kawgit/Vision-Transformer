

import random

from settings import *
from tqdm import tqdm
from utils import load_file_as_string, pickle_save

class Tokenizer:

    def __init__(self):
        pass

    def train(self, text):

        print("Training tokenizer...")

        self.btoi = {i : i for i in range(256)}
        self.itob = {i : bytes([i]) for i in range(256)}

        text_bytes = bytes(text, 'utf-8')
        text_indexes = [self.btoi[i] for i in text_bytes]

        pair_counts = {}

        for pair in zip(text_indexes[:-1], text_indexes[1:]):

            pair_counts[pair] = pair_counts[pair] + 1 if pair in pair_counts else 1

        for i in tqdm(range(vocab_size - len(self.btoi))):

            if len(text_indexes) == 1:
                break

            best_pair = max(pair_counts, key=pair_counts.get)

            best_pair_index = len(self.btoi)
            best_pair_bytes = self.itob[best_pair[0]] + self.itob[best_pair[1]]

            self.btoi[best_pair_bytes] = best_pair_index
            self.itob[best_pair_index] = best_pair_bytes

            pair_counts[best_pair] = 0

            for i in reversed(range(len(text_indexes) - 1)):
                if tuple(text_indexes[i:i+2]) != best_pair:
                    continue

                if i != 0:

                    old_pair = (text_indexes[i-1], text_indexes[i])
                    pair_counts[old_pair] -= 1

                    new_pair = (text_indexes[i-1], best_pair_index)
                    pair_counts[new_pair] = pair_counts[new_pair] + 1 if new_pair in pair_counts else 1

                if i != len(text_indexes) - 2:

                    old_pair = (text_indexes[i+1], text_indexes[i+2])
                    pair_counts[old_pair] -= 1

                    new_pair = (best_pair_index, text_indexes[i+2])
                    pair_counts[new_pair] = pair_counts[new_pair] + 1 if new_pair in pair_counts else 1
                
                text_indexes[i:i+2] = [best_pair_index]

        
        print("Training complete.")
        print(f"Compression ratio: {len(text) / len(text_indexes)}")

    def encode(self, text, dropout=.1):

        self.verify()

        text_bytes = bytes(text, 'utf-8')
        text_indexes = []
        
        i = 0
        while i < len(text_bytes):

            for token_index, token_bytes in reversed(self.itob.items()):

                if i + len(token_bytes) <= len(text_bytes) and text_bytes[i:i+len(token_bytes)] == token_bytes and random.random() > dropout:
                    text_indexes.append(token_index)
                    i += len(token_bytes)
                    break

        return text_indexes

    def decode(self, text_indexes):

        self.verify()

        text_bytes = b''
        
        for index in text_indexes:
            text_bytes += self.itob[index]
        
        return text_bytes.decode("utf-8")
    
    def verify(self):
        assert(len(self.btoi) == vocab_size)
        assert(len(self.itob) == vocab_size)


if __name__ == "__main__":
    
    text = load_file_as_string(f"datasets/{dataset_name}.txt")

    tokenizer = Tokenizer()
    tokenizer.train(text)

    pickle_save(tokenizer, f"tokenizers/{dataset_name}.pickle")



