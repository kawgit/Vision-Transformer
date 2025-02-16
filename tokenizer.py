#    Copyright 2025 Kenneth Wilber (kawgit)

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import random

from settings import vocab_size
from tqdm import tqdm

class Tokenizer:

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

    def encode(self, text, dropout=.1, use_pbar=False):

        self.verify()

        text_bytes = bytes(text, 'utf-8')
        text_indexes = []

        pbar = tqdm(total=len(text_bytes)) if use_pbar else None
        i = 0

        while i < len(text_bytes):
            for token_index, token_bytes in reversed(self.itob.items()):
                if i + len(token_bytes) <= len(text_bytes) and text_bytes[i:i+len(token_bytes)] == token_bytes and random.random() > dropout:
                    text_indexes.append(token_index)
                    i += len(token_bytes)
                    if pbar:
                        pbar.update(len(token_bytes))
                    break

        if pbar:
            pbar.close()

        return text_indexes
        
    def decode_bytes(self, text_indexes):

        self.verify()

        text_bytes = b''
        
        for index in text_indexes:
            text_bytes += self.itob[index]
        
        return text_bytes

    def decode(self, text_indexes):

        self.verify()

        return self.decode_bytes(text_indexes).decode()
    
    def verify(self):
        assert(len(self.btoi) == vocab_size)
        assert(len(self.itob) == vocab_size)
