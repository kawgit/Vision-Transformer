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
import itertools

from settings import vocab_size
from tqdm import tqdm
from utils import split_into_segments

class Tokenizer:

    def train(self, text):

        print("Training tokenizer...")

        self.btoi = {i : i for i in range(256)}
        self.itob = {i : bytes([i]) for i in range(256)}

        text_segments = split_into_segments(text)

        byte_segments = [bytes(text_segment, 'utf-8') for text_segment in text_segments]
        index_segments = [[self.btoi[byte] for byte in byte_segment] for byte_segment in byte_segments]

        pair_counts = {}

        for index_segment in index_segments:
            for pair in zip(index_segment[:-1], index_segment[1:]):
                pair_counts[pair] = pair_counts[pair] + 1 if pair in pair_counts else 1

        for i in tqdm(range(vocab_size - len(self.btoi))):
                
            best_pair = max(pair_counts, key=pair_counts.get)

            if pair_counts[best_pair] == 1:

                if len(index_segments) == 1:
                    print("Warning: vocab_size is so large relative to the training dataset that no more pairs were found with a higher count than 1.")
                    break
                
                index_segments = [list(itertools.chain.from_iterable(index_segments))]

            best_pair_index = len(self.btoi)
            best_pair_bytes = self.itob[best_pair[0]] + self.itob[best_pair[1]]

            self.btoi[best_pair_bytes] = best_pair_index
            self.itob[best_pair_index] = best_pair_bytes

            pair_counts[best_pair] = 0
            
            for index_segment in index_segments:
                for i in reversed(range(len(index_segment) - 1)):

                    if tuple(index_segment[i:i+2]) != best_pair:
                        continue

                    if i != 0:

                        old_pair = (index_segment[i-1], index_segment[i])
                        pair_counts[old_pair] -= 1

                        new_pair = (index_segment[i-1], best_pair_index)
                        pair_counts[new_pair] = pair_counts[new_pair] + 1 if new_pair in pair_counts else 1

                    if i != len(index_segment) - 2:

                        old_pair = (index_segment[i+1], index_segment[i+2])
                        pair_counts[old_pair] -= 1

                        new_pair = (best_pair_index, index_segment[i+2])
                        pair_counts[new_pair] = pair_counts[new_pair] + 1 if new_pair in pair_counts else 1
                    
                    index_segment[i:i+2] = [best_pair_index]

        print("Training complete.")
        print(f"Compression ratio: {len(text) / sum([len(index_segment) for index_segment in index_segments])}")

    def encode(self, text, dropout=.1, use_pbar=False):

        self.verify()

        text_segments = split_into_segments(text)
        byte_segments = [bytes(text_segment, 'utf-8') for text_segment in text_segments]
        indexes = []

        pbar = tqdm(total=sum([len(byte_segment) for byte_segment in byte_segments])) if use_pbar else None

        for byte_segment in byte_segments:
            i = 0
            while i < len(byte_segment):
                for token_index, token_bytes in reversed(self.itob.items()):
                    if i + len(token_bytes) <= len(byte_segment) and byte_segment[i:i+len(token_bytes)] == token_bytes and random.random() > dropout:
                        indexes.append(token_index)
                        i += len(token_bytes)
                        if pbar:
                            pbar.update(len(token_bytes))
                        break

        if pbar:
            pbar.close()

        return indexes
        
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
        assert(len(self.btoi) <= vocab_size)
        assert(len(self.itob) <= vocab_size)
