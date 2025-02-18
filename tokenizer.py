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

import itertools
import random

from settings import vocab_size
from tqdm import tqdm
from utils import split_into_segments

class Tokenizer:

    def train(self, text):

        print("Training tokenizer...")

        self.itob = [bytes([i]) for i in range(256)]

        text_segments = split_into_segments(text)
        index_segments = [list(text_segment.encode('utf-8')) for text_segment in text_segments]

        for i in range(10):

            index_segments = self.drop_tokens(index_segments)
            index_segments = self.generate_tokens(index_segments)

            total_tokens = sum([len(index_segment) for index_segment in index_segments])

            print(f"Compression ratio: {len(text) / total_tokens}")

            if total_tokens == len(index_segments):
                print(f"All words now represented by one token.")
                break

        self.itob = sorted(self.itob, key=len)
        self.itob.extend([bytes(" ") for i in range(len(self.itob) - vocab_size)])

        print("Training complete.")
        print(f"Compression ratio: {len(text) / sum([len(index_segment) for index_segment in index_segments])}")

    def count_singles(self, index_segments):
        single_counts = {}

        for index_segment in index_segments:
            for single in index_segment:
                single_counts[single] = single_counts[single] + 1 if single in single_counts else 1

        return single_counts


    def count_pairs(self, index_segments):
        pair_counts = {}

        for index_segment in index_segments:
            for pair in zip(index_segment[:-1], index_segment[1:]):
                pair_counts[pair] = pair_counts[pair] + 1 if pair in pair_counts else 1

        return pair_counts
        

    def drop_tokens(self, index_segments):

        if len(self.itob) == 256:
            return index_segments

        print("Dropping tokens... ")

        pair_counts = self.count_pairs(index_segments)
        single_counts = self.count_singles(index_segments)

        threshold = max(pair_counts.values()) * .8

        dropped_itob = {}
        dropped_list = list(range(len(self.itob)))

        for single in tqdm(range(len(self.itob) - 1, 255, -1)):
            if not single in single_counts or single_counts[single] < threshold:
                dropped_itob[single] = self.itob[single]
                del self.itob[single]
                del dropped_list[single]

        index_map = {old : new for new, old in enumerate(dropped_list)}

        for index_segment in tqdm(index_segments):
            
            index_segment[:] = [-index if index in dropped_itob else index_map[index] for index in index_segment]
            
            redo_start = None
            redo_end = -1
            
            for i, single in reversed(list(enumerate(index_segment))):

                if single < 0:
                    redo_end = max(redo_end, min(i + 10, len(index_segment)))
                    redo_start = max(i - 10, 0)
                
                if i == redo_start:
                    redo_bytes = [dropped_itob[-redo_index] if redo_index < 0 else self.itob[redo_index] for redo_index in index_segment[redo_start:redo_end]]
                    index_segment[redo_start:redo_end] = self.encode_bytes(redo_bytes, dropout=0, use_pbar=False)
                    
                    redo_start = None
                    redo_end = -1
                    
        return index_segments


    def generate_tokens(self, index_segments):

        print("Generating tokens...")

        pair_counts = self.count_pairs(index_segments)

        for t in tqdm(range(vocab_size - len(self.itob))):

            best_pair = max(pair_counts, key=pair_counts.get)

            if pair_counts[best_pair] == 0:
                break

            best_pair_index = len(self.itob)
            self.itob.append(self.itob[best_pair[0]] + self.itob[best_pair[1]])

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
        
        return index_segments

    def encode(self, text, dropout=.1, use_pbar=False):

        return self.encode_bytes(text.encode('utf-8'), dropout, use_pbar)

    def encode_bytes(self, text_bytes, dropout=.1, use_pbar=False):

        text_indexes = []

        pbar = tqdm(total=len(text_bytes)) if use_pbar else None
        i = 0

        while i < len(text_bytes):
            for token_index, token_bytes in reversed(list(enumerate(self.itob))):
                if i + len(token_bytes) <= len(text_bytes) and text_bytes[i:i+len(token_bytes)] == token_bytes and (token_index < 256 or random.random() > dropout):
                    text_indexes.append(token_index)
                    i += len(token_bytes)
                    if pbar:
                        pbar.update(len(token_bytes))
                    break

        if pbar:
            pbar.close()

        return text_indexes

    def decode_bytes(self, text_indexes):

        text_bytes = b''
        
        for index in text_indexes:
            text_bytes += self.itob[index] if index < len(self.itob) else b" "
        
        return text_bytes

    def decode(self, text_indexes):

        return self.decode_bytes(text_indexes).decode(errors='ignore')