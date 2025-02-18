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

import time

from settings import dataset_name
from tokenizer import Tokenizer
from transformer import Transformer
from utils import load_transformer, pickle_load

text = """
It took perhaps thirty seconds for Snape to realize that he was on fire.
A sudden yelp told her she had done her job. Scooping the fire off him
into a little jar in her pocket, she scrambled back along the row --
Snape would never know what had happened."""
mode = "append"

assert(mode in ["append", "reprint"])

transformer = load_transformer(Transformer)

tokenizer = pickle_load(Tokenizer, f"tokenizers/{dataset_name}.pickle")
text_tokens = tokenizer.encode(text)[:-5]

if mode == "append":
    print(tokenizer.decode(text_tokens), end='')

for new_token in transformer.generate(text_tokens, 300):
    text_tokens.append(new_token)

    if mode == "append":
        print(tokenizer.decode([new_token]), end='')
    else:
        print("=" * 100)
        print(tokenizer.decode(text_tokens))

print("")