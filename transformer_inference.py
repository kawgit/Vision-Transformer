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
from tokenizer import load_tokenizer
from transformer import load_transformer
from device import device

text = """
It seemed that Professor McGonagall had reached the point she was most
anxious to discuss, the real reason she had been waiting on a cold, hard
wall all day, for neither as a cat nor as a woman had she fixed"""
mode = "append"

assert(mode in ["append", "reprint"])

transformer = load_transformer().to(device)
tokenizer = load_tokenizer()
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