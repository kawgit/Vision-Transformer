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

text = """But on the edge of town, drills were driven out of his mind by something
else. As he sat in the usual morning traffic jam, he couldn't help
noticing that there seemed to be a lot of strangely dressed people
about. People in cloaks. Mr. Dursley couldn't bear people who dressed in
funny clothes -- the getups you saw on young people! He supposed this
was some stupid new fashion."""
mode = "append"

assert(mode in ["append", "reprint"])

transformer = load_transformer(Transformer)

tokenizer = pickle_load(Tokenizer, f"tokenizers/{dataset_name}.pickle")
text_tokens = tokenizer.encode(text)

for new_token in transformer.generate(text_tokens, 100):
    text_tokens.append(new_token)

    if mode == "append":
        print(tokenizer.decode([new_token]), end='')
    else:
        print("=" * 100)
        print(tokenizer.decode(text_tokens))