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

from settings import *
from tokenizer import Tokenizer
from utils import pickle_load, split_into_segments

text = """
It took perhaps thirty seconds for Snape to realize that he was on fire.
A sudden yelp told her she had done her job. Scooping the fire off him
into a little jar in her pocket, she scrambled back along the row --
Snape would never know what had happened.
"""

tokenizer = pickle_load(Tokenizer, f"tokenizers/{dataset_name}.pickle")
text_indexes = tokenizer.encode(text)

for token_index in text_indexes:

    token_bytes = tokenizer.itob[token_index]

    try:
        msg = token_bytes.decode('utf-8').replace('\n', '\\n').replace('\t', '\\t').replace(' ', '_')
    except:
        msg = str(token_bytes)

    print(msg)



