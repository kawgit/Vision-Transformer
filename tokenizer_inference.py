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
from tokenizer import load_tokenizer
from utils import split_into_segments

text = """
It seemed that Professor McGonagall had reached the point she was most
anxious to discuss, the real reason she had been waiting on a cold, hard
wall all day, for neither as a cat nor as a woman had she fixed
"""

tokenizer = load_tokenizer()

text_indexes = tokenizer.encode(text, dropout=0)

for token_index in text_indexes:

    token_bytes = tokenizer.itob[token_index]

    try:
        msg = token_bytes.decode('utf-8').replace('\n', '\\n').replace('\t', '\\t').replace(' ', '_')
    except:
        msg = str(token_bytes)

    print(msg)

