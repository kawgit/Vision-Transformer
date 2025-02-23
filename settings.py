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

import os

dataset_name = "harrypotter1"

vocab_size = 10000

context_size = 500
key_size = 128
head_size = 128
layer_size = 8
hidden_size = 2048
embedding_size = head_size * layer_size
num_layers = 2

assert(key_size % 2 == 0) # Necessary condition for rotary positional encoding

batch_size = 16
learning_rate = 3e-4
epochs = 10

model_name = f"{dataset_name}_c{context_size}_m{embedding_size}_k{key_size}_h{head_size}_l{num_layers}_v{hidden_size}"
model_path = os.path.join('transformers', model_name + '.pt')