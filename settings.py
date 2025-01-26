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

dataset_name = "tinyshakespeare"

vocab_size = 10000

context_size = 100
embedding_size = 512
key_size = 16
num_layers = 8
layer_size = 8

head_size = embedding_size // layer_size

assert(head_size * layer_size == embedding_size)

batch_size = 16
learning_rate = 3e-5
epochs = 10