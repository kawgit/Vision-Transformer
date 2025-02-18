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

dataset_name = "harrypotter1"

vocab_size = 10000

context_size = 100
key_size = 128
head_size = 128
layer_size = 8
hidden_size = 2048
embedding_size = head_size * layer_size
num_layers = 2

batch_size = 32
learning_rate = 3e-4
epochs = 10