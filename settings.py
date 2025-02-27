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
import math

dataset_name = "harrypotter1"

vocab_size = 10000

context_size = 500
key_size = 128
head_size = 128
layer_size = 8
hidden_size = 2048
embedding_size = head_size * layer_size
num_layers = 12

assert(key_size % 2 == 0)

batch_size = 16

lr_init = 3e-5
lr_max = 3e-4
lr_min = 2e-5
lr_decay = .9999
warmup_steps = 600 
period_steps = 5000
resume = True

def lambdalr(step):
    if step < warmup_steps:
        return (step / warmup_steps * (lr_max - lr_init) + lr_init)
    return (lr_max - lr_min) * (lr_decay ** (step - warmup_steps) * math.cos(math.pi * (step - warmup_steps) / period_steps) ** 2) + lr_min

model_name = f"{dataset_name}_c{context_size}_m{embedding_size}_k{key_size}_h{head_size}_l{num_layers}_v{hidden_size}"
checkpoint_path = os.path.join('checkpoints', model_name + '.checkpoint')
tokenizer_path = f"tokenizers/{dataset_name}.pickle"
