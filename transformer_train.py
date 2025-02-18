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

from torch.utils.data import DataLoader
import os
import time
import torch
import torch.nn.functional as functional
import torch.optim as optim

from dataset import load_dataset
from settings import vocab_size, batch_size, learning_rate, epochs, model_path
from trainer import Trainer
from transformer import Transformer
from utils import format_number, load_transformer

dataset = load_dataset()
torch.autograd.set_detect_anomaly(True)

transformer = load_transformer(Transformer)

def criterion(output_probs, expected_indexes):

    return functional.cross_entropy(output_probs.reshape(-1, vocab_size), expected_indexes.reshape(-1))

def after_batch(trainer, firing):

    print(f'Epoch: {trainer.epoch_idx}, Batch: {trainer.batch_idx}, Epoch Loss: {format_number(trainer.epoch_loss)}, Batch Loss: {format_number(trainer.batch_loss)}')

    if firing:
        print("Saving model...")
        os.makedirs('transformers', exist_ok=True)
        torch.save(transformer.state_dict(), model_path)

def after_epoch(trainer, firing):

    print(f'Epoch: {trainer.epoch_idx}, Epoch Loss: {trainer.epoch_loss}')
    
    if firing:
        print("Saving model...")
        os.makedirs('transformers', exist_ok=True)
        torch.save(transformer.state_dict(), model_path)

trainer = Trainer(
    transformer,
    DataLoader(dataset, batch_size, shuffle=True),
    criterion,
    optim.AdamW(transformer.parameters(), lr=learning_rate),
    after_batch,
    after_epoch
)

trainer.fit(epochs)
