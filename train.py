from torch.utils.data import DataLoader
import os
import time
import torch
import torch.nn.functional as functional
import torch.optim as optim

from dataset import load_dataset
from settings import *
from trainer import Trainer
from transformer import Transformer
from utils import format_number, load_transformer

dataset = load_dataset()

transformer = load_transformer(Transformer)

def criterion(output_probs, expected_indexes):

    return functional.cross_entropy(output_probs.reshape(-1, vocab_size), expected_indexes.reshape(-1))

def after_batch(trainer, firing):

    print(f'Epoch: {trainer.epoch_idx}, Batch: {trainer.batch_idx}, Epoch Loss: {format_number(trainer.epoch_loss)}, Batch Loss: {format_number(trainer.batch_loss)}')

    if firing:
        os.makedirs('transformers', exist_ok=True)
        torch.save(transformer.state_dict(), os.path.join('transformers', f"{round(time.time())}_{format_number(trainer.epoch_loss)}.pt"))

def after_epoch(trainer, firing):

    print(f'Epoch: {trainer.epoch_idx}, Epoch Loss: {trainer.epoch_loss}')
    
    if firing:
        os.makedirs('transformers', exist_ok=True)
        torch.save(transformer.state_dict(), os.path.join('transformers', f"{round(time.time())}_{format_number(trainer.epoch_loss)}.pt"))

trainer = Trainer(
    transformer,
    DataLoader(dataset, batch_size, shuffle=True),
    criterion,
    optim.AdamW(transformer.parameters(), lr=3e-4),
    after_batch,
    after_epoch
)

trainer.fit(10)
