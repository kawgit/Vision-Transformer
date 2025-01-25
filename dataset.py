from torch.utils.data import Dataset
import numpy as np
import os
import torch

from settings import context_size, dataset_name
from tokenizer import Tokenizer
from utils import load_file, pickle_load

class TransformerDataset(Dataset):

    def __init__(self, text_indexes):

        self.text_indexes = text_indexes

    def __len__(self):

        return len(self.text_indexes) - context_size

    def __getitem__(self, idx):
        return self.text_indexes[idx:idx+context_size], self.text_indexes[idx+1:idx+context_size+1]
    

def load_dataset():

    if os.path.exists(f"datasets/{dataset_name}.npy"):
        text_indexes = torch.tensor(np.load(f"datasets/{dataset_name}.npy"))
    else:
        text = load_file(f"datasets/{dataset_name}.txt")
        tokenizer = pickle_load(Tokenizer, f"tokenizers/{dataset_name}.pickle")
        text_indexes = torch.tensor(tokenizer.encode(text))

    return TransformerDataset(text_indexes)