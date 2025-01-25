import numpy as np
import torch

from device import device
from settings import *
from tokenizer import Tokenizer
from transformer import Transformer
from utils import make_or_load_model, pickle_load

transformer = make_or_load_model(Transformer)
transformer.eval()

tokenizer = pickle_load(Tokenizer, f"tokenizers/{dataset_name}.pickle")

seed = """More worthier than their voices. They know the corn
Was not our recompense, resting well assured
That ne'er did service for't: being press'd to the war,
Even when the navel of the state was touch'd,
They would not thread the gates. This kind of service"""
text_indexes = tokenizer.encode(seed)

print(seed, end='')

for i in range(1000):

    probs = transformer.forward(torch.tensor(text_indexes[-context_size:]).to(device).reshape(1, -1), inference=True).reshape(-1).detach().numpy()

    new_index = np.random.choice(range(vocab_size), p=probs)

    text_indexes.append(new_index)

    print(tokenizer.decode([new_index]), end='')