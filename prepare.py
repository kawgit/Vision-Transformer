
import numpy as np
from settings import *
from tokenizer import Tokenizer
from utils import load_file_as_string, pickle_load


text = load_file_as_string(f"datasets/{dataset_name}.txt")
tokenizer = pickle_load(Tokenizer, f"tokenizers/{dataset_name}.pickle")

text_indexes = np.array(tokenizer.encode(text))

np.save(f"datasets/{dataset_name}", text_indexes)
