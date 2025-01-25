import glob
import os
import pickle
import torch

from device import device

def load_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
    
def pickle_save(thing, path):
    with open(path, 'wb') as handle:
        pickle.dump(thing, handle, protocol=pickle.HIGHEST_PROTOCOL)

def pickle_load(thing_class, path):
    with open(path, 'rb') as handle:
        thing = pickle.load(handle)
    return thing

def format_number(number, total_length=7):
    return f"{number:0{total_length}.6f}"

def get_last_file(directory_path):
    files = glob.glob(os.path.join(directory_path, '*'))
    files.sort()
    return files[-1] if files else None

def load_transformer(model_class, *args, **kwargs):

    model = model_class(*args, **kwargs)
    model_path = get_last_file('transformers')

    if model_path:
        print(f"Loading weights from {model_path}.")
        model.load_state_dict(torch.load(model_path, weights_only=True), strict=False)
    
    return model.to(device)