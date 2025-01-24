import pickle

def load_file_as_string(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
    
def pickle_save(thing, path):
    with open(path, 'wb') as handle:
        pickle.dump(thing, handle, protocol=pickle.HIGHEST_PROTOCOL)

def pickle_load(path):
    with open(path, 'rb') as handle:
        thing = pickle.load(handle)
    return thing
