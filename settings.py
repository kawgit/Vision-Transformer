
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