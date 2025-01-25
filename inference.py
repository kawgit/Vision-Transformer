from settings import *
from transformer import Transformer
from tokenizer import Tokenizer
from utils import load_transformer

text = """More worthier than their voices. They know the corn
Was not our recompense, resting well assured
That ne'er did service for't: being press'd to the war,
Even when the navel of the state was touch'd,
They would not thread the gates. This kind of service"""

transformer = load_transformer(Transformer)
text_bytes = text.encode()

for token_bytes in transformer.generate(text, 100):
    text_bytes += token_bytes
    print(text_bytes.decode())