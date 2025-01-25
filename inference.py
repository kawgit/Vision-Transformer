from settings import dataset_name
from tokenizer import Tokenizer
from transformer import Transformer
from utils import load_transformer, pickle_load

text = """More worthier than their voices. They know the corn
Was not our recompense, resting well assured
That ne'er did service for't: being press'd to the war,
Even when the navel of the state was touch'd,
They would not thread the gates. This kind of service"""

transformer = load_transformer(Transformer)

tokenizer = pickle_load(Tokenizer, f"tokenizers/{dataset_name}.pickle")
text_tokens = tokenizer.encode(text)

for new_token in transformer.generate(text_tokens, 1000):
    text_tokens.append(new_token)
    print(tokenizer.decode([new_token]), end='')

print("\n" * 100)
print("Final Output:")
print(tokenizer.decode(text_tokens))