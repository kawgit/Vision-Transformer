from transformer import Transformer

model = Transformer()

count = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"This model has {count} parameters")