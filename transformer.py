#    Copyright 2025 Kenneth Wilber (kawgit)

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

from device import device
from settings import *

class Transformer(nn.Module):

    def __init__(self):
        super().__init__()

        self.register_buffer(
            "causal_buffer",
            torch.triu(torch.ones(context_size, context_size).float() * -torch.inf, 1)
        )

        self.register_buffer(
            "cos_buffer",
            torch.tensor(np.array([
                [
                    math.cos(m * 10000 ** ((-2 / key_size) * (i % (key_size // 2)))) for i in range(key_size)
                ] for m in range(context_size)
            ])).float()
        )

        self.register_buffer(
            "sin_buffer",
            torch.tensor(np.array([
                [
                    math.sin(m * 10000 ** ((-2 / key_size) * (i % (key_size // 2)))) * (-1 if i < key_size // 2 else 1) for i in range(key_size)
                ] for m in range(context_size)
            ])).float()
        )

        self.table = nn.Embedding(vocab_size, embedding_size)
        self.layers = nn.Sequential(*[TransformerLayer() for i in range(num_layers)])
        self.predictor = nn.Linear(embedding_size, vocab_size)

    def forward(self, tokens):

        current_context_size = tokens.shape[1]
        assert(current_context_size <= context_size)

        causal_buffer = self.causal_buffer[:current_context_size, :current_context_size]
        cos_buffer = self.cos_buffer[:current_context_size, :]
        sin_buffer = self.sin_buffer[:current_context_size, :]

        embeddings = self.table(tokens)
    
        for layer in self.layers:
            embeddings = layer(embeddings, causal_buffer, cos_buffer, sin_buffer)

        logits = self.predictor(embeddings if self.training else embeddings[:, -1, :])

        return logits if self.training else functional.softmax(logits, dim=-1).cpu().detach()

    def generate(self, seed_tokens, num_new_tokens):

        self.eval()

        for i in range(num_new_tokens):

            input = torch.tensor(seed_tokens[-context_size:]).to(device).reshape(1, -1)
            output = self.forward(input).reshape(-1).numpy()

            new_token = np.random.choice(range(vocab_size), p=output)
            seed_tokens.append(new_token)
            yield new_token

class TransformerLayer(nn.Module):
    def __init__(self):

        super().__init__()

        self.attention_norm = nn.RMSNorm(embedding_size)
        self.attention_heads = nn.ModuleList([SelfAttentionHead() for j in range(layer_size)])
        self.attention_proj = nn.Linear(embedding_size, embedding_size, bias=False)

        self.mlp_norm = nn.RMSNorm(embedding_size)
        self.mlp_upward = nn.Linear(embedding_size, hidden_size, bias=False)
        self.mlp_gate = nn.Sequential(
                nn.Linear(embedding_size, hidden_size, bias=False),
                nn.ELU()
            )
        self.mlp_downward = nn.Linear(hidden_size, embedding_size, bias=False)

    def forward(self, embeddings, causal_buffer, cos_buffer, sin_buffer):

        attention_input = self.attention_norm(embeddings)
        attention_values = torch.cat([
                head(embeddings, causal_buffer, cos_buffer, sin_buffer) for head in self.attention_heads
            ], dim=-1)
        embeddings = embeddings + self.attention_proj(attention_values)
        
        mlp_input = self.mlp_norm(embeddings)
        mlp_up = self.mlp_upward(mlp_input) * self.mlp_gate(mlp_input)
        embeddings = embeddings + self.mlp_downward(mlp_up)

        return embeddings

class SelfAttentionHead(nn.Module):
    def __init__(self):

        super().__init__()

        self.key_maker = nn.Linear(embedding_size, key_size, bias=False)
        self.query_maker = nn.Linear(embedding_size, key_size, bias=False)
        self.value_maker = nn.Linear(embedding_size, head_size, bias=False)

    def forward(self, embeddings, causal_buffer, cos_buffer, sin_buffer):

        keys = self.rotate(self.key_maker(embeddings), cos_buffer, sin_buffer) # B, Ck, K
        queries = self.rotate(self.query_maker(embeddings), cos_buffer, sin_buffer) # B, Cq, K
        values = self.value_maker(embeddings) # B, Ck, V

        attention_scores = torch.bmm(queries, keys.transpose(1, 2)) # B, Cq, K + B, K, Ck -> B, Cq, Ck
        attention_scores = attention_scores / math.sqrt(key_size)

        attention_scores = attention_scores + causal_buffer

        attention_weights = functional.softmax(attention_scores, dim=-1) # B, Cq, Ck
        output = torch.bmm(attention_weights, values) # B, Cq, Ck + B, Ck, V -> B, Cq, V

        return output

    def rotate(self, embeddings, cos_buffer, sin_buffer):
        
        swapped = torch.cat((embeddings[:, :, key_size // 2:], embeddings[:, :, :key_size // 2]), dim=-1)

        return cos_buffer * embeddings + sin_buffer * swapped

def load_transformer():

    transformer = Transformer()

    if resume and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}.")

        checkpoint = torch.load(checkpoint_path, weights_only=True)
        transformer.load_state_dict(checkpoint["transformer"])

    return transformer
