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
from settings import vocab_size, context_size, embedding_size, key_size, num_layers, layer_size, head_size, hidden_size

class Transformer(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.encoder = TokenEncoder()
        self.layers = nn.Sequential(*[TransformerLayer() for i in range(num_layers)])
        self.predictor = TokenPredictor()

    def forward(self, tokens):

        embeddings = self.encoder(tokens)

        embeddings = self.layers(embeddings)
            
        logits = self.predictor(embeddings if self.training else embeddings[:, -1, :])

        if self.training:
            return logits
        
        return functional.softmax(logits, dim=-1).cpu().detach()
        
    def generate(self, seed_tokens, num_new_tokens):

        self.eval()

        for i in range(num_new_tokens):

            input = torch.tensor(seed_tokens[-context_size:]).to(device).reshape(1, -1)
            output = self.forward(input).reshape(-1).numpy()

            new_token = np.random.choice(range(vocab_size), p=output)
            seed_tokens.append(new_token)
            yield new_token

class TokenEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding_table = nn.Embedding(vocab_size, embedding_size)

        positional_encoding = torch.zeros((context_size, embedding_size))

        for pos in range(context_size):
            for i in range(embedding_size):
                f = math.sin if i % 2 == 0 else math.cos
                positional_encoding[pos][i] = f(pos / 10000 ** (2 * i / embedding_size))

        self.register_buffer(
            "positional_encoding",
            positional_encoding
        )
        
    def forward(self, tokens):

        current_context_size = tokens.shape[1]
        assert(current_context_size <= context_size)

        return self.embedding_table(tokens) + self.positional_encoding[:current_context_size]
    
class TransformerLayer(nn.Module):
    def __init__(self):

        super().__init__()
            
        self.attention_norm = nn.LayerNorm((embedding_size,))
        self.heads = nn.ModuleList([
                SelfAttentionHead() for j in range(layer_size)
            ])

        self.mlp_norm = nn.LayerNorm((embedding_size,))
        self.mlp = nn.Sequential(
                nn.Linear(embedding_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, embedding_size)
            )

    def forward(self, embeddings):

        embeddings = self.attention_norm(embeddings)
        embeddings = embeddings + torch.cat([head(embeddings) for head in self.heads], dim=-1)
        embeddings = self.mlp_norm(embeddings)
        embeddings = embeddings + self.mlp(embeddings)

        return embeddings

class SelfAttentionHead(nn.Module):
    def __init__(self):

        super().__init__()

        self.key_maker = nn.Sequential(
            nn.Linear(embedding_size, key_size),
        )
        
        self.query_maker = nn.Sequential(
            nn.Linear(embedding_size, key_size),
        )
        
        self.value_maker = nn.Sequential(
            nn.Linear(embedding_size, head_size),
        )
      
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(context_size, context_size).float() * -torch.inf, 1)
        )

        assert(not torch.isnan(self.causal_mask).any())
        
    def forward(self, embeddings):

        current_context_size = embeddings.shape[1]
        assert(current_context_size <= context_size)

        keys = self.key_maker(embeddings) # B, Ck, K
        queries = self.query_maker(embeddings) # B, Cq, K
        values = self.value_maker(embeddings) # B, Ck, V

        attention_scores = torch.bmm(queries, keys.transpose(1, 2)) # B, Cq, K + B, K, Ck -> B, Cq, Ck
        attention_scores = attention_scores / math.sqrt(key_size)
      
        causal_mask = self.causal_mask[:current_context_size, :current_context_size]
        attention_scores = attention_scores + causal_mask

        attention_weights = functional.softmax(attention_scores, dim=-1) # B, Cq, Ck
        output = torch.bmm(attention_weights, values) # B, Cq, Ck + B, Ck, V -> B, Cq, V

        return output

class TokenPredictor(nn.Module):

    def __init__(self):
        super().__init__()

        self.predictor = nn.Sequential(
            nn.Linear(embedding_size, vocab_size)
        )
        self.predictor2 = nn.Softmax(dim=-1)

    def forward(self, embeddings):

        return self.predictor(embeddings)