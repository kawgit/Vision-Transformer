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
from settings import vocab_size, context_size, embedding_size, key_size, num_layers, layer_size, head_size

class Transformer(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.encoder = TokenEncoder()

        self.heads = nn.ModuleList([
                nn.ModuleList([
                    SelfAttentionHead() for j in range(layer_size)
                ]) for i in range(num_layers)
            ])
        
        self.perceptrons = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(embedding_size),
                    nn.Linear(embedding_size, embedding_size),
                    nn.ReLU(),
                    nn.Linear(embedding_size,embedding_size),
                    nn.ReLU()
                ) for i in range(num_layers)
            ])
        
        self.predictor = TokenPredictor()

    def _verify(self):
        self.encoder._verify()
        for heads in self.heads:
            for head in heads:
                head._verify()
        self.predictor._verify()

    def forward(self, tokens, inference=False):

        self._verify()

        embeddings = self.encoder(tokens)

        for heads, perceptron in zip(self.heads, self.perceptrons):

            embeddings = perceptron(embeddings + torch.cat([head(embeddings) for head in heads], dim=-1))
            
        logits = self.predictor(embeddings if not inference else embeddings[:, -1, :])

        if not inference:
            return logits
        
        return functional.softmax(logits, dim=-1).cpu().detach()
        
    def generate(self, seed_tokens, num_new_tokens):

        self.eval()

        for i in range(num_new_tokens):

            input = torch.tensor(seed_tokens[-context_size:]).to(device).reshape(1, -1)
            output = self.forward(input, inference=True).reshape(-1).numpy()

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

    def _verify(self):
        
        assert(self.embedding_table.weight.shape == (vocab_size, embedding_size))
        assert(self.positional_encoding.shape == (context_size, embedding_size))
        
    def forward(self, tokens):

        current_context_size = tokens.shape[1]
        assert(current_context_size <= context_size)

        return self.embedding_table(tokens) + self.positional_encoding[:current_context_size]
    
class SelfAttentionHead(nn.Module):
    def __init__(self):

        super().__init__()

        self.key_maker = nn.Sequential(
            nn.Linear(embedding_size, key_size),
            nn.ReLU(),
            nn.Linear(key_size, key_size),
            nn.ReLU()
        )
        
        self.query_maker = nn.Sequential(
            nn.Linear(embedding_size, key_size),
            nn.ReLU(),
            nn.Linear(key_size, key_size),
            nn.ReLU()
        )
        
        self.value_maker = nn.Sequential(
            nn.Linear(embedding_size, head_size)
        )
      
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(context_size, context_size)).float() * -10000000000
        )

    def _verify(self):

        assert(self.key_maker[0].weight.shape == (key_size, embedding_size))
        assert(self.query_maker[0].weight.shape == (key_size, embedding_size))
        assert(self.value_maker[0].weight.shape == (head_size, embedding_size))
        
    def forward(self, embeddings):

        current_context_size = embeddings.shape[1]
        assert(current_context_size <= context_size)

        keys = self.key_maker(embeddings)
        queries = self.query_maker(embeddings)
        values = self.value_maker(embeddings)

        attention_scores = torch.bmm(queries, keys.transpose(1, 2))
        attention_scores = attention_scores / (key_size ** 0.5)
      
        causal_mask = self.causal_mask[:current_context_size, :current_context_size]
        attention_scores = attention_scores + causal_mask

        attention_weights = functional.softmax(attention_scores, dim=-1)

        output = torch.bmm(attention_weights, values)

        return output

class TokenPredictor(nn.Module):

    def __init__(self):
        super().__init__()

        self.predictor = nn.Sequential(
            nn.Linear(embedding_size, vocab_size)
        )
        self.predictor2 = nn.Softmax(dim=-1)

    def _verify(self):

        assert(self.predictor[0].weight.shape == (vocab_size, embedding_size))

    def forward(self, embeddings):

        return self.predictor(embeddings)