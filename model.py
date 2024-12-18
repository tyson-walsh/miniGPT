# model.py

"""
This file defines a transformer-based language model called MiniGPT. The architecture includes:
- Token embeddings: Converts integer token IDs into continuous embeddings of shape [batch_size, seq_len, embed_dim].
- Positional embeddings: Adds positional information to each token embedding, producing a tensor of shape [batch_size, seq_len, embed_dim].
- Stacked Transformer layers: Each layer includes multi-head self-attention and feed-forward sub-layers.
- A linear projection head from hidden states to vocabulary logits: Maps [batch_size, seq_len, embed_dim] to [batch_size, seq_len, vocab_size].

Shapes and Dimensions:
- Input token IDs: [batch_size, seq_len]
- Token embeddings: [batch_size, seq_len, embed_dim]
- Position embeddings: [1, seq_len, embed_dim]
- After adding positional embeddings: [batch_size, seq_len, embed_dim]
- MultiHeadAttention splits embed_dim into num_heads, each of dimension head_dim = embed_dim / num_heads.
- Output head: transforms [batch_size, seq_len, embed_dim] into [batch_size, seq_len, vocab_size].

Why and how it works:
A language model predicts the next token given previous tokens. The transformer architecture uses self-attention 
to aggregate information from all previous positions. The causal mask ensures no leakage of future tokens. 
The final linear layer maps hidden states to probability distributions over the vocabulary.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        # Multi-head attention projects the input x of shape [batch_size, seq_len, embed_dim]
        # into queries, keys, and values each with shape [batch_size, num_heads, seq_len, head_dim].
        # head_dim = embed_dim / num_heads
        # The attention operation is QK^T / sqrt(head_dim) producing a [batch_size, num_heads, seq_len, seq_len] scores matrix.
        # A causal mask ensures that only previous tokens are attended to.
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)  # Produces Q of shape [batch_size, seq_len, embed_dim]
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)    # Produces K of shape [batch_size, seq_len, embed_dim]
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)  # Produces V of shape [batch_size, seq_len, embed_dim]
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch_size, seq_len, embed_dim]
        bsz, seq_len, embed_dim = x.size()
        # Q, K, V are computed by linear projections and then reshaped into multiple heads.
        q = self.query(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        k = self.key(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)    # same shape as q
        v = self.value(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # same shape as q

        # Create a causal mask of shape [seq_len, seq_len] with True where future positions should be masked.
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

        # Compute attention scores as QK^T / sqrt(head_dim)
        scores = (q @ k.transpose(-2, -1)) / self.scale  # [batch_size, num_heads, seq_len, seq_len]
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Apply softmax along the seq_len dimension of the last axis to get probabilities
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Multiply probabilities by values: attn_weights @ v -> [batch_size, num_heads, seq_len, head_dim]
        attn_output = (attn_weights @ v).transpose(1, 2).contiguous().view(bsz, seq_len, embed_dim)

        # Project back to original embed_dim
        attn_output = self.out_proj(attn_output)
        return attn_output

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        # A transformer layer includes:
        # - MultiHeadAttention followed by residual connection and LayerNorm
        # - FeedForward network followed by residual connection and LayerNorm
        # The feedforward network maps [batch_size, seq_len, embed_dim] to [batch_size, seq_len, embed_dim] through an intermediate dimension ff_dim.
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # The input x has shape [batch_size, seq_len, embed_dim].
        # After attention, a residual connection adds the result to x.
        # Layer normalization stabilizes training and reduces covariate shift.
        attn_output = self.attention(x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # The feedforward network processes each token independently.
        ff_output = self.ff(x)  # [batch_size, seq_len, embed_dim]
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        return x

class MiniGPT(nn.Module):
    def __init__(self, config):
        # The model creates token embeddings of shape [batch_size, seq_len, embed_dim]
        # and positional embeddings of the same shape but broadcasted over batch_size.
        # The embeddings are summed and passed through multiple TransformerLayers.
        # Finally, a layer normalization and a linear head map to vocab_size logits.
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([
            TransformerLayer(config.embed_dim, config.num_heads, config.ff_dim, config.dropout)
            for _ in range(config.num_layers)
        ])
        self.layer_norm = nn.LayerNorm(config.embed_dim)
        self.output_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.prefix_tuning = None

    def forward(self, input_ids):
        # input_ids: [batch_size, seq_len], containing integer token IDs.
        # Embeddings: token_embedding: [batch_size, seq_len, embed_dim]
        # Positions: [seq_len] -> position_embedding: [1, seq_len, embed_dim] broadcast to [batch_size, seq_len, embed_dim]

        bsz, seq_len = input_ids.size()
        token_embeds = self.token_embedding(input_ids)
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        pos_embeds = self.position_embedding(positions)
        x = self.dropout(token_embeds + pos_embeds)  # [batch_size, seq_len, embed_dim]

        # If prefix_tuning is enabled, it may modify x at each layer. 
        for i, layer in enumerate(self.layers):
            if self.prefix_tuning is not None:
                x = self.prefix_tuning(x, i)
            x = layer(x)

        # Apply a final layer norm before projecting to vocab logits.
        x = self.layer_norm(x)  # [batch_size, seq_len, embed_dim]
        logits = self.output_head(x)  # [batch_size, seq_len, vocab_size]
        return logits

def build_model(config):
    return MiniGPT(config)
