# peft/lora.py

"""
LoRA module, previously just adapted the output layer.
No major logic change required here since we now call it multiple times from train.py.

We keep the code as is, but the calling code in train.py now applies LoRA to multiple layers.
We might just add more comments explaining how LoRA is now also used on attention layers.
"""

import torch
import torch.nn as nn

class LoRA(nn.Module):
    """
    LoRA: Low-Rank Adaptation of Transformers.
    Adds low-rank matrices A and B to a linear layer for efficient fine-tuning.

    Note: With expanded usage in train.py, LoRA can be applied not only to output_head,
    but also to attention projections (query/value) for more expressive adaptation.
    """
    def __init__(self, base_module, rank=4, alpha=1):
        super(LoRA, self).__init__()
        self.base_module = base_module
        self.rank = rank
        self.alpha = alpha

        weight = base_module.weight  # [out_features, in_features]
        out_features, in_features = weight.shape
        self.lora_A = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.scaling = alpha / rank

    def forward(self, x):
        # x: [batch_size, seq_len, in_features]
        batch_size, seq_length, in_features = x.size()
        out_features = self.lora_A.size(0)
        original_out = self.base_module(x)  # [batch_size, seq_len, out_features]

        x_2d = x.view(-1, in_features)
        lora_intermediate = torch.matmul(x_2d, self.lora_B.T)        # [N, rank]
        lora_adjustment_2d = torch.matmul(lora_intermediate, self.lora_A.T) * self.scaling  # [N, out_features]
        lora_adjustment = lora_adjustment_2d.view(batch_size, seq_length, out_features)

        return original_out + lora_adjustment

    def named_parameters(self, recurse=True):
        # Return both the base module parameters and LoRA parameters.
        # Base module is frozen, so only LoRA parameters effectively update.
        for name, param in self.base_module.named_parameters(recurse=recurse):
            yield name, param
        yield "lora_A", self.lora_A
        yield "lora_B", self.lora_B
