# peft/army_prefix_tuning.py

"""
Modified ArmyFootballPrefixTuning for a longer prefix and potentially more tokens.
We have updated the prefix_context in config and prefix_tokens to 50.
This module now just takes these parameters and creates a larger prefix.
"""

import torch
import torch.nn as nn

class ArmyFootballPrefixTuning(nn.Module):
    """
    Specialized prefix tuning with a personality text.
    Now with more prefix tokens and a longer, domain-specific personality text for stronger influence.
    This should guide the model heavily towards Army Football narratives.
    """
    def __init__(self, embed_dim, prefix_length, num_layers, tokenizer, personality_text, base_model_embeddings):
        super().__init__()
        self.prefix_length = prefix_length
        self.num_layers = num_layers

        # Encode the personality text into tokens.
        # If the text is shorter than prefix_length, pad with eos_token_ids.
        # If longer, truncate. Ideally, the text should be at least prefix_length tokens.
        tokens = tokenizer.encode(personality_text, add_special_tokens=False)
        if len(tokens) < prefix_length:
            tokens += [tokenizer.eos_token_id]*(prefix_length - len(tokens))
        else:
            tokens = tokens[:prefix_length]

        device = base_model_embeddings.weight.device
        tokens_tensor = torch.tensor([tokens], dtype=torch.long, device=device)

        with torch.no_grad():
            prefix_embeds = base_model_embeddings(tokens_tensor)  # [1, prefix_length, embed_dim]

        # Now we store this as a Parameter so it can be fine-tuned.
        self.prefix_embeddings = nn.Parameter(prefix_embeds.clone()[0])  # shape [prefix_length, embed_dim]

    def forward(self, x, layer_idx):
        # We apply the prefix only at the first layer to prepend prefix embeddings.
        # This ensures the prefix context is introduced at the very start of the model.
        if layer_idx == 0:
            batch_size = x.size(0)
            prefix = self.prefix_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
            return torch.cat([prefix, x], dim=1)
        else:
            return x
