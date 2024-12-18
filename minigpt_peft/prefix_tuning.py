# peft/prefix_tuning.py
import torch
import torch.nn as nn

class PrefixTuning(nn.Module):
    """
    Prefix Tuning: Prepend the same trainable prefix embeddings at layer 0 only.

    Previously, we had prefix embeddings per layer in a ParameterList, but that caused
    unused parameters for layers beyond 0. Here, we store a single prefix embedding parameter
    and apply it only at layer_idx=0. This ensures the prefix parameters are always used
    and get gradients.
    """
    def __init__(self, embed_dim, prefix_length, total_layers):
        super(PrefixTuning, self).__init__()
        self.prefix_length = prefix_length
        self.embed_dim = embed_dim
        self.total_layers = total_layers
        # A single prefix embedding parameter for layer 0 only
        self.prefix_embeddings = nn.Parameter(torch.randn(prefix_length, embed_dim) * 0.01)

    def forward(self, x, layer_idx):
        if layer_idx == 0:
            batch_size = x.size(0)
            prefix = self.prefix_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
            return torch.cat([prefix, x], dim=1)
        else:
            # No prefix added at other layers, avoiding unused parameters
            return x

def create_prefix_tuning_optimizer(model, optimizer_cls, optimizer_kwargs):
    """
    Creates an optimizer for Prefix Tuning, ensuring only prefix embeddings are updated.
    """
    prefix_params = [
        param for name, param in model.named_parameters() if "prefix_embeddings" in name and param.requires_grad
    ]

    if not prefix_params:
        raise ValueError("No trainable prefix parameters found.")

    return optimizer_cls(prefix_params, **optimizer_kwargs)
