# peft/adapter_tuning.py
import torch
import torch.nn as nn

class AdapterLayer(nn.Module):
    """
    Adapter Layer with a bottleneck structure.
    """
    def __init__(self, input_dim, bottleneck_dim=64):
        super(AdapterLayer, self).__init__()
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, input_dim)

    def forward(self, x):
        return x + self.up_proj(torch.relu(self.down_proj(x)))

class AdapterTuningTransformerLayer(nn.Module):
    """
    A Transformer layer that includes an AdapterLayer for efficient tuning.
    This is a stand-in for the original TransformerLayer, adding an adapter.
    We'll assume the base model has a similar structure and we wrap it.
    """
    def __init__(self, base_layer, bottleneck_dim=64):
        super().__init__()
        # base_layer: a TransformerLayer instance from model.py
        self.base_layer = base_layer
        embed_dim = base_layer.norm1.normalized_shape[0]
        self.adapter = AdapterLayer(embed_dim, bottleneck_dim)

    def forward(self, x):
        # Run the original transformer layer
        x = self.base_layer(x)
        # Add adapter on top
        x = self.adapter(x)
        return x

def insert_adapters(model, bottleneck_dim=64):
    """
    Inserts adapters into each TransformerLayer of the given model.
    This modifies the model in place, turning each layer into an AdapterTuningTransformerLayer.
    """
    for i in range(len(model.layers)):
        base_layer = model.layers[i]
        model.layers[i] = AdapterTuningTransformerLayer(base_layer, bottleneck_dim=bottleneck_dim)

def create_adapter_optimizer(model, optimizer_cls, optimizer_kwargs):
    """
    Creates an optimizer for adapter tuning, ensuring only adapter parameters are updated.
    """
    adapter_params = [
        param for name, param in model.named_parameters() if "down_proj" in name or "up_proj" in name
    ]

    if not adapter_params:
        raise ValueError("No trainable adapter parameters found.")

    return optimizer_cls(adapter_params, **optimizer_kwargs)
