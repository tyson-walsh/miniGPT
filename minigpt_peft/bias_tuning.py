# peft/bias_tuning.py
import torch
import torch.nn as nn

class BiasTuning(nn.Module):
    """
    Bias Tuning: Freeze all parameters except biases.
    """
    def __init__(self, base_model):
        super(BiasTuning, self).__init__()
        self.base_model = base_model
        # Freezing handled externally; this wrapper is to clarify architecture if needed.

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

def create_bias_tuning_optimizer(model, optimizer_cls, optimizer_kwargs):
    """
    Creates an optimizer for bias tuning, ensuring only bias parameters are updated.
    """
    bias_params = [
        param for name, param in model.named_parameters() if "bias" in name and param.requires_grad
    ]

    if not bias_params:
        raise ValueError("No trainable bias parameters found for bias tuning.")

    return optimizer_cls(bias_params, **optimizer_kwargs)
