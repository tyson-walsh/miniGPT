# peft/lora_plus.py
import torch
import torch.nn as nn

class LoRAPlus(nn.Module):
    """
    LoRAPlus: Similar to LoRA, but includes a different learning rate ratio and possibly embeddings.
    """
    def __init__(self, base_module, rank=4, alpha=1, lr_ratio=0.1):
        super(LoRAPlus, self).__init__()
        self.base_module = base_module
        self.rank = rank
        self.alpha = alpha
        self.lr_ratio = lr_ratio

        weight = base_module.weight
        out_features, in_features = weight.shape
        self.lora_A = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.scaling = self.alpha / self.rank

    def forward(self, x):
        batch_size, seq_length, in_features = x.size()
        out_features = self.lora_A.size(0)
        original_out = self.base_module(x)

        x_2d = x.view(-1, in_features)
        lora_intermediate = torch.matmul(x_2d, self.lora_B.T)
        lora_adjustment_2d = torch.matmul(lora_intermediate, self.lora_A.T) * self.scaling
        lora_adjustment = lora_adjustment_2d.view(batch_size, seq_length, out_features)

        return original_out + lora_adjustment

def create_loraplus_optimizer(model, optimizer_cls, optimizer_kwargs, loraplus_lr_ratio, loraplus_lr_embedding=None):
    """
    Creates an optimizer configured for LoRAPlus. Differentiates LR for LoRA B params and embeddings if needed.
    """
    if loraplus_lr_embedding is None:
        loraplus_lr_embedding = 1e-6

    groupA_params = []
    groupB_params = []
    embedding_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_B" in name:
            groupB_params.append(param)
        elif "embedding" in name:
            embedding_params.append(param)
        else:
            groupA_params.append(param)

    optimizer_grouped_parameters = [
        {
            "params": groupA_params,
            "weight_decay": optimizer_kwargs.get("weight_decay", 0.01),
            "lr": optimizer_kwargs["lr"],
        },
        {
            "params": groupB_params,
            "weight_decay": 0.0,
            "lr": optimizer_kwargs["lr"] * loraplus_lr_ratio,
        },
        {
            "params": embedding_params,
            "weight_decay": 0.0,
            "lr": loraplus_lr_embedding,
        },
    ]

    return optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
