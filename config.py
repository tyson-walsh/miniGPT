# config.py

"""
Defines a configuration dataclass and a seed-setting function for training a MiniGPT model.
Specifies all model hyperparameters, optimization parameters, paths, and optional modes.
Now updated to extend training epochs, lower learning rate, and increase prefix tokens for prefix tuning,
as well as enable gradient checkpointing by default for better memory efficiency.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
import torch
import random
import numpy as np
from transformers import GPT2Tokenizer

def set_seed(seed: int = 142):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@dataclass
class MiniGPTConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    data_path: Path = Path("data")
    save_path: Path = Path("models")

    tokenizer_name: str = "gpt2"
    batch_size: int = 64  # kept 64, but can adjust if memory issues arise
    epochs: int = 10  # Increased from 3 to 10 for more training time
    learning_rate: float = 5e-5  # Lowered from 3e-4 to 5e-5 for more stable training
    weight_decay: float = 0.01
    grad_accum_steps: int = 1
    mixed_precision: bool = False
    num_workers: int = 8

    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    ff_dim: int = 512
    dropout: float = 0.1
    context_length: int = 512

    tokenizer: GPT2Tokenizer = field(init=False)
    vocab_size: int = field(init=False)

    peft_enabled: bool = False
    peft_variant: Optional[str] = None
    peft_variants: List[str] = field(default_factory=lambda: [
        "base", "bias", "lora", "lora_plus", "adapter", "prefix_default", "prefix_army_football"
    ])

    lora_r: int = 8  # Increased rank from 4 to 8 to allow more expressive low-rank adaptation
    lora_alpha: int = 16  # Increased alpha for stronger scaling
    lora_dropout: float = 0.1
    lora_plus_lr_ratio: float = 10.0
    lora_plus_lr_embedding: float = 5e-4

    prefix_tokens: int = 0
    prefix_context: str = "Default prefix context"

    adapter_hidden_dim: int = 64
    adapter_enabled: bool = False
    bias_tuning: bool = False

    dev_mode: bool = False
    train_batch_limit: Optional[int] = None
    val_batch_limit: Optional[int] = None
    gradient_checkpointing: bool = True  # Enabled by default now

    # Early stopping patience: number of epochs to wait for improvement in val loss.
    early_stopping_patience: int = 3  # If val loss does not improve for 3 epochs, stop early.

    def __post_init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = self.tokenizer.vocab_size

        if os.getenv("DEV_MODE", "false").lower() == "true":
            self.dev_mode = True
            self.epochs = int(os.getenv("EPOCHS", "3"))  # If in dev_mode, can still override to fewer epochs
            self.batch_size = int(os.getenv("BATCH_SIZE", "16"))
            self.mixed_precision = True
            self.grad_accum_steps = 2
            self.num_workers = 2
            tb_limit = os.getenv("TRAIN_BATCH_LIMIT", "")
            vb_limit = os.getenv("VAL_BATCH_LIMIT", "")
            self.train_batch_limit = int(tb_limit) if tb_limit.isdigit() else None
            self.val_batch_limit = int(vb_limit) if vb_limit.isdigit() else None
            self.gradient_checkpointing = True

        env_save_path = os.getenv("SAVE_PATH")
        if env_save_path is not None:
            self.save_path = Path(env_save_path)

    def update_for_peft(self, variant: str):
        self.peft_variant = variant
        self.peft_enabled = True
        self.adapter_enabled = False
        self.bias_tuning = False
        self.prefix_tokens = 0

        if variant == "base":
            self.peft_enabled = False
        elif variant == "bias":
            self.bias_tuning = True
        elif variant == "lora":
            # LoRA parameters already adjusted above.
            pass
        elif variant == "lora_plus":
            # LoRA+ uses same parameters but can differ in LR ratio.
            pass
        elif variant == "adapter":
            self.adapter_enabled = True
            self.adapter_hidden_dim = 64
        elif variant == "prefix_default":
            self.prefix_tokens = 50  # Increased from 10 to 50 for stronger prefix influence
        elif variant == "prefix_army_football":
            self.prefix_tokens = 50  # More prefix tokens
            # A longer, more domain-specific prefix context emphasizing Army football
            self.prefix_context = (
                "As an avid Army Football fan and proud West Point graduate, I recall the thunderous cheers "
                "echoing through Michie Stadium on crisp fall afternoons. The 'Beat Navy!' chants, the precision "
                "of the Corps of Cadets marching on the field, and the grit and determination of our Black Knights "
                "define the legacy of Army Football. Each snap, each tackle, and each disciplined drive reflects "
                "the honor, courage, and commitment that stand at the core of West Point's values. "
            )
        else:
            raise ValueError(f"Unknown PEFT variant: {variant}")
