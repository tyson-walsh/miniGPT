# train.py

"""
Training procedure for MiniGPT with enhancements:
- Longer training (e.g., 10 epochs).
- Early stopping with patience.
- Adjusted LoRA application to include attention layers (Q and V projections).
- Lowered LR for stability.
- These changes aim to improve model quality and output coherence.
"""

import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from config import MiniGPTConfig, set_seed
from dataset import TinyStoriesDataset
from model import build_model
from pathlib import Path
import argparse
import logging

# Importing PEFT methods:
from minigpt_peft.lora import LoRA
from minigpt_peft.lora_plus import LoRAPlus
from minigpt_peft.bias_tuning import BiasTuning
from minigpt_peft.prefix_tuning import PrefixTuning
from minigpt_peft.army_prefix_tuning import ArmyFootballPrefixTuning
from minigpt_peft.adapter_tuning import insert_adapters

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def freeze_all_parameters(model):
    for p in model.parameters():
        p.requires_grad = False

def unfreeze_bias_parameters(model):
    for name, p in model.named_parameters():
        if "bias" in name:
            p.requires_grad = True

def unfreeze_adapter_parameters(model):
    for name, p in model.named_parameters():
        if "down_proj" in name or "up_proj" in name:
            p.requires_grad = True

def wrap_and_unfreeze_lora(model, config):
    """
    Previously, we only applied LoRA to the output_head.
    Now, we apply LoRA to both the output_head and the attention projections
    (query and value) in each Transformer layer for broader adaptation.

    Steps:
    - Freeze all parameters initially.
    - Wrap the output_head with LoRA as before.
    - Also wrap the attention query and value linear layers in each Transformer layer.
    - Unfreeze the LoRA parameters so they can be trained.

    This should give LoRA more capacity to adapt the model's internal representations.
    """

    freeze_all_parameters(model)

    # Wrap output_head:
    model.output_head = LoRA(model.output_head, rank=config.lora_r, alpha=config.lora_alpha)

    # For each layer, wrap query and value projections in LoRA:
    for layer in model.layers:
        # layer.attention.query, layer.attention.value, and layer.attention.key
        # The MultiHeadAttention class in model.py doesn't store separate q, k, v as attributes,
        # but we can modify code here assuming they are linear layers named query, key, value.
        # If they are not separate attributes, we must modify the model to expose them, or wrap them directly.

        # Let's assume model.attention has query/key/value as linear layers:
        # If we need them, we can do:
        layer.attention.query = LoRA(layer.attention.query, rank=config.lora_r, alpha=config.lora_alpha)
        layer.attention.value = LoRA(layer.attention.value, rank=config.lora_r, alpha=config.lora_alpha)
        # We could also wrap key if desired, but often LoRA on query/value is enough.
        # layer.attention.key = LoRA(layer.attention.key, rank=config.lora_r, alpha=config.lora_alpha)

    # Now unfreeze LoRA parameters:
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad = True

def wrap_and_unfreeze_lora_plus(model, config):
    """
    Similar to wrap_and_unfreeze_lora, but uses LoRAPlus.
    We'll do the same expansions to attention layers.
    """
    freeze_all_parameters(model)
    model.output_head = LoRAPlus(model.output_head, rank=config.lora_r, alpha=config.lora_alpha, lr_ratio=config.lora_plus_lr_ratio)

    for layer in model.layers:
        layer.attention.query = LoRAPlus(layer.attention.query, rank=config.lora_r, alpha=config.lora_alpha, lr_ratio=config.lora_plus_lr_ratio)
        layer.attention.value = LoRAPlus(layer.attention.value, rank=config.lora_r, alpha=config.lora_alpha, lr_ratio=config.lora_plus_lr_ratio)

    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad = True

def apply_prefix_tuning_(model, config):
    freeze_all_parameters(model)
    model.prefix_tuning = PrefixTuning(config.embed_dim, config.prefix_tokens, config.num_layers)
    for p in model.prefix_tuning.parameters():
        p.requires_grad = True

def apply_army_prefix_tuning_(model, config):
    freeze_all_parameters(model)
    model.prefix_tuning = ArmyFootballPrefixTuning(
        config.embed_dim,
        config.prefix_tokens,
        config.num_layers,
        config.tokenizer,
        config.prefix_context,
        model.token_embedding
    )
    for p in model.prefix_tuning.parameters():
        p.requires_grad = True

def apply_bias_tuning_(model):
    freeze_all_parameters(model)
    unfreeze_bias_parameters(model)

def apply_adapter_tuning_(model, config):
    freeze_all_parameters(model)
    insert_adapters(model, bottleneck_dim=config.adapter_hidden_dim)
    unfreeze_adapter_parameters(model)

def apply_peft(model, config):
    if not config.peft_enabled or config.peft_variant == "base":
        return model
    elif config.bias_tuning:
        apply_bias_tuning_(model)
        model = BiasTuning(model)
    elif config.peft_variant == "lora":
        wrap_and_unfreeze_lora(model, config)
    elif config.peft_variant == "lora_plus":
        wrap_and_unfreeze_lora_plus(model, config)
    elif config.peft_variant == "adapter":
        apply_adapter_tuning_(model, config)
    elif config.peft_variant == "prefix_default":
        apply_prefix_tuning_(model, config)
    elif config.peft_variant == "prefix_army_football":
        apply_army_prefix_tuning_(model, config)
    return model

def run_epoch(model, dataloader, criterion, optimizer, scheduler, scaler, device, config, epoch, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    steps = 0

    prefix_length = config.prefix_tokens if (config.peft_variant and config.peft_variant.startswith("prefix")) else 0
    start_time = time.time()

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(inputs)
                    if prefix_length > 0:
                        seq_len_out = outputs.shape[1] - prefix_length
                        outputs = outputs[:, prefix_length:, :]
                        targets = targets[:, :seq_len_out]
                    loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
            else:
                outputs = model(inputs)
                if prefix_length > 0:
                    seq_len_out = outputs.shape[1] - prefix_length
                    outputs = outputs[:, prefix_length:, :]
                    targets = targets[:, :seq_len_out]
                loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
        else:
            with torch.no_grad():
                if scaler is not None:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)
                if prefix_length > 0:
                    seq_len_out = outputs.shape[1] - prefix_length
                    outputs = outputs[:, prefix_length:, :]
                    targets = targets[:, :seq_len_out]
                loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))

        total_loss += loss.item()
        steps += 1

        if config.dev_mode and steps % 10 == 0:
            logger.info(f"[Rank {dist.get_rank()}] Step {steps}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / steps
    avg_loss_tensor = torch.tensor([avg_loss], device=device)
    dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
    avg_loss = avg_loss_tensor.item()

    end_time = time.time()
    if config.dev_mode:
        logger.info(f"{'Train' if train else 'Val'} Epoch {epoch} done in {end_time - start_time:.2f}s, Loss: {avg_loss:.4f}")
    return avg_loss

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train MiniGPT model variants.")
    parser.add_argument("--data_path", type=str, default="data", help="Path to dataset.")
    parser.add_argument("--save_path", type=str, default="models", help="Where to save checkpoints.")
    parser.add_argument("--variants", type=str, default=None, help="Comma-separated PEFT variants.")
    parser.add_argument("--dev_mode", action='store_true', help="Run in dev mode.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    if args.dev_mode:
        os.environ["DEV_MODE"] = "true"

    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    set_seed(142)

    config = MiniGPTConfig()
    config.device = torch.device(f"cuda:{local_rank}")
    config.data_path = Path(args.data_path)

    if config.dev_mode:
        config.save_path = "models_debug"
    else:
        config.save_path = "k8s_models"

    if args.variants is not None:
        chosen_variants = [v.strip() for v in args.variants.split(",")]
    else:
        chosen_variants = config.peft_variants

    if rank == 0:
        mode_str = "DEV" if config.dev_mode else "FULL"
        logger.info(f"Running {mode_str} training: epochs={config.epochs}, batch_size={config.batch_size}")
        logger.info(f"Variants to train: {chosen_variants}")
        logger.info(f"Model output directory: {config.save_path}")

    for variant in chosen_variants:
        config.update_for_peft(variant)
        if rank == 0:
            logger.info(f"Training variant={config.peft_variant}")

        train_dataset = TinyStoriesDataset(
            data_folder=config.data_path,
            mode="train",
            context_length=config.context_length,
            shuffle=True,
            tokenizer=config.tokenizer
        )

        val_dataset = TinyStoriesDataset(
            data_folder=config.data_path,
            mode="test",
            context_length=config.context_length,
            shuffle=False,
            tokenizer=config.tokenizer
        )

        if config.dev_mode:
            from torch.utils.data import Subset
            if config.train_batch_limit is not None:
                train_dataset = Subset(train_dataset, range(min(len(train_dataset), config.train_batch_limit)))
            if config.val_batch_limit is not None:
                val_dataset = Subset(val_dataset, range(min(len(val_dataset), config.val_batch_limit)))

        if rank == 0:
            logger.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

        train_sampler = DistributedSampler(train_dataset, shuffle=True) if dist.get_world_size() > 1 else None
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if dist.get_world_size() > 1 else None

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                                  shuffle=(train_sampler is None), sampler=train_sampler,
                                  num_workers=config.num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                                shuffle=False, sampler=val_sampler,
                                num_workers=config.num_workers, pin_memory=True)

        base_model = build_model(config).to(config.device)
        model = apply_peft(base_model, config).to(config.device)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        if rank == 0:
            logger.info(f"Variant={config.peft_variant}: Trainable Params={trainable_params}, "
                        f"Total Params={total_params} ({(trainable_params/total_params)*100:.2f}% trainable)")

        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

        criterion = nn.CrossEntropyLoss().to(config.device)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                      lr=config.learning_rate, weight_decay=config.weight_decay)

        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * config.epochs
        if total_steps > 0:
            warmup_steps = max(1, total_steps // 10)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda step: min((step+1)/warmup_steps, 1.0)
            )
        else:
            scheduler = None

        scaler = torch.cuda.amp.GradScaler() if (config.mixed_precision and config.device.type == 'cuda') else None

        # Early stopping variables
        best_val_loss = float('inf')
        epochs_no_improve = 0
        stop_training = False

        for epoch in range(1, config.epochs+1):
            if stop_training:
                break
            train_loss = run_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, config.device, config, epoch, train=True)
            val_loss = run_epoch(model, val_loader, criterion, optimizer, scheduler, scaler, config.device, config, epoch, train=False)

            if rank == 0:
                logger.info(f"Variant={config.peft_variant} Epoch={epoch} Train Loss={train_loss:.4f} Val Loss={val_loss:.4f}")

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    # Save best model so far
                    variant_dir = os.path.join(str(config.save_path), config.peft_variant if config.peft_variant else "base")
                    os.makedirs(variant_dir, exist_ok=True)
                    best_model_path = os.path.join(variant_dir, "model_best.pth")
                    torch.save(model.module.state_dict(), best_model_path)
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= config.early_stopping_patience:
                        logger.info("Early stopping triggered due to no improvement in validation loss.")
                        stop_training = True

        # After training all epochs or early stop triggered, save final model checkpoint from rank 0
        if rank == 0:
            variant_dir = os.path.join(str(config.save_path), config.peft_variant if config.peft_variant else "base")
            os.makedirs(variant_dir, exist_ok=True)
            final_model_path = os.path.join(variant_dir, "model_final.pth")
            torch.save(model.module.state_dict(), final_model_path)
            logger.info(f"Saved final model for variant={config.peft_variant} to {final_model_path}")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
