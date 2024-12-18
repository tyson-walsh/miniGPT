# evaluate_models.py

"""
This script evaluates trained MiniGPT variants on a test dataset and on generated samples.

Steps:
1. Loads saved model checkpoints for multiple variants from disk.
2. For each variant:
   - Computes perplexity on a test dataset. Perplexity is exp(average_cross_entropy_loss), 
     measuring how well the model predicts the next token.
   - Generates samples from a set of given prompts to qualitatively inspect performance.
   - Computes text-based metrics (METEOR, BLEU, ROUGE-L) comparing generated samples to reference texts.
   - Computes semantic similarity between generated texts and references using embeddings 
     from a sentence transformer model (cosine similarity).
3. Compares results from all variants, including how they differ from the base model variant.
4. Saves results (perplexities, metrics, parameter counts, and sample generations) to a specified output file.

Why and how it works:
- Evaluations provide insight into how parameter-efficient fine-tuning methods perform compared to a base model.
- Perplexity quantifies language model quality.
- Text generation samples offer qualitative checks on model behavior.
- METEOR, BLEU, ROUGE-L, and cosine similarity score alignment with ground truth references.
- Storing results in a table format and sample outputs helps track which variants performed best and how.

Shapes and Data:
- The model output for perplexity calculation: [batch_size, seq_len, vocab_size]
  where seq_len = context_length, vocab_size = tokenizer.vocab_size.
- During text generation, we autoregressively sample one token at a time, expanding the sequence from [1, init_seq_len] to [1, init_seq_len + new_tokens].
- Text metrics operate on tokenized strings or embeddings, focusing on comparing model outputs to reference text samples.
"""

import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import MiniGPTConfig, set_seed
from dataset import TinyStoriesDataset
from model import MiniGPT
from train import apply_peft
from tabulate import tabulate
import logging
import argparse
from pathlib import Path

import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
import numpy as np

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

set_seed(142)

def parse_arguments():
    # Parsing command line arguments:
    # --data_path: Path to test dataset (used for perplexity calculations).
    # --output_file: Where to store the evaluation results (tables of metrics, samples).
    # --variants: Which model variants to evaluate.
    # --dev_mode: Enables a reduced dataset evaluation for quick checks.
    parser = argparse.ArgumentParser(description="Evaluate MiniGPT model variants.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset folder.")
    parser.add_argument("--output_file", type=str, default="evaluate_models.txt", help="File to save evaluation results.")
    parser.add_argument("--variants", type=str, default="base,lora,lora_plus,prefix_default,prefix_army_football,bias,adapter",
                        help="Comma-separated list of model variants.")
    parser.add_argument("--dev_mode", action='store_true', help="Run in dev mode for a quick evaluation.")
    return parser.parse_args()

def load_variant_model(config, variant, base_dir):
    # Loads a trained model checkpoint for a given variant.
    # Sets config for that variant, builds a MiniGPT model, applies PEFT if needed.
    config.update_for_peft(variant)
    config.context_length = 512  # Ensure model can handle this sequence length during evaluation.
    model = MiniGPT(config)
    model = apply_peft(model, config)

    ckpt_path = os.path.join(base_dir, variant, "model_final.pth")
    if not os.path.exists(ckpt_path):
        logger.error(f"Checkpoint not found for variant '{variant}' at {ckpt_path}.")
        return None

    # Load saved state_dict into model parameters.
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.to(config.device)
    model.eval() # evaluation mode: disables dropout, etc.
    return model

def count_trainable_params(model):
    # Counts how many parameters require gradient updates. Useful for verifying PEFT reduces training load.
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_checkpoint_size(base_dir, variant):
    # Computes size of the checkpoint file in megabytes.
    ckpt_path = os.path.join(base_dir, variant, "model_final.pth")
    if not os.path.exists(ckpt_path):
        return None
    size_bytes = os.path.getsize(ckpt_path)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb

def compute_perplexity(model, dataset, config):
    # Measures how well the model predicts the test dataset tokens.
    # Perplexity = exp(average_loss), with loss computed via CrossEntropy.
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False,
                        num_workers=config.num_workers, pin_memory=True)
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    logger.info("Computing perplexity...")

    prefix_length = config.prefix_tokens if (config.peft_variant and config.peft_variant.startswith("prefix")) else 0

    # Iterates through test dataset:
    # inputs: [batch_size, context_length]
    # targets: [batch_size, context_length]
    for inputs, targets in tqdm(loader, desc="Computing PPL", leave=True):
        inputs, targets = inputs.to(config.device), targets.to(config.device)
        with torch.no_grad():
            outputs = model(inputs)  # [batch_size, context_length, vocab_size]

        if prefix_length > 0:
            # If prefix tuning is enabled, remove prefix_tokens from loss calculation.
            seq_len_out = outputs.shape[1] - prefix_length
            outputs = outputs[:, prefix_length:, :] # shape: [batch_size, seq_len_out, vocab_size]
            targets = targets[:, :seq_len_out]

        # Flatten:
        # outputs: [batch_size * seq_len_out, vocab_size]
        # targets: [batch_size * seq_len_out]
        bsz, seq_len, vocab = outputs.shape
        outputs = outputs.reshape(-1, vocab)
        targets = targets.reshape(-1)

        loss = criterion(outputs, targets) # Cross entropy loss per token.
        total_loss += loss.item() * targets.size(0)
        total_tokens += targets.size(0)

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    ppl = math.exp(avg_loss) if avg_loss < float('inf') else float('inf')
    return ppl

def top_p_sampling(logits, top_p=0.9, temperature=1.0):
    # Implements top-p (nucleus) sampling for generation:
    # Sorts vocabulary tokens by probability, keeps cumulative mass <= top_p, then samples from that subset.
    # temperature scales logits: >1.0 = more random, <1.0 = more deterministic.
    if temperature != 1.0:
        logits = logits / temperature
    sorted_probs, sorted_indices = torch.sort(torch.softmax(logits, dim=-1), descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    cutoff = (cumulative_probs > top_p).nonzero(as_tuple=True)[0]
    if len(cutoff) > 0:
        cutoff_idx = cutoff[0]
        # Keep only tokens up to cutoff_idx
        sorted_probs = sorted_probs[:cutoff_idx + 1]
        sorted_indices = sorted_indices[:cutoff_idx + 1]
    next_token_id = sorted_indices[torch.multinomial(sorted_probs, 1)]
    return next_token_id

def generate_samples(model, tokenizer, prompts, config, max_new_tokens=50, temperature=0.7, top_p=0.9):
    # Generates text for each prompt. 
    # Initially, model sees input_ids=[1, init_seq_length]. 
    # Each iteration: forward pass -> next token probability distribution.
    # Sample next token (top-p), append to sequence, stop if EOS or max_new_tokens reached.
    model.eval()
    results = {}
    logger.info("Generating samples...")
    for prompt in prompts:
        try:
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(config.device)
            # input_ids: [1, initial_prompt_length]
            generated = input_ids.clone()
            for _ in range(max_new_tokens):
                with torch.no_grad():
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                        outputs = model(generated) # [1, current_seq_len, vocab_size]
                # Get the logits for the last generated token:
                next_token_logits = outputs[0, -1, :]  # shape: [vocab_size]
                # Sample next token:
                next_token_id = top_p_sampling(next_token_logits, top_p=top_p, temperature=temperature)
                # Append the newly generated token:
                generated = torch.cat([generated, next_token_id.unsqueeze(0)], dim=1) # [1, current_seq_len+1]
                # If this token is the EOS, stop generation:
                if next_token_id.item() == config.tokenizer.eos_token_id:
                    break
            # Decode the entire generated sequence into text:
            text = config.tokenizer.decode(generated[0], skip_special_tokens=True)
            results[prompt] = text
        except Exception as e:
            # In case of any error in generation, log it and produce "ERROR" as output.
            logger.error(f"Error generating sample for prompt '{prompt}': {e}")
            results[prompt] = "ERROR"
    return results

def compute_additional_metrics(samples, references):
    # Given generated samples and corresponding references, compute:
    # METEOR: Considers exact, stem, and synonym matches. 
    # BLEU: Measures n-gram precision.
    # ROUGE-L: Longest common subsequence measure between ref and generated text.
    # Cosine similarity: Uses sentence embeddings to measure semantic closeness.
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    st_model = SentenceTransformer(model_name)

    meteor_scores = []
    rouge_scores = []
    bleu_scores = []
    cos_sims = []
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    for prompt, gen_text in samples.items():
        ref_text = references.get(prompt, "")
        # Tokenize reference and generated text for METEOR/BLEU:
        ref_tokens = nltk.word_tokenize(ref_text.lower()) if ref_text.strip() else []
        gen_tokens = nltk.word_tokenize(gen_text.lower()) if gen_text.strip() else []

        meteor_val = meteor_score([ref_tokens], gen_tokens) if ref_tokens else 0.0
        # BLEU compares n-gram overlap. BLEU is computed as sentence_bleu([reference_tokens], generated_tokens).
        bleu_val = sentence_bleu([ref_tokens], gen_tokens) if ref_tokens else 0.0

        if ref_text.strip():
            # ROUGE-L measures longest common subsequence. High ROUGE-L means good overlap with reference text.
            rouge_result = rouge_scorer_instance.score(ref_text, gen_text)
            rougeL_val = rouge_result['rougeL'].fmeasure
        else:
            rougeL_val = 0.0

        if ref_text.strip():
            # Encode ref_text and gen_text into embeddings and compute cosine similarity.
            embeddings = st_model.encode([ref_text, gen_text], convert_to_tensor=True)
            # embeddings: [2, embedding_dim]
            ref_emb, gen_emb = embeddings[0], embeddings[1]
            # Cosine similarity: close to 1 means semantically similar.
            cos_sim = torch.nn.functional.cosine_similarity(ref_emb.unsqueeze(0), gen_emb.unsqueeze(0)).item()
        else:
            cos_sim = 0.0

        meteor_scores.append(meteor_val)
        bleu_scores.append(bleu_val)
        rouge_scores.append(rougeL_val)
        cos_sims.append(cos_sim)

    avg_meteor = sum(meteor_scores)/len(meteor_scores) if meteor_scores else "N/A"
    avg_rouge = sum(rouge_scores)/len(rouge_scores) if rouge_scores else "N/A"
    avg_bleu = sum(bleu_scores)/len(bleu_scores) if bleu_scores else "N/A"
    avg_cos = sum(cos_sims)/len(cos_sims) if cos_sims else "N/A"

    def format_float(val):
        return f"{val:.4f}" if isinstance(val, float) else "N/A"

    return (format_float(avg_meteor),
            format_float(avg_rouge),
            format_float(avg_bleu),
            format_float(avg_cos))

def evaluate_variants(variants, dataset, tokenizer, prompts, config, base_dir):
    # Evaluate multiple variants:
    # For each variant:
    #  - Load model
    #  - Compute perplexity on the dataset
    #  - Generate samples
    #  - Compute METEOR, ROUGE, BLEU, cosine similarity
    #  - Measure how many parameters are trainable and checkpoint size.
    param_counts = {}
    perplexities = {}
    sample_outputs = {}
    metric_results = {}
    param_percentage = {}
    checkpoint_sizes = {}

    logger.info("Evaluating models...")

    references = {
        "Once upon a time": "Once upon a time there was a small village by the river.",
        "In a world consumed by Generative AI": "In a world consumed by Generative AI, humans struggled to discern truth.",
        "If Army beats Navy this year": "If Army beats Navy this year, the celebration will be grand."
    }

    for variant in variants:
        logger.info(f"Evaluating {variant} variant:")
        model = load_variant_model(config, variant, base_dir)
        if model is None:
            continue
        total_params = sum(p.numel() for p in model.parameters())
        trained_params = count_trainable_params(model)
        params_percent = (trained_params / total_params) * 100.0

        ppl = compute_perplexity(model, dataset, config)
        samples = generate_samples(model, tokenizer, prompts, config)
        meteor, rouge, bleu, cos_sim = compute_additional_metrics(samples, references)
        ckpt_size = compute_checkpoint_size(base_dir, variant)

        # Store results for this variant:
        param_counts[variant] = trained_params
        perplexities[variant] = ppl
        sample_outputs[variant] = samples
        metric_results[variant] = (meteor, rouge, bleu, cos_sim)
        param_percentage[variant] = params_percent
        checkpoint_sizes[variant] = ckpt_size if ckpt_size else 0.0

    # Compare variants to the base model (if present):
    diff_from_base = {}
    if "base" in metric_results:
        base_meteor, base_rouge, base_bleu, base_cos = metric_results["base"]
        base_ppl = perplexities["base"]
        base_params_percent = param_percentage["base"]
        base_ckpt_size = checkpoint_sizes["base"]

        def safe_float(x):
            try:
                return float(x)
            except:
                return np.nan

        for variant in variants:
            if variant not in metric_results:
                continue
            v_meteor, v_rouge, v_bleu, v_cos = metric_results[variant]

            meteor_diff = safe_float(v_meteor) - safe_float(base_meteor)
            rouge_diff = safe_float(v_rouge) - safe_float(base_rouge)
            bleu_diff = safe_float(v_bleu) - safe_float(base_bleu)
            cos_diff = safe_float(v_cos) - safe_float(base_cos)

            ppl_diff = perplexities[variant] - base_ppl
            params_diff = param_percentage[variant] - base_params_percent
            ckpt_diff = checkpoint_sizes[variant] - base_ckpt_size

            diff_from_base[variant] = {
                "meteor_diff": meteor_diff,
                "rouge_diff": rouge_diff,
                "bleu_diff": bleu_diff,
                "cos_diff": cos_diff,
                "ppl_diff": ppl_diff,
                "params_percent_diff": params_diff,
                "ckpt_size_diff": ckpt_diff
            }

    return {
        "param_counts": param_counts,
        "perplexities": perplexities,
        "sample_outputs": sample_outputs,
        "metrics": metric_results,
        "param_percentage": param_percentage,
        "checkpoint_sizes": checkpoint_sizes,
        "diff_from_base": diff_from_base
    }

def write_results_to_file(results, output_file):
    # Writes evaluation results into an output file.
    # This includes trainable parameter counts, perplexities, metrics (METEOR, ROUGE, BLEU, Cosine Sim), and checkpoint sizes.
    param_counts = results["param_counts"]
    perplexities = results["perplexities"]
    sample_outputs = results["sample_outputs"]
    metric_results = results["metrics"]
    param_percentage = results["param_percentage"]
    checkpoint_sizes = results["checkpoint_sizes"]
    diff_from_base = results["diff_from_base"]

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=== Model Comparison Summary ===\n\n")
        f.write("Trainable Parameter Counts, Perplexities, Additional Metrics, and Checkpoint Sizes:\n")

        table_headers = ["Variant", "Trainable Params", "% of Full Params", "PPL", "METEOR", "ROUGE", "BLEU", "Cosine Sim", "Checkpoint Size (MB)"]
        table_data = []
        for variant in param_counts.keys():
            meteor, rouge, bleu, cosine_sim = metric_results[variant]
            table_data.append([
                variant,
                param_counts[variant],
                f"{param_percentage[variant]:.2f}%",
                f"{perplexities[variant]:.4f}",
                meteor,
                rouge,
                bleu,
                cosine_sim,
                f"{checkpoint_sizes[variant]:.2f}"
            ])
        # Tabulate the main results for clarity.
        f.write(tabulate(table_data, headers=table_headers))
        f.write("\n\n")

        if diff_from_base:
            f.write("Differences from Base Model:\n")
            diff_headers = ["Variant", "METEOR_diff", "ROUGE_diff", "BLEU_diff", "Cos_diff", "PPL_diff", "Params%_diff", "CKPT_Size_diff"]
            diff_data = []
            for variant, diffs in diff_from_base.items():
                def fmt_diff(val):
                    return f"{val:.4f}" if not np.isnan(val) else "N/A"
                diff_data.append([
                    variant,
                    fmt_diff(diffs['meteor_diff']),
                    fmt_diff(diffs['rouge_diff']),
                    fmt_diff(diffs['bleu_diff']),
                    fmt_diff(diffs['cos_diff']),
                    f"{diffs['ppl_diff']:.4f}",
                    f"{diffs['params_percent_diff']:.4f}%",
                    f"{diffs['ckpt_size_diff']:.4f}"
                ])
            f.write(tabulate(diff_data, headers=diff_headers))
            f.write("\n\n")

        f.write("Sample Generations:\n")
        for variant, samples in sample_outputs.items():
            f.write(f"\n--- {variant} ---\n")
            for prompt, text in samples.items():
                # Shows the prompt and what the model generated for a qualitative check.
                f.write(f"Prompt: {prompt}\n{text}\n\n")

def main():
    args = parse_arguments()

    if args.dev_mode:
        os.environ["DEV_MODE"] = "true"
    else:
        os.environ["DEV_MODE"] = "false"

    config = MiniGPTConfig()
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.data_path = Path(args.data_path)

    # If dev mode is on, might store results in models_debug, else in k8s_models
    if config.dev_mode:
        base_dir = "k8s_models" #change to `models_debug`` if you want to run this on the smaller debug models
    else:
        base_dir = "k8s_models"

    tokenizer = config.tokenizer

    val_dataset = TinyStoriesDataset(
        data_folder=config.data_path,
        mode="test",
        context_length=config.context_length,
        shuffle=False,
        tokenizer=tokenizer
    )

    # If dev_mode is True and val_batch_limit set, use a subset of val dataset.
    if config.dev_mode and config.val_batch_limit is not None:
        from torch.utils.data import Subset
        val_dataset = Subset(val_dataset, range(min(len(val_dataset), config.val_batch_limit)))
        logger.info(f"DEV MODE: Using only {len(val_dataset)} samples for evaluation.")

    variants = [v.strip() for v in args.variants.split(",")]
    # Prompts to generate samples from. Each prompt: tokenized and fed into model for autoregressive generation.
    prompts = [
        "Once upon a time",
        "In a world consumed by Generative AI",
        "If Army beats Navy this year"
    ]

    results = evaluate_variants(variants, val_dataset, tokenizer, prompts, config, base_dir)
    write_results_to_file(results, args.output_file)
    logger.info(f"Evaluation complete. Results saved to {args.output_file}.")

if __name__ == "__main__":
    main()