# dataset.py

"""
This file defines a dataset class that provides training and validation samples 
from a binary file containing tokenized data. The dataset maps integer token sequences 
to corresponding next-token prediction targets for language modeling tasks.

Key operations:
1. Loads integer token data from a .bin file using np.memmap to allow memory-efficient reading of large files.
2. Produces (input_seq, target_seq) pairs where input_seq is a sequence of tokens of length [context_length], and target_seq is the immediate next tokens offset by one position.
3. Optionally shuffles the order of sequences to introduce randomness during training.
4. Ensures that the dataset can produce multiple examples by indexing into the memmapped array.

Shapes and Data Types:
- The underlying data is stored as a NumPy memmap of shape [N], where N is the number of tokens in the file.
- The input_seq and target_seq each have shape [context_length] after slicing, with dtype torch.int64.
- context_length: an integer specifying how many tokens form the input and target sequences.

Why and how it works:
A language model requires input sequences and their next-token targets. By slicing a large array of tokenized text, 
it is possible to produce overlapping sequences (sliding window) for training. Using memmapped data reduces memory overhead 
because not all data must be loaded into RAM. Shuffling indices provides randomness and prevents the model from seeing tokens in the same order every epoch.
"""

import random
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

class TinyStoriesDataset(Dataset):
    def __init__(
        self,
        data_folder: Path,
        mode: str = "train",
        context_length: int = 20,
        shuffle: bool = False,
        tokenizer: PreTrainedTokenizer = None,
    ):
        if mode not in ["train", "test"]:
            raise ValueError("mode must be 'train' or 'test'.")
        self.data_path = data_folder / f"{mode}.bin"
        self.context_length = context_length
        self.shuffle = shuffle
        self.tokenizer = tokenizer

        # Load a memmapped array from a binary file of dtype uint16. Shape is [total_tokens].
        # This array stores token IDs.
        try:
            self.data_memmap = np.memmap(self.data_path, dtype=np.uint16, mode="r")
            print(f"Loaded data from {self.data_path} with shape {self.data_memmap.shape}", flush=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at {self.data_path}")

        # The number of sequences is (total_length - (context_length + 1)) 
        # because each input sequence requires context_length tokens and one additional token for the target.
        self.num_sequences = len(self.data_memmap) - self.context_length - 1
        if self.num_sequences <= 0:
            raise ValueError("Dataset too small for given context_length.")

        # Create a list of indices [0, 1, 2, ..., num_sequences-1].
        # Each index corresponds to a starting position for a sequence in the memmap array.
        self.indices = list(range(self.num_sequences))

        # If shuffle is True, randomize the order of indices for training.
        # This ensures that each epoch sees sequences in a different order.
        if self.shuffle:
            random.seed(142)
            random.shuffle(self.indices)
            print("Shuffled the dataset indices for randomness.", flush=True)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        # The idx-th sample uses the starting index from self.indices[idx].
        # input_seq consists of tokens from data_memmap at [start: start+context_length]
        # target_seq consists of tokens at [start+1: start+context_length+1]
        start = self.indices[idx]
        input_seq = self.data_memmap[start: start + self.context_length].astype(np.int64)
        target_seq = self.data_memmap[start + 1: start + self.context_length + 1].astype(np.int64)

        # Convert to torch Tensors of shape [context_length] and dtype long (int64).
        input_seq = torch.from_numpy(input_seq)
        target_seq = torch.from_numpy(target_seq)
        return input_seq, target_seq
