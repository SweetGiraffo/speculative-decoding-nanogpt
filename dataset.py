"""
TinyStories Dataset Preparation and DataLoader.
Follows Andrej Karpathy's NanoGPT binary format:
- Downloads TinyStories dataset from Hugging Face
- Tokenizes text into compact np.uint16 arrays (train.bin, val.bin)
- High-performance memory-mapped DataLoader (np.memmap) with zero-copy tensor slicing
- Includes synthetic dataset generator for offline / local testing
"""

import argparse
import os
import numpy as np
import torch
from typing import Optional, Tuple

from tokenizer import TinyStoriesTokenizer, get_tokenizer


def prepare_tinystories(
    data_dir: str = "data/tinystories",
    num_samples: Optional[int] = 50000,
    vocab_size: int = 10000,
):
    """
    Downloads TinyStories dataset from Hugging Face, trains/loads a BPE tokenizer,
    and serializes the tokenized dataset into train.bin and val.bin.
    """
    os.makedirs(data_dir, exist_ok=True)
    train_bin_path = os.path.join(data_dir, "train.bin")
    val_bin_path = os.path.join(data_dir, "val.bin")
    tokenizer_path = os.path.join(data_dir, "tokenizer.json")

    print(f"[*] Preparing TinyStories dataset in {data_dir}...")
    from datasets import load_dataset

    print("[*] Loading 'roneneldan/TinyStories' from Hugging Face...")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    # Gather sample stories for tokenizer training and serialization
    stories = []
    print(f"[*] Extracting {num_samples if num_samples else 'all'} stories...")
    for i, item in enumerate(ds):
        stories.append(item["text"])
        if num_samples is not None and i + 1 >= num_samples:
            break

    # Split into train (95%) and val (5%)
    split_idx = int(0.95 * len(stories))
    train_stories = stories[:split_idx]
    val_stories = stories[split_idx:]

    # Train tokenizer on training split
    print(f"[*] Training ByteLevelBPETokenizer with vocab_size={vocab_size}...")
    tok = TinyStoriesTokenizer.train_from_iterator(
        train_stories, vocab_size=vocab_size, save_path=tokenizer_path
    )

    # Encode train split
    print("[*] Tokenizing and writing train.bin...")
    train_tokens = []
    for s in train_stories:
        train_tokens.extend(tok.encode(s))
        train_tokens.append(tok.encode("<|endoftext|>")[0] if "<|endoftext|>" in tok.tokenizer.get_vocab() else 0)

    train_arr = np.array(train_tokens, dtype=np.uint16)
    train_arr.tofile(train_bin_path)
    print(f"[+] train.bin written: {len(train_arr):,} tokens ({train_arr.nbytes / 1e6:.2f} MB)")

    # Encode val split
    print("[*] Tokenizing and writing val.bin...")
    val_tokens = []
    for s in val_stories:
        val_tokens.extend(tok.encode(s))
        val_tokens.append(tok.encode("<|endoftext|>")[0] if "<|endoftext|>" in tok.tokenizer.get_vocab() else 0)

    val_arr = np.array(val_tokens, dtype=np.uint16)
    val_arr.tofile(val_bin_path)
    print(f"[+] val.bin written: {len(val_arr):,} tokens ({val_arr.nbytes / 1e6:.2f} MB)")

    return tokenizer_path, train_bin_path, val_bin_path


def create_synthetic_data(data_dir: str = "data/synthetic", num_tokens: int = 200000, vocab_size: int = 10000):
    """Creates synthetic token arrays for rapid testing without external network calls."""
    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")

    rng = np.random.default_rng(42)
    train_tokens = rng.integers(0, vocab_size, size=num_tokens, dtype=np.uint16)
    val_tokens = rng.integers(0, vocab_size, size=num_tokens // 5, dtype=np.uint16)

    train_tokens.tofile(train_path)
    val_tokens.tofile(val_path)
    return train_path, val_path


class TinyStoriesDataLoader:
    """
    Memory-mapped DataLoader matching Karpathy's NanoGPT design.
    Efficiently samples random contiguous token blocks with zero disk re-reading overhead.
    """
    def __init__(self, data_dir: str, block_size: int = 256, batch_size: int = 32, device: str = "cpu"):
        self.data_dir = data_dir
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

        train_path = os.path.join(data_dir, "train.bin")
        val_path = os.path.join(data_dir, "val.bin")

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Missing {train_path}. Run prepare_tinystories() or create_synthetic_data() first.")

        self.train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
        self.val_data = np.memmap(val_path, dtype=np.uint16, mode="r")

    def get_batch(self, split: str = "train") -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([torch.from_numpy((data[i : i + self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + self.block_size]).astype(np.int64)) for i in ix])

        if "cuda" in str(self.device):
            # Pin and move asynchronously to GPU
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)

        return x, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/tinystories")
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    if args.synthetic:
        create_synthetic_data(args.data_dir, vocab_size=args.vocab_size)
    else:
        prepare_tinystories(args.data_dir, num_samples=args.num_samples, vocab_size=args.vocab_size)
