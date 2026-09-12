"""
Tokenizer Utilities for TinyStories and NanoGPT.
Provides:
1. TinyStoriesTokenizer: Custom Byte-Level BPE Tokenizer (vocab_size=10,000)
2. TiktokenGPT2Tokenizer: GPT-2 BPE Tokenizer fallback (vocab_size=50,257)
3. Helper functions to train, load, and serialize tokenizers.
"""

import os
from typing import List, Optional, Union


class TinyStoriesTokenizer:
    """
    Byte-Level Byte-Pair Encoding (BPE) Tokenizer tailored for TinyStories.
    Configured with vocab_size=10,000 to enable exact pairing of:
    - 10.8M Target Model
    - 1.0M Draft Model
    """
    def __init__(self, tokenizer_file: Optional[str] = None):
        self.tokenizer = None
        self.vocab_size = 10000

        if tokenizer_file is not None and os.path.exists(tokenizer_file):
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_file(tokenizer_file)
            self.vocab_size = self.tokenizer.get_vocab_size()

    @classmethod
    def train_from_iterator(
        cls,
        iterator,
        vocab_size: int = 10000,
        save_path: Optional[str] = None,
    ) -> "TinyStoriesTokenizer":
        """Trains a Byte-Level BPE tokenizer on a text iterator."""
        from tokenizers import ByteLevelBPETokenizer

        bpe = ByteLevelBPETokenizer()
        bpe.train_from_iterator(
            iterator,
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=["<|endoftext|>", "<|pad|>"],
        )
        if save_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            bpe.save(save_path)

        instance = cls()
        instance.tokenizer = bpe
        instance.vocab_size = vocab_size
        return instance

    def encode(self, text: str) -> List[int]:
        if self.tokenizer is not None:
            return self.tokenizer.encode(text).ids
        # Fallback character-level encoding if tokenizer not yet trained
        return [min(ord(c), self.vocab_size - 1) for c in text]

    def decode(self, tokens: List[int]) -> str:
        if self.tokenizer is not None:
            return self.tokenizer.decode(tokens)
        # Fallback character-level decoding
        return "".join(chr(t) for t in tokens if t < 256)


class TiktokenGPT2Tokenizer:
    """Standard GPT-2 Tiktoken tokenizer (vocab_size=50,257)."""
    def __init__(self):
        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = 50257

    def encode(self, text: str) -> List[int]:
        return self.enc.encode_ordinary(text)

    def decode(self, tokens: List[int]) -> str:
        return self.enc.decode(tokens)


def get_tokenizer(name: str = "tinystories", tokenizer_file: Optional[str] = None):
    """Factory helper to obtain a tokenizer instance."""
    if name.lower() in ("tinystories", "custom"):
        return TinyStoriesTokenizer(tokenizer_file=tokenizer_file)
    elif name.lower() in ("gpt2", "tiktoken"):
        return TiktokenGPT2Tokenizer()
    else:
        raise ValueError(f"Unknown tokenizer type: {name}")
