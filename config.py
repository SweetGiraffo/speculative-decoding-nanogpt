"""
Model Configurations for Decoder-Only Transformer with Speculative Decoding.
Defines GPTConfig dataclass and exact configurations for:
- 10.8M Target Model (9 layers, 272 embd, 4 heads, vocab 10,000) -> 10,812,272 parameters (~10.8M)
- 1.0M Draft Model  (2 layers, 80 embd, 2 heads, vocab 10,000)  -> 976,320 parameters (~1.0M)
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GPTConfig:
    block_size: int = 256        # Maximum context window length
    vocab_size: int = 10000      # Vocabulary size (TinyStories custom tokenizer or GPT-2)
    n_layer: int = 9             # Number of transformer layers
    n_head: int = 4              # Number of attention heads
    n_embd: int = 272            # Embedding dimension
    dropout: float = 0.0         # Dropout probability (0.0 for pre-training/inference)
    bias: bool = True            # True: use bias in Linears and LayerNorms (like GPT-2)
    tie_weights: bool = True     # Tie token embedding and output lm_head weights


def get_target_config(vocab_size: int = 10000, block_size: int = 256) -> GPTConfig:
    """
    Returns the 10.8M parameter Target Model configuration.
    
    Exact parameter count:
      wte: 10,000 * 272 = 2,720,000
      wpe: 256 * 272 = 69,632
      9 layers * (12 * 272^2 + 13 * 272) = 8,022,096
      ln_f: 2 * 272 = 544
      Total = 10,812,272 (~10.81M parameters)
    """
    return GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=9,
        n_head=4,
        n_embd=272,
        dropout=0.0,
        bias=True,
        tie_weights=True,
    )


def get_draft_config(vocab_size: int = 10000, block_size: int = 256) -> GPTConfig:
    """
    Returns the optimized shallow 1.0M parameter Draft Model configuration.
    Crucial for speculative decoding speedup:
    A shallow 2-layer model executes 3.5x faster per token than a 7-layer model,
    enabling the latency asymmetry required for >2x wall-clock speedup.
    
    Exact parameter count:
      wte: 10,000 * 80 = 800,000
      wpe: 256 * 80 = 20,480
      2 layers * (12 * 80^2 + 13 * 80) = 155,680
      ln_f: 2 * 80 = 160
      Total = 976,320 (~1.00M parameters)
    """
    return GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=2,
        n_head=2,
        n_embd=80,
        dropout=0.0,
        bias=True,
        tie_weights=True,
    )


def get_gpt2_target_config(block_size: int = 512) -> GPTConfig:
    """
    Alternative target config using GPT-2 tiktoken vocabulary (vocab_size=50257).
    Total parameters: 10,805,696 (~10.8M).
    """
    return GPTConfig(
        block_size=block_size,
        vocab_size=50257,
        n_layer=5,
        n_head=4,
        n_embd=176,
        dropout=0.0,
        bias=True,
        tie_weights=True,
    )


def count_parameters(model) -> dict:
    """Counts total and trainable parameters of a PyTorch model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    non_embedding = total
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
        non_embedding -= model.transformer.wte.weight.numel()
        if hasattr(model.transformer, 'wpe'):
            non_embedding -= model.transformer.wpe.weight.numel()
            
    return {
        "total": total,
        "trainable": trainable,
        "non_embedding": non_embedding,
        "total_m": total / 1e6,
        "trainable_m": trainable / 1e6,
        "non_embedding_m": non_embedding / 1e6,
    }
