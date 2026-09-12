"""
Decoder-Only Transformer Architecture built from scratch in PyTorch.
Features:
- Multi-Head Causal Self-Attention with Causal Masking
- FlashAttention (F.scaled_dot_product_attention) for training and full prefill
- Key-Value (KV) Caching for fast autoregressive generation and speculative verification
- Dynamic KV Cache rollback and slicing for speculative rejection sampling
- Pre-LayerNorm architecture with residual scaling
- Weight tying between token embedding and language model head
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from config import GPTConfig


class LayerNorm(nn.Module):
    """LayerNorm with optional bias (PyTorch's default LayerNorm always requires bias)."""
    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention with support for:
    1. Standard scaled dot-product attention with causal mask (training/prefill)
    2. KV-caching for O(1) step incremental token generation
    3. Multi-token speculative verification over cached prefixes
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # Key, Query, Value projections in a single batched linear layer
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with optional KV-cache.
        
        Args:
            x: Input tensor of shape (Batch, SeqLen, Dim)
            past_kv: Tuple of (cached_k, cached_v), each of shape (Batch, n_head, PastLen, head_dim)
            use_cache: Whether to return updated (k, v) cache
            
        Returns:
            Tuple of (output, new_past_kv)
        """
        B, T, C = x.size()

        # Calculate query, key, values for all heads
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)

        # Concatenate with past KV-cache if present
        if past_kv is not None:
            k_past, v_past = past_kv
            k = torch.cat([k_past, k], dim=2)  # (B, nh, PastLen + T, hs)
            v = torch.cat([v_past, v], dim=2)  # (B, nh, PastLen + T, hs)

        current_kv = (k, v) if use_cache else None
        total_kv_len = k.size(2)
        past_len = total_kv_len - T

        if past_kv is None and not use_cache:
            # Standard fast causal attention (PyTorch FlashAttention kernel)
            is_causal = (T > 1)
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal
            )
        else:
            # Incremental generation with KV cache using fused scaled_dot_product_attention
            dropout_p = self.dropout if self.training else 0.0
            if T == 1:
                # Single new token attending to all past tokens + itself: all keys <= current pos
                y = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=False
                )
            else:
                # Multi-token speculative verification (T candidate tokens) over past prefix
                q_indices = torch.arange(past_len, total_kv_len, device=x.device).unsqueeze(1)  # (T, 1)
                k_indices = torch.arange(0, total_kv_len, device=x.device).unsqueeze(0)        # (1, total_kv_len)
                causal_mask = (k_indices <= q_indices).unsqueeze(0).unsqueeze(0)                # (1, 1, T, total_kv_len)
                y = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=causal_mask, dropout_p=dropout_p, is_causal=False
                )

        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, current_kv


class MLP(nn.Module):
    """Standard Transformer Feed-Forward Network with GELU activation."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer Block: LayerNorm -> Attention -> Residual -> LayerNorm -> MLP -> Residual."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, next_kv = self.attn(self.ln_1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, next_kv


class GPT(nn.Module):
    """
    Decoder-Only Transformer Language Model.
    Matches Karpathy's nanoGPT design while featuring KV caching and speculative decoding capabilities.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: ties input embedding weights with output linear projection weights
        if config.tie_weights:
            self.transformer.wte.weight = self.lm_head.weight

        # Initialize all weights following standard normal distribution
        self.apply(self._init_weights)

        # Apply special scaled init to residual projections (GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_kvs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass.
        
        Args:
            idx: Token indices tensor of shape (Batch, SeqLen)
            targets: Optional ground-truth target tokens of shape (Batch, SeqLen) for loss calculation
            past_kvs: Optional list of (k, v) cache tuples, one per transformer layer
            use_cache: Whether to return updated KV-caches
            
        Returns:
            Tuple of (logits, loss, next_kvs)
        """
        device = idx.device
        b, t = idx.size()

        # Compute position indices accounting for past cached length
        if past_kvs is not None and past_kvs[0] is not None:
            past_len = past_kvs[0][0].size(2)
            pos = torch.arange(past_len, past_len + t, dtype=torch.long, device=device)
        else:
            past_len = 0
            pos = torch.arange(0, t, dtype=torch.long, device=device)

        assert past_len + t <= self.config.block_size, (
            f"Cannot forward sequence of length {past_len + t}, block size is {self.config.block_size}"
        )

        # Forward embeddings
        tok_emb = self.transformer.wte(idx)  # (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Forward through transformer blocks
        next_kvs = [] if use_cache else None
        for i, block in enumerate(self.transformer.h):
            past_kv = past_kvs[i] if past_kvs is not None else None
            x, next_kv = block(x, past_kv=past_kv, use_cache=use_cache)
            if use_cache:
                next_kvs.append(next_kv)

        x = self.transformer.ln_f(x)

        # Compute logits
        if targets is not None:
            # Training: compute cross-entropy loss over all tokens
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference: compute logits for all positions forwarded
            logits = self.lm_head(x)
            loss = None

        return logits, loss, next_kvs

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        device_type: str,
    ) -> torch.optim.Optimizer:
        """
        Configures AdamW optimizer with decoupled weight decay.
        All 2D parameters (matrix weights) are decayed.
        All 1D parameters (biases, layernorm weights) are NOT decayed.
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        # Subtle: lm_head.weight and wte.weight are tied, ensure no double counting
        if self.config.tie_weights:
            decay.discard("lm_head.weight")

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"Parameters {inter_params} made it into both decay/no_decay sets!"
        assert len(param_dict.keys() - union_params) == 0, f"Parameters {param_dict.keys() - union_params} were not separated!"

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        
        # Use fused AdamW if supported and on CUDA
        use_fused = (device_type == "cuda") and ("fused" in torch.optim.AdamW.__init__.__code__.co_varnames)
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Standard autoregressive generation with optional KV caching.
        Used as the baseline comparison against speculative decoding.
        """
        self.eval()
        past_kvs = None

        for _ in range(max_new_tokens):
            # If sequence exceeds block_size, crop it
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            if use_cache:
                if past_kvs is None:
                    # Initial prefill pass: forward the full prompt
                    logits, _, past_kvs = self(idx_cond, use_cache=True)
                    logits = logits[:, -1, :]
                else:
                    # Incremental step: forward only the most recent token
                    logits, _, past_kvs = self(idx_cond[:, -1:], past_kvs=past_kvs, use_cache=True)
                    logits = logits[:, -1, :]
            else:
                # Non-cached forward pass (recomputing past tokens)
                logits, _, _ = self(idx_cond, use_cache=False)
                logits = logits[:, -1, :]

            # Apply temperature scaling
            if temperature == 0.0:
                # Greedy decoding
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx
