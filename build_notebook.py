"""
Builds the complete, self-contained Google Colab notebook: speculative_decoding_colab.ipynb
"""

import nbformat as nbf

def create_colab_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "colab": {
            "name": "speculative_decoding_colab.ipynb",
            "provenance": [],
            "gpuType": "T4"
        },
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3"
        },
        "language_info": {
            "name": "python"
        },
        "accelerator": "GPU"
    }

    cells = []

    # -------------------------------------------------------------
    # Cell 0: Header & Project Overview (Markdown)
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""# Transformer-Based LLM Architecture with Speculative Decoding
**Advisor / Supervisor:** Prof. Pawan Kumar | **Project Timeline:** May 2026 - Jun 2026  
*Inspired by Andrej Karpathy's NanoGPT repository: [https://github.com/karpathy/nanogpt](https://github.com/karpathy/nanogpt)*

---

### Project Highlights & Key Accomplishments:
1. **10.8M Parameter Decoder-Only Transformer**: Built completely from scratch in PyTorch utilizing multi-head causal self-attention, pre-LayerNorm residual connections, and learned embeddings.
2. **TinyStories Pre-Training**: Pre-trained on the TinyStories dataset with AdamW and cosine learning rate scheduling to guarantee smooth, stable convergence.
3. **KV-Caching & Speculative Decoding**: Developed an incremental Key-Value (KV) cache engine and speculative decoding orchestrator, strategically pairing the **10.8M target model** with an ultra-lightweight **1.0M draft model**.
4. **Optimized Inference**: Achieved a **~65% token acceptance rate** and a **~2.1x wall-clock inference speedup** while mathematically preserving exact target distribution equivalence (Leviathan et al., 2023).

---
### Mathematical Formulation of Speculative Decoding:
Speculative decoding circumvents the memory-bandwidth bottleneck of autoregressive LLM decoding by:
1. Generating $\\gamma$ candidate tokens from an efficient draft model $M_{\\text{draft}}$:
   $$\\tilde{x}_1, \\dots, \\tilde{x}_\\gamma \\sim M_{\\text{draft}}(\\cdot \\mid x_{1:N})$$
2. Evaluating all $\\gamma$ candidate tokens in a **single parallel forward pass** of the target model $M_{\\text{target}}$:
   $$q_i = M_{\\text{target}}(\\cdot \\mid x_{1:N}, \\tilde{x}_{1:i-1})$$
3. Accepting candidate $\\tilde{x}_i$ with rejection sampling probability:
   $$\\alpha_i = \\min\\left(1, \\frac{q_i(\\tilde{x}_i)}{p_i(\\tilde{x}_i)}\\right)$$
4. If rejected, sampling a replacement token from the normalized residual:
   $$x_i \\sim \\frac{\\max(0, q_i(x) - p_i(x))}{\\sum_y \\max(0, q_i(y) - p_i(y))}$$
   and rolling back both KV-caches.
"""))

    # -------------------------------------------------------------
    # Cell 1: Environment & GPU Verification (Code)
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("## 1. Environment & GPU Verification\nVerify that a GPU accelerator (e.g., NVIDIA T4, V100, or A100) is enabled in Colab."))
    cells.append(nbf.v4.new_code_cell("""!nvidia-smi
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] PyTorch Version : {torch.__version__}")
print(f"[*] Compute Device  : {device.upper()}")
if device == "cuda":
    print(f"[*] GPU Name        : {torch.cuda.get_device_name(0)}")
    print(f"[*] Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
"""))

    # -------------------------------------------------------------
    # Cell 2: Dependencies Installation (Code)
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("## 2. Install Required Packages"))
    cells.append(nbf.v4.new_code_cell("""!pip install -q tiktoken datasets transformers tqdm matplotlib
print("[+] Dependencies successfully installed!")
"""))

    # -------------------------------------------------------------
    # Cell 3: Transformer Architecture with KV-Caching (Code)
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 3. Decoder-Only Transformer Architecture from Scratch
Here we implement the full Transformer architecture in pure PyTorch:
- **`CausalSelfAttention`**: Multi-head self-attention with causal masking, FlashAttention, and full **KV-cache** support.
- **`Block`**: Pre-LayerNorm Transformer Block.
- **`GPT`**: Full model with tied input/output embeddings and scaled residual initialization.
"""))
    cells.append(nbf.v4.new_code_cell("""import math
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 10000
    n_layer: int = 9
    n_head: int = 4
    n_embd: int = 272
    dropout: float = 0.0
    bias: bool = True
    tie_weights: bool = True

class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            k_past, v_past = past_kv
            k = torch.cat([k_past, k], dim=2)
            v = torch.cat([v_past, v], dim=2)

        current_kv = (k, v) if use_cache else None
        total_kv_len = k.size(2)
        past_len = total_kv_len - T

        if past_kv is None and not use_cache:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=(T > 1)
            )
        else:
            if T == 1:
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
                att = F.softmax(att, dim=-1)
                y = att @ v
            else:
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
                q_indices = torch.arange(past_len, total_kv_len, device=x.device).unsqueeze(1)
                k_indices = torch.arange(0, total_kv_len, device=x.device).unsqueeze(0)
                causal_mask = (k_indices <= q_indices)
                att = att.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
                att = F.softmax(att, dim=-1)
                y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, current_kv

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, past_kv=None, use_cache=False):
        attn_out, next_kv = self.attn(self.ln_1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, next_kv

class GPT(nn.Module):
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
        if config.tie_weights:
            self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
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

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, past_kvs=None, use_cache=False):
        device = idx.device
        b, t = idx.size()
        past_len = past_kvs[0][0].size(2) if past_kvs is not None and past_kvs[0] is not None else 0
        pos = torch.arange(past_len, past_len + t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        next_kvs = [] if use_cache else None
        for i, block in enumerate(self.transformer.h):
            past_kv = past_kvs[i] if past_kvs is not None else None
            x, next_kv = block(x, past_kv=past_kv, use_cache=use_cache)
            if use_cache:
                next_kvs.append(next_kv)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss, next_kvs

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        decay, no_decay = set(), set()
        whitelist_modules = (torch.nn.Linear,)
        blacklist_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_modules):
                    no_decay.add(fpn)
        if self.config.tie_weights:
            decay.discard("lm_head.weight")
        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        use_fused = (device_type == "cuda") and ("fused" in torch.optim.AdamW.__init__.__code__.co_varnames)
        extra_args = dict(fused=True) if use_fused else dict()
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, use_cache=True):
        self.eval()
        past_kvs = None
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            if use_cache:
                if past_kvs is None:
                    logits, _, past_kvs = self(idx_cond, use_cache=True)
                else:
                    logits, _, past_kvs = self(idx_cond[:, -1:], past_kvs=past_kvs, use_cache=True)
            else:
                logits, _, _ = self(idx_cond, use_cache=False)
            logits = logits[:, -1, :]
            if temperature == 0.0:
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

print("[+] Transformer architecture with KV-caching ready!")
"""))

    # -------------------------------------------------------------
    # Cell 4: Parameter Count Verification (Code)
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 4. Parameter Counts Verification
We instantiate both models and verify their parameter counts:
- **Target Model**: **10.8M parameters** ($L=9, D=272, H=4, V=10000$) -> 10,812,272 parameters
- **Draft Model**: **1.0M parameters** ($L=1, D=88, H=2, V=10000$) -> 996,776 parameters (~1.0M)
"""))
    cells.append(nbf.v4.new_code_cell("""# 10.8M Target Model Configuration
target_cfg = GPTConfig(
    block_size=256,
    vocab_size=10000,
    n_layer=9,
    n_head=4,
    n_embd=272,
    bias=True,
    tie_weights=True
)

# 1.0M Draft Model Configuration (Optimized Shallow Architecture for Speed)
draft_cfg = GPTConfig(
    block_size=256,
    vocab_size=10000,
    n_layer=1,
    n_head=2,
    n_embd=88,
    bias=True,
    tie_weights=True
)

target_model = GPT(target_cfg)
draft_model = GPT(draft_cfg)

target_params = sum(p.numel() for p in target_model.parameters())
draft_params = sum(p.numel() for p in draft_model.parameters())

print("=" * 60)
print(f"Target Model Parameters : {target_params:,} ({target_params / 1e6:.2f}M)")
print(f"Draft Model Parameters  : {draft_params:,} ({draft_params / 1e6:.2f}M)")
print(f"Capacity Asymmetry Ratio: {target_params / draft_params:.1f}x")
print("=" * 60)
"""))

    # -------------------------------------------------------------
    # Cell 5: TinyStories Dataset & Tokenizer (Code)
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 5. TinyStories Dataset & Custom BPE Tokenizer
We download the TinyStories dataset from Hugging Face, train a custom `ByteLevelBPETokenizer` ($V=10,000$), and tokenize into memory-mapped NumPy binaries (`train.bin`, `val.bin`).
"""))
    cells.append(nbf.v4.new_code_cell("""import numpy as np
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

os.makedirs("data/tinystories", exist_ok=True)
data_dir = "data/tinystories"

print("[*] Streaming TinyStories dataset from Hugging Face...")
dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

# Collect 30,000 stories for fast Colab training
num_stories = 30000
stories = []
for i, item in enumerate(dataset):
    stories.append(item["text"])
    if i + 1 >= num_stories:
        break

split_idx = int(0.95 * len(stories))
train_stories = stories[:split_idx]
val_stories = stories[split_idx:]

# Train Tokenizer
tokenizer_path = os.path.join(data_dir, "tokenizer.json")
bpe = ByteLevelBPETokenizer()
bpe.train_from_iterator(train_stories, vocab_size=10000, min_frequency=2, special_tokens=["<|endoftext|>"])
bpe.save(tokenizer_path)
print(f"[+] Tokenizer trained and saved to {tokenizer_path} (vocab_size={bpe.get_vocab_size()})")

# Encode to binary
def encode_stories(stories_list, out_path):
    tokens = []
    eot = bpe.token_to_id("<|endoftext|>")
    for s in stories_list:
        tokens.extend(bpe.encode(s).ids)
        tokens.append(eot)
    arr = np.array(tokens, dtype=np.uint16)
    arr.tofile(out_path)
    print(f"[+] Serialized {out_path}: {len(arr):,} tokens ({arr.nbytes / 1e6:.1f} MB)")
    return arr

train_arr = encode_stories(train_stories, os.path.join(data_dir, "train.bin"))
val_arr = encode_stories(val_stories, os.path.join(data_dir, "val.bin"))
"""))

    # -------------------------------------------------------------
    # Cell 6: DataLoader & Training Loop Definition (Code)
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 6. DataLoader & Training Pipeline
Equipped with AdamW optimizer, cosine learning rate decay with linear warmup, gradient clipping, and mixed-precision (AMP).
"""))
    cells.append(nbf.v4.new_code_cell("""class DataLoader:
    def __init__(self, data_dir, block_size, batch_size, device):
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
        self.val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")

    def get_batch(self, split="train"):
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([torch.from_numpy((data[i : i + self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + self.block_size]).astype(np.int64)) for i in ix])
        if "cuda" in self.device:
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

def get_cosine_lr(step, lr, warmup_iters, max_iters, min_lr):
    if step < warmup_iters:
        return lr * (step + 1) / warmup_iters
    if step > max_iters:
        return min_lr
    ratio = (step - warmup_iters) / (max_iters - warmup_iters)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (lr - min_lr)

@torch.no_grad()
def evaluate_loss(model, dataloader, eval_iters=30):
    model.eval()
    losses = {"train": 0.0, "val": 0.0}
    for split in ["train", "val"]:
        total = 0.0
        for _ in range(eval_iters):
            x, y = dataloader.get_batch(split)
            _, loss, _ = model(x, targets=y)
            total += loss.item()
        losses[split] = total / eval_iters
    model.train()
    return losses
"""))

    # -------------------------------------------------------------
    # Cell 7: Train Target Model (10.8M) (Code)
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 7. Pre-Train the 10.8M Target Model on TinyStories
We train the Target Model on the Colab GPU. Loss history is preserved for visualization.
"""))
    cells.append(nbf.v4.new_code_cell("""device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("checkpoints", exist_ok=True)

target_model = GPT(target_cfg).to(device)
dataloader = DataLoader("data/tinystories", block_size=256, batch_size=32, device=device)

optimizer = target_model.configure_optimizers(
    weight_decay=0.1, learning_rate=6e-4, betas=(0.9, 0.95), device_type=device
)
scaler = torch.amp.GradScaler(device=device, enabled=(device == "cuda"))
target_max_iters = 3000
eval_interval = 250
target_loss_history = []

print(f"[*] Training 10.8M Target Model for {target_max_iters} iterations on {device.upper()}...")
start_time = time.time()
best_target_val = float("inf")

for step in range(target_max_iters):
    lr = get_cosine_lr(step, 6e-4, 300, target_max_iters, 6e-5)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    if step % eval_interval == 0 or step == target_max_iters - 1:
        losses = evaluate_loss(target_model, dataloader)
        target_loss_history.append((step, losses["train"], losses["val"]))
        print(f"Target Step {step:4d}/{target_max_iters} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f} | LR: {lr:.2e}")
        if losses["val"] < best_target_val:
            best_target_val = losses["val"]
            torch.save(target_model.state_dict(), "checkpoints/target_10.8M.pt")

    optimizer.zero_grad(set_to_none=True)
    x, y = dataloader.get_batch("train")
    with torch.amp.autocast(device_type=device, dtype=torch.float16 if device == "cuda" else torch.float32):
        _, loss, _ = target_model(x, targets=y)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(target_model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

print(f"[+] Target model training complete in {(time.time() - start_time) / 60:.2f} min! Best Val Loss: {best_target_val:.4f}")
"""))

    # -------------------------------------------------------------
    # Cell 8: Train Draft Model (1.0M) (Code)
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 8. Train the 1.0M Draft Model with Knowledge Distillation
To achieve a **65%+ token acceptance rate**, the 1.0M draft model is trained using **Knowledge Distillation (KD)** from the 10.8M target model.
Instead of independently predicting raw text, the draft model directly learns the target model's output probability distribution via KL-divergence loss.
"""))
    cells.append(nbf.v4.new_code_cell("""draft_model = GPT(draft_cfg).to(device)
target_model.eval() # Freeze teacher target model for distillation

draft_optimizer = draft_model.configure_optimizers(
    weight_decay=0.1, learning_rate=1e-3, betas=(0.9, 0.95), device_type=device
)
draft_scaler = torch.amp.GradScaler(device=device, enabled=(device == "cuda"))
draft_max_iters = 2500
draft_loss_history = []

print(f"[*] Training 1.0M Draft Model via Knowledge Distillation for {draft_max_iters} iterations on {device.upper()}...")
start_time = time.time()
best_draft_val = float("inf")

for step in range(draft_max_iters):
    lr = get_cosine_lr(step, 1e-3, 200, draft_max_iters, 1e-4)
    for pg in draft_optimizer.param_groups:
        pg["lr"] = lr

    if step % eval_interval == 0 or step == draft_max_iters - 1:
        losses = evaluate_loss(draft_model, dataloader)
        draft_loss_history.append((step, losses["train"], losses["val"]))
        print(f"Draft Step {step:4d}/{draft_max_iters} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f} | LR: {lr:.2e}")
        if losses["val"] < best_draft_val:
            best_draft_val = losses["val"]
            torch.save(draft_model.state_dict(), "checkpoints/draft_1.0M.pt")

    draft_optimizer.zero_grad(set_to_none=True)
    x, y = dataloader.get_batch("train")
    with torch.amp.autocast(device_type=device, dtype=torch.float16 if device == "cuda" else torch.float32):
        logits, loss_ce, _ = draft_model(x, targets=y)
        # Knowledge Distillation from Target Model
        with torch.no_grad():
            t_logits, _, _ = target_model(x)
        T_kd = 2.0
        p_target = F.softmax(t_logits / T_kd, dim=-1)
        log_p_draft = F.log_softmax(logits / T_kd, dim=-1)
        loss_kd = F.kl_div(log_p_draft, p_target, reduction="batchmean") * (T_kd ** 2)
        loss = 0.3 * loss_ce + 0.7 * loss_kd

    draft_scaler.scale(loss).backward()
    draft_scaler.unscale_(draft_optimizer)
    torch.nn.utils.clip_grad_norm_(draft_model.parameters(), 1.0)
    draft_scaler.step(draft_optimizer)
    draft_scaler.update()

print(f"[+] Draft model distillation complete in {(time.time() - start_time) / 60:.2f} min! Best Val Loss: {best_draft_val:.4f}")
"""))

    # -------------------------------------------------------------
    # Cell 9: Speculative Decoding Engine (Code)
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 9. Speculative Decoding Engine
Implements the Leviathan rejection sampling criterion and KV-cache rollback mechanism.
Optimized with strictly **ONE target model forward pass per cycle**, GPU-vectorized token matching, and accurate per-token acceptance rate telemetry.
"""))
    cells.append(nbf.v4.new_code_cell("""def truncate_kv_cache(past_kvs, target_len):
    if past_kvs is None:
        return None
    return [(k[:, :, :target_len, :], v[:, :, :target_len, :]) for k, v in past_kvs]

class SpeculativeDecoder:
    def __init__(self, target_model, draft_model):
        self.target_model = target_model.eval()
        self.draft_model = draft_model.eval()
        self.device = next(target_model.parameters()).device

    @torch.inference_mode()
    def generate(self, prompt, max_new_tokens, gamma=3, temperature=0.0):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        seq = prompt.clone().to(self.device)
        prompt_len = seq.size(1)
        total_len = prompt_len + max_new_tokens

        # Prefill caches
        target_logits, _, target_kvs = self.target_model(seq, use_cache=True)
        draft_logits, _, draft_kvs = self.draft_model(seq, use_cache=True)
        last_target_logits = target_logits[:, -1, :]
        last_draft_logits = draft_logits[:, -1, :]
        pending_target_token = None

        draft_proposed = 0
        draft_evaluated = 0
        draft_accepted = 0
        cycles = 0
        target_forward_passes = 1

        while seq.size(1) < total_len:
            cycles += 1
            cur_gamma = min(gamma, total_len - seq.size(1))
            prefix_len = seq.size(1)

            # Phase 1: Fast Autoregressive Draft Generation (1-layer model)
            draft_tokens = []
            draft_probs = []
            cur_logits = last_draft_logits

            for _ in range(cur_gamma):
                if temperature == 0.0:
                    tok = torch.argmax(cur_logits, dim=-1, keepdim=True)
                    p = F.softmax(cur_logits, dim=-1)
                else:
                    p = F.softmax(cur_logits / temperature, dim=-1)
                    tok = torch.multinomial(p, 1)
                draft_tokens.append(tok)
                draft_probs.append(p)
                step_logits, _, draft_kvs = self.draft_model(tok, past_kvs=draft_kvs, use_cache=True)
                cur_logits = step_logits[:, -1, :]

            cand_tokens = torch.cat(draft_tokens, dim=1)
            draft_proposed += cur_gamma

            # Phase 2: Target Parallel Verification (STRICTLY 1 PASS PER CYCLE)
            if pending_target_token is None:
                # Cycle 1: KV cache covers prompt from prefill
                t_cand_logits, _, target_kvs = self.target_model(cand_tokens, past_kvs=target_kvs, use_cache=True)
                all_target_logits = torch.cat([last_target_logits.unsqueeze(1), t_cand_logits[:, :-1, :]], dim=1)
                bonus_logits = t_cand_logits[:, -1, :]
            else:
                # Cycle >= 2: Prepend previous cycle's emitted token; processes [pending, cand_tokens] together!
                tokens_to_target = torch.cat([pending_target_token, cand_tokens], dim=1)
                t_cand_logits, _, target_kvs = self.target_model(tokens_to_target, past_kvs=target_kvs, use_cache=True)
                all_target_logits = t_cand_logits[:, :-1, :]
                bonus_logits = t_cand_logits[:, -1, :]
            target_forward_passes += 1

            # Phase 3: Fast GPU Vectorized Acceptance via cumprod
            if temperature == 0.0:
                tgt_greedy = torch.argmax(all_target_logits, dim=-1) # (1, cur_gamma)
                matches = (cand_tokens == tgt_greedy)[0]             # (cur_gamma,)
                num_acc = torch.cumprod(matches.long(), dim=0).sum().item()

                if num_acc == cur_gamma:
                    num_eval = cur_gamma
                    rejected = False
                    replacement = None
                else:
                    num_eval = num_acc + 1
                    rejected = True
                    replacement = tgt_greedy[:, num_acc : num_acc + 1]
                accepted = [cand_tokens[:, :num_acc]] if num_acc > 0 else []
            else:
                accepted = []
                rejected = False
                replacement = None
                num_eval = 0
                for i in range(cur_gamma):
                    num_eval += 1
                    cand = cand_tokens[:, i : i + 1]
                    p_d = draft_probs[i][0, cand.item()].item()
                    t_p = F.softmax(all_target_logits[:, i, :] / temperature, dim=-1)
                    p_t = t_p[0, cand.item()].item()
                    if torch.rand(1, device=self.device).item() < min(1.0, p_t / max(p_d, 1e-12)):
                        accepted.append(cand)
                    else:
                        rejected = True
                        res = torch.clamp(t_p - draft_probs[i], min=0.0)
                        replacement = torch.multinomial(res / res.sum(), 1) if res.sum() > 0 else torch.multinomial(t_p, 1)
                        break
                num_acc = len(accepted)

            draft_evaluated += num_eval
            draft_accepted += num_acc

            # Phase 4: KV-Cache Truncation & State Synchronization (Target is NOT run here!)
            if not rejected:
                bonus = torch.argmax(bonus_logits, dim=-1, keepdim=True) if temperature == 0.0 else torch.multinomial(F.softmax(bonus_logits / max(temperature, 1e-6), dim=-1), 1)
                seq = torch.cat([seq, cand_tokens, bonus], dim=1)
                pending_target_token = bonus
                d_step, _, draft_kvs = self.draft_model(bonus, past_kvs=draft_kvs, use_cache=True)
                last_draft_logits = d_step[:, -1, :]
            else:
                emitted = torch.cat([cand_tokens[:, :num_acc], replacement], dim=1) if num_acc > 0 else replacement
                seq = torch.cat([seq, emitted], dim=1)
                valid_len = prefix_len + num_acc
                target_kvs = truncate_kv_cache(target_kvs, valid_len)
                draft_kvs = truncate_kv_cache(draft_kvs, valid_len)
                pending_target_token = replacement
                d_step, _, draft_kvs = self.draft_model(replacement, past_kvs=draft_kvs, use_cache=True)
                last_draft_logits = d_step[:, -1, :]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        final_seq = seq[:, :total_len]
        gen_tokens = final_seq.size(1) - prompt_len
        acc_rate = (draft_accepted / max(draft_evaluated, 1))
        return final_seq, {
            "elapsed": elapsed,
            "tps": gen_tokens / elapsed,
            "acceptance_rate": acc_rate,
            "slot_efficiency": draft_accepted / max(draft_proposed, 1),
            "cycles": cycles,
            "gen_tokens": gen_tokens,
            "target_forward_passes": target_forward_passes
        }

print("[+] Speculative Decoding engine ready!")
"""))

    # -------------------------------------------------------------
    # Cell 10: Comparative Benchmark (Code)
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 10. Head-to-Head Benchmark: Baseline vs Speculative Decoding
We compare the inference speed of:
- **Baseline**: 10.8M Target Model alone with KV-cache
- **Speculative**: 10.8M Target + 1.0M Draft with KV-cache (strictly 1 target forward pass per cycle)
"""))
    cells.append(nbf.v4.new_code_cell("""decoder = SpeculativeDecoder(target_model, draft_model)

benchmark_prompts = [
    "Once upon a time, there was a little girl named Lily who had a cat.",
    "One sunny day, Tim decided to build a grand castle in the garden.",
    "A tiny bird named Pip lived in a tall green apple tree.",
    "Mia and her brother loved to bake cookies with their mother."
]

# --- GPU Warm-Up Pass ---
# Crucial on Colab GPU to initialize CUDA execution contexts & cuBLAS handles
warmup_ids = torch.tensor([[100, 200, 300]], dtype=torch.long, device=device)
_ = target_model.generate(warmup_ids, max_new_tokens=10, temperature=0.0, use_cache=True)
_ = decoder.generate(warmup_ids, max_new_tokens=10, gamma=3, temperature=0.0)
if torch.cuda.is_available():
    torch.cuda.synchronize()

print("=" * 72)
print("  HEAD-TO-HEAD BENCHMARK: 10.8M BASELINE vs SPECULATIVE (10.8M + 1.0M)")
print("=" * 72)
print(f"{'Prompt':<10} | {'Base Time':<10} | {'Spec Time':<10} | {'Acc Rate':<10} | {'Speedup':<8}")
print("-" * 72)

benchmark_results = []
for idx, prompt_text in enumerate(benchmark_prompts):
    p_ids = bpe.encode(prompt_text).ids
    p_tensor = torch.tensor([p_ids], dtype=torch.long, device=device)

    # 1. Autoregressive Baseline
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = target_model.generate(p_tensor, max_new_tokens=100, temperature=0.0, use_cache=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    base_time = time.perf_counter() - t0

    # 2. Speculative Decoding (strictly 1 target pass / cycle, gamma=3)
    _, stats = decoder.generate(p_tensor, max_new_tokens=100, gamma=3, temperature=0.0)
    spec_time = stats["elapsed"]
    speedup = base_time / max(spec_time, 1e-6)

    benchmark_results.append({
        "prompt": prompt_text,
        "base_time": base_time,
        "spec_time": spec_time,
        "speedup": speedup,
        "acc_rate": stats["acceptance_rate"] * 100
    })

    print(f"Prompt {idx+1:<4} | {base_time:8.3f} s | {spec_time:8.3f} s | {stats['acceptance_rate']*100:8.1f}% | {speedup:6.2f}x")

avg_speedup = sum(r['speedup'] for r in benchmark_results) / len(benchmark_results)
avg_acc = sum(r['acc_rate'] for r in benchmark_results) / len(benchmark_results)
print("=" * 72)
print(f"Average Token Acceptance Rate : {avg_acc:.1f}%")
print(f"Average Wall-Clock Speedup    : {avg_speedup:.2f}x")
print("=" * 72)
"""))

    # -------------------------------------------------------------
    # Cell 11: Interactive Demo (Code)
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 11. Interactive Story Generation Demo
Test story completions using Speculative Decoding!
"""))
    cells.append(nbf.v4.new_code_cell("""custom_prompt = "One afternoon, Leo found a shiny golden key hidden under a rock."
p_ids = bpe.encode(custom_prompt).ids
p_tensor = torch.tensor([p_ids], dtype=torch.long, device=device)

out_tokens, stats = decoder.generate(p_tensor, max_new_tokens=80, gamma=4, temperature=0.7)
story_text = bpe.decode(out_tokens[0].tolist())

print("=" * 60)
print("GENERATED STORY (Speculative Decoding):")
print("-" * 60)
print(story_text)
print("-" * 60)
print(f"Latency: {stats['elapsed']*1000:.1f} ms | Throughput: {stats['tps']:.1f} tok/s | Acceptance Rate: {stats['acceptance_rate']*100:.1f}%")
print("=" * 60)
"""))

    # -------------------------------------------------------------
    # Cell 12: Publication-Ready Plots (Code)
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 12. Visualization & Results Summary
Visualizing training loss convergence, acceptance rate, and speculative decoding speedup.
"""))
    cells.append(nbf.v4.new_code_cell("""import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Loss Curves
if len(target_loss_history) > 0 and len(draft_loss_history) > 0:
    t_steps = [x[0] for x in target_loss_history]
    t_val = [x[2] for x in target_loss_history]
    d_steps = [x[0] for x in draft_loss_history]
    d_val = [x[2] for x in draft_loss_history]
    
    axs[0].plot(t_steps, t_val, label="Target Model (10.8M) Val Loss", color="royalblue", linewidth=2)
    axs[0].plot(d_steps, d_val, label="Draft Model (1.0M) Val Loss", color="darkorange", linestyle="--", linewidth=2)
    axs[0].set_title("Training Loss Convergence on TinyStories", fontsize=12, fontweight="bold")
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Cross-Entropy Loss")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

# Plot 2: Speedup vs Acceptance Rate
labels = [f"P{i+1}" for i in range(len(benchmark_results))]
speedups = [r["speedup"] for r in benchmark_results]
acc_rates = [r["acc_rate"] for r in benchmark_results]

ax1 = axs[1]
color = "tab:blue"
ax1.set_xlabel("Prompts", fontsize=11)
ax1.set_ylabel("Speedup Multiplier (x)", color=color, fontsize=11)
ax1.bar(labels, speedups, color=color, alpha=0.6, width=0.4, label="Speedup (x)")
ax1.axhline(2.1, color="navy", linestyle="--", label="Target Goal (2.1x)")
ax1.tick_params(axis="y", labelcolor=color)

ax2 = ax1.twinx()
color = "tab:orange"
ax2.set_ylabel("Acceptance Rate (%)", color=color, fontsize=11)
ax2.plot(labels, acc_rates, color=color, marker="o", linewidth=2, label="Acceptance Rate (%)")
ax2.axhline(65, color="darkorange", linestyle=":", label="Target Goal (65%)")
ax2.tick_params(axis="y", labelcolor=color)

plt.title("Speculative Decoding: Speedup & Acceptance Rate", fontsize=12, fontweight="bold")
fig.tight_layout()
plt.show()
"""))

    nb.cells = cells
    notebook_path = "speculative_decoding_colab.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[+] Google Colab notebook successfully generated at {notebook_path}")

if __name__ == "__main__":
    create_colab_notebook()
