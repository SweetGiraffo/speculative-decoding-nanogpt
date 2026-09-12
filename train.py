"""
Pre-Training Script for Target (10.8M) and Draft (1.0M) Transformers on TinyStories.
Features:
- AdamW optimizer with decoupled 1D/2D parameter weight decay
- Cosine Annealing Learning Rate schedule with linear warmup
- Gradient clipping (1.0) and Gradient Accumulation
- Mixed Precision training (bfloat16 / float16 via torch.amp.autocast)
- Evaluation loop and best checkpoint saving (target_10.8M.pt, draft_1.0M.pt)
- Designed for Google Colab GPU training (T4, V100, A100)
"""

import argparse
import math
import os
import time
from typing import Tuple

import torch
from torch.nn import functional as F

from config import get_draft_config, get_target_config, count_parameters
from dataset import TinyStoriesDataLoader
from model import GPT


def get_lr(it: int, learning_rate: float, warmup_iters: int, lr_decay_iters: int, min_lr: float) -> float:
    """Computes learning rate with linear warmup and cosine decay."""
    # 1) Linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    # 2) If it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) In between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


@torch.no_grad()
def estimate_loss(model: GPT, dataloader: TinyStoriesDataLoader, eval_iters: int = 50) -> dict:
    """Estimates cross-entropy loss on train and validation splits."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = dataloader.get_batch(split)
            _, loss, _ = model(x, targets=y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(
    model_type: str = "target",
    data_dir: str = "data/tinystories",
    out_dir: str = "checkpoints",
    batch_size: int = 32,
    gradient_accumulation_steps: int = 2,
    max_iters: int = 5000,
    learning_rate: float = 6e-4,
    min_lr: float = 6e-5,
    warmup_iters: int = 500,
    lr_decay_iters: int = 5000,
    eval_interval: int = 250,
    eval_iters: int = 50,
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dry_run: bool = False,
    teacher_ckpt: Optional[str] = None,
):
    """
    Main training execution function.
    """
    os.makedirs(out_dir, exist_ok=True)
    device_type = "cuda" if "cuda" in device else "cpu"

    print("=" * 60)
    print(f"[*] Training {model_type.upper()} Model on {device} (dry_run={dry_run})")
    print("=" * 60)

    # Instantiate model architecture
    if model_type.lower() == "target":
        config = get_target_config()
        ckpt_name = "target_10.8M.pt"
    elif model_type.lower() == "draft":
        config = get_draft_config()
        ckpt_name = "draft_1.0M.pt"
        # Draft model can benefit from slightly higher learning rate
        learning_rate = max(learning_rate, 8e-4)
        min_lr = 0.1 * learning_rate
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = GPT(config).to(device)
    param_info = count_parameters(model)
    print(f"[+] Model instantiated: {param_info['total']:,} total parameters ({param_info['total_m']:.2f}M)")

    # DataLoader
    dataloader = TinyStoriesDataLoader(
        data_dir=data_dir,
        block_size=config.block_size,
        batch_size=batch_size,
        device=device,
    )

    # Optimizer
    optimizer = model.configure_optimizers(
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        betas=(0.9, 0.95),
        device_type=device_type,
    )

    # Automatic Mixed Precision setup
    scaler = torch.amp.GradScaler(device=device_type, enabled=(device_type == "cuda"))
    dtype = (
        torch.bfloat16
        if device_type == "cuda" and torch.cuda.is_bf16_supported()
        else (torch.float16 if device_type == "cuda" else torch.float32)
    )

    # Optional teacher model for Knowledge Distillation (KD)
    teacher_model = None
    if model_type.lower() == "draft":
        default_teacher_path = os.path.join(out_dir, "target_10.8M.pt")
        teacher_path = teacher_ckpt if teacher_ckpt is not None else default_teacher_path
        if os.path.exists(teacher_path):
            print(f"[+] Loading 10.8M Target Teacher from {teacher_path} for Knowledge Distillation...")
            t_cfg = get_target_config()
            teacher_model = GPT(t_cfg).to(device)
            ckpt = torch.load(teacher_path, map_location=device)
            state_dict = ckpt["model"] if "model" in ckpt else ckpt
            teacher_model.load_state_dict(state_dict)
            teacher_model.eval()
            print("[+] Distillation enabled: Student draft model will align with target model distribution!")

    best_val_loss = float("inf")
    start_time = time.time()
    iters = 2 if dry_run else max_iters

    model.train()
    for iter_num in range(iters):
        # Update learning rate according to cosine schedule
        lr = get_lr(iter_num, learning_rate, warmup_iters, lr_decay_iters, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Periodic evaluation & checkpointing
        if (iter_num % eval_interval == 0 or iter_num == iters - 1) and not dry_run:
            losses = estimate_loss(model, dataloader, eval_iters=eval_iters)
            print(
                f"Step {iter_num:5d}/{iters:5d} | Train Loss: {losses['train']:.4f} | "
                f"Val Loss: {losses['val']:.4f} | LR: {lr:.2e}"
            )
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                }
                save_path = os.path.join(out_dir, ckpt_name)
                torch.save(checkpoint, save_path)
                print(f"  [+] Saved new best checkpoint to {save_path} (val_loss={best_val_loss:.4f})")

        # Forward, backward, and gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(gradient_accumulation_steps):
            x, y = dataloader.get_batch("train")
            with torch.amp.autocast(device_type=device_type, dtype=dtype):
                logits, loss, _ = model(x, targets=y)
                if teacher_model is not None:
                    with torch.no_grad():
                        t_logits, _, _ = teacher_model(x)
                    T_kd = 2.0
                    p_target = F.softmax(t_logits / T_kd, dim=-1)
                    log_p_draft = F.log_softmax(logits / T_kd, dim=-1)
                    loss_kd = F.kl_div(log_p_draft, p_target, reduction="batchmean") * (T_kd ** 2)
                    loss = 0.3 * loss + 0.7 * loss_kd
                loss = loss / gradient_accumulation_steps
            accum_loss += loss.item()
            scaler.scale(loss).backward()

        # Gradient clipping
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        if iter_num % 50 == 0:
            elapsed = time.time() - start_time
            print(f"Step {iter_num:5d} | Loss: {accum_loss:.4f} | LR: {lr:.2e} | Elapsed: {elapsed:.1f}s")

    total_time = time.time() - start_time
    print(f"[*] Training finished in {total_time / 60:.2f} minutes.")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="target", choices=["target", "draft"])
    parser.add_argument("--data_dir", type=str, default="data/tinystories")
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--warmup_iters", type=int, default=500)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--teacher_ckpt", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    train(
        model_type=args.model_type,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_iters=args.max_iters,
        learning_rate=args.learning_rate,
        warmup_iters=args.warmup_iters,
        eval_interval=args.eval_interval,
        teacher_ckpt=args.teacher_ckpt,
        dry_run=args.dry_run,
    )
