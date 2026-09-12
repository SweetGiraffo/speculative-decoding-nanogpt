"""
Comprehensive Benchmarking Suite for Speculative Decoding.
Compares:
1. Standard Autoregressive Baseline (10.8M Target Model with KV-Cache)
2. Speculative Decoding (1.0M Draft + 10.8M Target with KV-Cache)

Calculates:
- Acceptance Rate (~65%)
- Inference Latency and Throughput (Tokens per Second)
- Effective Speedup Multiplier (~2.1x)
"""

import argparse
import os
import time
from typing import List

import torch

from config import get_draft_config, get_target_config
from model import GPT
from speculative import SpeculativeDecoder
from tokenizer import get_tokenizer


DEFAULT_PROMPTS = [
    "Once upon a time, in a small quiet forest, there lived a little bird named Pip.",
    "Lily loved to visit her grandmother's garden because it was full of colorful flowers.",
    "One sunny morning, Leo found a mysterious shiny golden key under an old oak tree.",
    "Tommy was excited because today his mother promised they would visit the grand zoo.",
]


def run_benchmark(
    target_ckpt: str = "checkpoints/target_10.8M.pt",
    draft_ckpt: str = "checkpoints/draft_1.0M.pt",
    prompts: List[str] = DEFAULT_PROMPTS,
    max_new_tokens: int = 128,
    gamma: int = 4,
    temperature: float = 0.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_plot: bool = False,
):
    print("=" * 70)
    print("   SPECULATIVE DECODING BENCHMARK SUITE")
    print(f"   Device: {device.upper()} | Gamma: {gamma} | Temperature: {temperature}")
    print("=" * 70)

    # 1. Load or Initialize Target Model (10.8M)
    target_cfg = get_target_config()
    target_model = GPT(target_cfg).to(device)
    if os.path.exists(target_ckpt):
        print(f"[+] Loading Target Model weights from {target_ckpt}")
        ckpt = torch.load(target_ckpt, map_location=device)
        target_model.load_state_dict(ckpt["model"])
    else:
        print(f"[!] Target checkpoint {target_ckpt} not found; benchmarking with initialized weights.")
    target_model.eval()

    # 2. Load or Initialize Draft Model (1.0M)
    draft_cfg = get_draft_config()
    draft_model = GPT(draft_cfg).to(device)
    if os.path.exists(draft_ckpt):
        print(f"[+] Loading Draft Model weights from {draft_ckpt}")
        ckpt = torch.load(draft_ckpt, map_location=device)
        draft_model.load_state_dict(ckpt["model"])
    else:
        print(f"[!] Draft checkpoint {draft_ckpt} not found; benchmarking with initialized weights.")
    draft_model.eval()

    decoder = SpeculativeDecoder(target_model, draft_model)
    tok = get_tokenizer("tinystories")

    results = []

    # Warmup pass
    warmup_tokens = torch.tensor([[1, 2, 3, 4]], device=device)
    _ = target_model.generate(warmup_tokens, max_new_tokens=4, use_cache=True)
    _, _ = decoder.generate(warmup_tokens, max_new_tokens=4, gamma=gamma, temperature=temperature)

    print("\n" + "-" * 70)
    print(f"{'Prompt #':<10} | {'Base (s)':<10} | {'Spec (s)':<10} | {'Accept Rate':<12} | {'Speedup':<10}")
    print("-" * 70)

    total_base_time = 0.0
    total_spec_time = 0.0
    total_accepted_tokens = 0
    total_proposed_tokens = 0

    for idx, prompt_text in enumerate(prompts):
        prompt_ids = tok.encode(prompt_text)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        # Baseline: Autoregressive target model alone
        _, base_time, base_tps = decoder.autoregressive_baseline(
            prompt_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        # Speculative: Draft (1.0M) + Target (10.8M)
        _, spec_metrics = decoder.generate(
            prompt_tensor,
            max_new_tokens=max_new_tokens,
            gamma=gamma,
            temperature=temperature,
        )

        speedup = base_time / max(spec_metrics.wall_clock_time_sec, 1e-6)
        spec_metrics.speedup_ratio = speedup

        total_base_time += base_time
        total_spec_time += spec_metrics.wall_clock_time_sec
        total_accepted_tokens += spec_metrics.total_draft_tokens_accepted
        total_proposed_tokens += spec_metrics.total_draft_tokens_proposed

        results.append({
            "prompt": prompt_text,
            "base_time": base_time,
            "spec_time": spec_metrics.wall_clock_time_sec,
            "speedup": speedup,
            "acceptance_rate": spec_metrics.acceptance_rate,
            "base_tps": base_tps,
            "spec_tps": spec_metrics.tokens_per_second,
        })

        print(
            f"Prompt {idx + 1:<4} | {base_time:<10.3f} | {spec_metrics.wall_clock_time_sec:<10.3f} | "
            f"{spec_metrics.acceptance_rate * 100:>10.2f}% | {speedup:>9.2f}x"
        )

    overall_acc_rate = (total_accepted_tokens / max(total_proposed_tokens, 1)) * 100
    overall_speedup = total_base_time / max(total_spec_time, 1e-6)

    print("-" * 70)
    print(f"Overall Benchmark Summary:")
    print(f"  Mean Token Acceptance Rate : {overall_acc_rate:.2f}%")
    print(f"  Total Baseline Time        : {total_base_time:.3f} s")
    print(f"  Total Speculative Time     : {total_spec_time:.3f} s")
    print(f"  Mean Effective Speedup     : {overall_speedup:.2f}x")
    print("=" * 70)

    if save_plot:
        try:
            import matplotlib.pyplot as plt
            prompts_labels = [f"P{i+1}" for i in range(len(prompts))]
            speedups = [r["speedup"] for r in results]
            acc_rates = [r["acceptance_rate"] * 100 for r in results]

            fig, ax1 = plt.subplots(figsize=(8, 4))
            color = "tab:blue"
            ax1.set_xlabel("Prompts")
            ax1.set_ylabel("Speedup Multiplier (x)", color=color)
            bars = ax1.bar(prompts_labels, speedups, color=color, alpha=0.7, width=0.4, label="Speedup")
            ax1.tick_params(axis="y", labelcolor=color)
            ax1.axhline(2.1, color="navy", linestyle="--", label="Target Goal (2.1x)")

            ax2 = ax1.twinx()
            color = "tab:orange"
            ax2.set_ylabel("Acceptance Rate (%)", color=color)
            lines = ax2.plot(prompts_labels, acc_rates, color=color, marker="o", linewidth=2, label="Acceptance Rate")
            ax2.tick_params(axis="y", labelcolor=color)
            ax2.axhline(65, color="darkorange", linestyle=":", label="Target Goal (65%)")

            plt.title("Speculative Decoding: Speedup and Acceptance Rate")
            fig.tight_layout()
            out_img = "benchmark_results.png"
            plt.savefig(out_img, dpi=200)
            print(f"[+] Benchmark chart saved to {out_img}")
        except Exception as e:
            print(f"[!] Could not plot: {e}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_ckpt", type=str, default="checkpoints/target_10.8M.pt")
    parser.add_argument("--draft_ckpt", type=str, default="checkpoints/draft_1.0M.pt")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--save_plot", action="store_true")
    args = parser.parse_args()

    run_benchmark(
        target_ckpt=args.target_ckpt,
        draft_ckpt=args.draft_ckpt,
        max_new_tokens=args.max_new_tokens,
        gamma=args.gamma,
        temperature=args.temperature,
        save_plot=args.save_plot,
    )
