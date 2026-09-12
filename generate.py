"""
Text Generation CLI tool supporting both Speculative Decoding and Standard Autoregressive Generation.
"""

import argparse
import os
import torch

from config import get_draft_config, get_target_config
from model import GPT
from speculative import SpeculativeDecoder
from tokenizer import get_tokenizer


def generate_text(
    prompt: str,
    target_ckpt: str = "checkpoints/target_10.8M.pt",
    draft_ckpt: str = "checkpoints/draft_1.0M.pt",
    mode: str = "speculative",
    max_new_tokens: int = 100,
    gamma: int = 4,
    temperature: float = 0.8,
    top_k: int = 40,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print(f"[*] Initializing models on {device}...")
    target_cfg = get_target_config()
    target_model = GPT(target_cfg).to(device)

    if os.path.exists(target_ckpt):
        ckpt = torch.load(target_ckpt, map_location=device)
        target_model.load_state_dict(ckpt["model"])
        print(f"[+] Loaded target weights from {target_ckpt}")

    tok = get_tokenizer("tinystories")
    prompt_ids = tok.encode(prompt)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    print(f"\n--- Prompt ---\n{prompt}\n")

    if mode.lower() == "speculative":
        draft_cfg = get_draft_config()
        draft_model = GPT(draft_cfg).to(device)
        if os.path.exists(draft_ckpt):
            d_ckpt = torch.load(draft_ckpt, map_location=device)
            draft_model.load_state_dict(d_ckpt["model"])
            print(f"[+] Loaded draft weights from {draft_ckpt}")

        decoder = SpeculativeDecoder(target_model, draft_model)
        out_tokens, metrics = decoder.generate(
            prompt_tensor,
            max_new_tokens=max_new_tokens,
            gamma=gamma,
            temperature=temperature,
            top_k=top_k,
        )
        gen_text = tok.decode(out_tokens[0].tolist())
        print(f"--- Generated Story (Speculative Decoding) ---\n{gen_text}\n")
        print(metrics.summary())

    else:
        out_tokens = target_model.generate(
            prompt_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            use_cache=True,
        )
        gen_text = tok.decode(out_tokens[0].tolist())
        print(f"--- Generated Story (Autoregressive Baseline) ---\n{gen_text}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Once upon a time, there was a tiny blue bird named Pip.")
    parser.add_argument("--mode", type=str, default="speculative", choices=["speculative", "standard"])
    parser.add_argument("--target_ckpt", type=str, default="checkpoints/target_10.8M.pt")
    parser.add_argument("--draft_ckpt", type=str, default="checkpoints/draft_1.0M.pt")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()

    generate_text(
        prompt=args.prompt,
        target_ckpt=args.target_ckpt,
        draft_ckpt=args.draft_ckpt,
        mode=args.mode,
        max_new_tokens=args.max_new_tokens,
        gamma=args.gamma,
        temperature=args.temperature,
        top_k=args.top_k,
    )
