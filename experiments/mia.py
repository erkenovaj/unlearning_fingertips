"""MINT-inspired MIA metrics for unlearning detection.

Adapts token-level likelihood metrics from MINT (Koike et al., ICMLW 2026)
to detect whether a model has been unlearned on the TOFU benchmark.

Key insight from MINT: membership inference and machine-text detection share
the same optimal metric — the likelihood of a sample under the model. We apply
this to unlearning: if a model has truly forgotten, its likelihood on forget
prompts should drop (higher loss, lower probability tokens).

Per-prompt metrics computed:
  - loss:       negative cross-entropy on the full sequence
  - min_k_plus: mean standardized log-prob of lowest 20% tokens (MINT)
  - logrank:    mean log-rank of true tokens (lower rank = more predictable)
  - entropy:    mean next-token entropy (higher = more uncertain)

For each model, we compute these on forget and retain prompts, then:
  1. Per-prompt Δ = forget_score - retain_score
  2. Paired significance tests (Welch's t, Mann-Whitney U)
  3. Effect sizes (Cohen's d)

Usage:
  python mia.py --model original --output_dir results/mia/
  python mia.py --model rmu    --output_dir results/mia/
"""

import argparse
import os
import sys

import numpy as np
import torch
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    MODELS, DTYPE_MAP, get_device, load_tofu_prompts, format_prompts, load_model,
    compute_token_scores, save_json,
)


def main():
    p = argparse.ArgumentParser(description="MINT-inspired MIA metrics for unlearning detection.")
    p.add_argument("--model", default=None, help="Key from MODELS registry in common.py")
    p.add_argument("--model_path", default=None, help="Direct HF model path")
    p.add_argument("--model_tag", default=None, help="Tag for output files")
    p.add_argument("--output_dir", default="results/mia")
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--forget_fraction", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--device", default=None)
    p.add_argument("--raw_prompts", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.model_path:
        model_path = args.model_path
        tag = args.model_tag or args.model_path.split("/")[-1]
    elif args.model:
        model_path = MODELS[args.model]
        tag = args.model_tag or args.model
    else:
        p.error("Specify --model (registry key) or --model_path (HF path)")

    device = args.device or get_device()
    dtype = DTYPE_MAP[args.dtype]

    print(f"[mia] model={tag}  path={model_path}")
    print(f"[mia] device={device}  dtype={args.dtype}  samples={args.num_samples}")

    forget_qs, retain_qs = load_tofu_prompts(args.forget_fraction, args.num_samples, seed=args.seed)
    print(f"[mia] forget={len(forget_qs)}  retain={len(retain_qs)}  seed={args.seed}")

    model, tokenizer = load_model(model_path, dtype=dtype, device=device)

    forget_prompts = format_prompts(forget_qs, tokenizer, raw=args.raw_prompts)
    retain_prompts = format_prompts(retain_qs, tokenizer, raw=args.raw_prompts)

    print("[mia] computing forget scores...")
    forget_scores = compute_token_scores(model, tokenizer, forget_prompts, device, args.batch_size)
    print("[mia] computing retain scores...")
    retain_scores = compute_token_scores(model, tokenizer, retain_prompts, device, args.batch_size)

    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()

    metrics = {}
    for key in ["loss", "min_k_plus", "logrank", "entropy"]:
        f = forget_scores[key]
        r = retain_scores[key]
        delta = f - r

        t_stat, t_pval = sp_stats.ttest_rel(f, r, alternative="two-sided")
        u_stat, u_pval = sp_stats.mannwhitneyu(delta, np.zeros_like(delta), alternative="two-sided")
        cohens_d = float(delta.mean() / (delta.std(ddof=1) + 1e-12))

        metrics[key] = {
            "forget_mean": float(f.mean()),
            "forget_std": float(f.std(ddof=1)),
            "retain_mean": float(r.mean()),
            "retain_std": float(r.std(ddof=1)),
            "delta_mean": float(delta.mean()),
            "delta_std": float(delta.std(ddof=1)),
            "t_stat": float(t_stat),
            "t_pvalue": float(t_pval),
            "u_stat": float(u_stat),
            "u_pvalue": float(u_pval),
            "cohens_d": cohens_d,
            "per_prompt_forget": f.tolist(),
            "per_prompt_retain": r.tolist(),
        }

    results = {
        "model_tag": tag,
        "model_path": model_path,
        "num_samples": args.num_samples,
        "forget_fraction": args.forget_fraction,
        "seed": args.seed,
        "raw_prompts": args.raw_prompts,
        "metrics": metrics,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, f"{tag}_mia.json")
    save_json(results, json_path)
    print(f"[mia] saved -> {json_path}")

    print(f"\n[mia] summary for {tag}:")
    print(f"  {'metric':12s}  {'Δforget-retain':>14s}  {'t-stat':>8s}  {'p-value':>10s}  {'Cohen d':>8s}")
    print(f"  {'-'*12}  {'-'*14}  {'-'*8}  {'-'*10}  {'-'*8}")
    for key in ["loss", "min_k_plus", "logrank", "entropy"]:
        m = metrics[key]
        stars = ""
        if m["t_pvalue"] < 0.001:
            stars = "***"
        elif m["t_pvalue"] < 0.01:
            stars = "**"
        elif m["t_pvalue"] < 0.05:
            stars = "*"
        print(f"  {key:12s}  {m['delta_mean']:+14.6f}  {m['t_stat']:+8.2f}  {m['t_pvalue']:10.4e}  {m['cohens_d']:+8.3f}  {stars}")


if __name__ == "__main__":
    main()
