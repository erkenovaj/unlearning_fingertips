"""E1/E2/E3: Per-layer spectral analysis of activations on forget vs retain.

For each model, collects last-token hidden states at every layer on forget and
retain prompts, then computes per layer:
  - effective rank of the activation covariance
  - spectral entropy
  - participation ratio
  - mean pairwise cosine similarity

The key signal is Δ(l) = stat(l, forget) - stat(l, retain):
  - unlearned model: localized anomaly at trained layers (rank collapse, cosine spike)
  - retain-only model: no anomaly (neither domain is special)
  - original model: no anomaly or opposite direction (memorized = richer)

Usage:
  python spectral.py --model original --output_dir results/spectral/
  python spectral.py --model rmu    --output_dir results/spectral/
  python spectral.py --model_path path/to/model --model_tag custom --output_dir results/spectral/
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    MODELS, DTYPE_MAP, get_device, load_tofu_prompts, format_prompts, load_model,
    collect_hidden_states, effective_rank, mean_cosine, save_json,
)


def compute_layer_stats(hidden_states):
    """Compute spectral statistics for each layer.

    hidden_states: (n_layers+1, n_prompts, hidden_dim)
    Returns: list of dicts per layer.
    """
    results = []
    for l in range(hidden_states.shape[0]):
        H = hidden_states[l]
        erank, entropy, pr = effective_rank(H)
        cos = mean_cosine(H)
        results.append({
            "layer": l,
            "erank": erank,
            "entropy": entropy,
            "participation_ratio": pr,
            "mean_cosine": cos,
        })
    return results


def plot_results(results, model_tag, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = [r["layer"] for r in results["forget"]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(layers, [r["erank"] for r in results["forget"]], "o-", label="forget", color="crimson")
    ax.plot(layers, [r["erank"] for r in results["retain"]], "s-", label="retain", color="steelblue")
    ax.set_xlabel("Layer"); ax.set_ylabel("Effective Rank"); ax.legend()
    ax.set_title("Effective Rank (forget vs retain)")

    ax = axes[0, 1]
    delta_erank = [f["erank"] - r["erank"] for f, r in zip(results["forget"], results["retain"])]
    ax.bar(layers, delta_erank, color=["crimson" if d < -0.5 else "steelblue" for d in delta_erank])
    ax.set_xlabel("Layer"); ax.set_ylabel("Δ Effective Rank")
    ax.set_title("Rank Collapse (forget − retain)\nNegative = forget domain collapsed")
    ax.axhline(0, color="black", linewidth=0.5)

    ax = axes[1, 0]
    ax.plot(layers, [r["mean_cosine"] for r in results["forget"]], "o-", label="forget", color="crimson")
    ax.plot(layers, [r["mean_cosine"] for r in results["retain"]], "s-", label="retain", color="steelblue")
    ax.set_xlabel("Layer"); ax.set_ylabel("Mean Cosine"); ax.legend()
    ax.set_title("Cosine Collapse (higher = more collapsed)")

    ax = axes[1, 1]
    ax.plot(layers, [r["participation_ratio"] for r in results["forget"]], "o-", label="forget", color="crimson")
    ax.plot(layers, [r["participation_ratio"] for r in results["retain"]], "s-", label="retain", color="steelblue")
    ax.set_xlabel("Layer"); ax.set_ylabel("Participation Ratio"); ax.legend()
    ax.set_title("Participation Ratio (low = dominated by few directions)")

    fig.suptitle(f"Spectral Analysis: {model_tag}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, f"{model_tag}_spectral.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  plot -> {path}")


def main():
    p = argparse.ArgumentParser(description="Per-layer spectral analysis of activations.")
    p.add_argument("--model", default=None, help="Key from MODELS registry in common.py")
    p.add_argument("--model_path", default=None, help="Direct HF model path")
    p.add_argument("--model_tag", default=None, help="Tag for output files (default: --model or derived)")
    p.add_argument("--output_dir", default="results/spectral")
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--forget_fraction", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--device", default=None)
    p.add_argument("--raw_prompts", action="store_true",
                   help="Use legacy 'Question: ...\\nAnswer:' format (default: chat template)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for prompt sampling")
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

    print(f"[spectral] model={tag}  path={model_path}")
    print(f"[spectral] device={device}  dtype={args.dtype}  samples={args.num_samples}")

    forget_qs, retain_qs = load_tofu_prompts(args.forget_fraction, args.num_samples, seed=args.seed)
    print(f"[spectral] forget={len(forget_qs)}  retain={len(retain_qs)}  raw_prompts={args.raw_prompts}  seed={args.seed}")

    model, tokenizer = load_model(model_path, dtype=dtype, device=device)

    forget_prompts = format_prompts(forget_qs, tokenizer, raw=args.raw_prompts)
    retain_prompts = format_prompts(retain_qs, tokenizer, raw=args.raw_prompts)

    print("[spectral] collecting forget activations...")
    forget_hs = collect_hidden_states(model, tokenizer, forget_prompts, device, args.batch_size)
    print(f"  shape: {forget_hs.shape}  (n_layers+1, n_prompts, hidden_dim)")

    print("[spectral] collecting retain activations...")
    retain_hs = collect_hidden_states(model, tokenizer, retain_prompts, device, args.batch_size)
    print(f"  shape: {retain_hs.shape}")

    print("[spectral] computing statistics...")
    forget_stats = compute_layer_stats(forget_hs)
    retain_stats = compute_layer_stats(retain_hs)

    delta = []
    for f, r in zip(forget_stats, retain_stats):
        delta.append({
            "layer": f["layer"],
            "delta_erank": f["erank"] - r["erank"],
            "delta_cosine": f["mean_cosine"] - r["mean_cosine"],
            "delta_pr": f["participation_ratio"] - r["participation_ratio"],
        })

    results = {
        "model_tag": tag,
        "model_path": model_path,
        "num_samples": args.num_samples,
        "forget_fraction": args.forget_fraction,
        "seed": args.seed,
        "raw_prompts": args.raw_prompts,
        "n_layers": len(forget_stats),
        "forget": forget_stats,
        "retain": retain_stats,
        "delta": delta,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, f"{tag}_spectral.json")
    save_json(results, json_path)
    print(f"[spectral] saved -> {json_path}")

    plot_results(results, tag, args.output_dir)

    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()

    min_delta_erank = min(d["delta_erank"] for d in delta)
    max_delta_cos = max(d["delta_cosine"] for d in delta)
    anomalous_layers = [d["layer"] for d in delta if d["delta_erank"] < -1.0]
    print(f"\n[spectral] summary for {tag}:")
    print(f"  min Δerank = {min_delta_erank:.4f}  (negative = rank collapse on forget)")
    print(f"  max Δcos   = {max_delta_cos:.4f}  (positive = cosine collapse on forget)")
    if anomalous_layers:
        print(f"  anomalous layers (Δerank < -1): {anomalous_layers}")
    else:
        print(f"  no anomalous layers detected")


if __name__ == "__main__":
    main()
