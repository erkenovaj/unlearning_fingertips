"""E4/E5: Weight-only analysis — detect unlearning from weight statistics alone.

No forward passes, no data needed. For each model:
  - computes stable rank per weight matrix per layer
  - z-score outlier detection per layer
  - contiguous-subset scan to localize anomalous (unlearning-trained) layers

The key insight: targeted unlearning methods (RMU, NPO, etc.) modify only a few
layers, creating outliers in weight statistics. Full fine-tuning methods (GA)
create diffuse changes. Either way, the unlearned model's weight statistics
differ from what a naturally-trained model would show, and the anomaly is
detectable from the model alone — no original model needed for the z-score
outlier detection (it uses the model's own layers as the null distribution).

Usage:
  python weights.py --model original --output_dir results/weights/
  python weights.py --model rmu    --output_dir results/weights/
  python weights.py --model_path path/to/model --model_tag custom --output_dir results/weights/
"""

import argparse
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MODELS, DTYPE_MAP, get_device, load_model, stable_rank, save_json


def extract_layer_idx(name):
    """Extract decoder layer index from parameter name."""
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def compute_weight_stats(model):
    """Compute stable rank for each 2D weight matrix, grouped by layer.

    Returns: list of dicts per layer with per-matrix stats and layer-level mean.
    """
    layer_data = {}

    for name, param in model.named_parameters():
        if param.ndim != 2:
            continue
        layer_idx = extract_layer_idx(name)
        if layer_idx is None:
            continue
        W = param.detach().float().cpu().numpy()
        sr = stable_rank(W)
        layer_data.setdefault(layer_idx, []).append({
            "name": name,
            "shape": list(W.shape),
            "stable_rank": sr,
        })

    results = []
    for idx in sorted(layer_data.keys()):
        entries = layer_data[idx]
        srs = [e["stable_rank"] for e in entries]
        results.append({
            "layer": idx,
            "mean_stable_rank": float(np.mean(srs)),
            "std_stable_rank": float(np.std(srs)),
            "matrices": entries,
        })
    return results


def detect_anomalies(stats):
    """Z-score outlier detection + contiguous-subset scan.

    Z-score: each layer's mean stable rank vs the model-wide mean/std.
    Contiguous scan: find the contiguous layer block [a, b) whose mean
    differs most from the rest — this localizes the unlearning-trained region.
    """
    srs = np.array([s["mean_stable_rank"] for s in stats])
    n = len(srs)

    mu, sigma = srs.mean(), srs.std()
    z = ((srs - mu) / (sigma + 1e-12)).tolist()

    anomalous = [i for i, zi in enumerate(z) if abs(zi) > 2.0]

    best_score, best_a, best_b = 0.0, 0, 0
    for a in range(n):
        for b in range(a + 1, n + 1):
            inside = srs[a:b]
            outside = np.concatenate([srs[:a], srs[b:]])
            if len(outside) == 0:
                continue
            diff = abs(inside.mean() - outside.mean())
            pooled = np.sqrt(
                (inside.var() * len(inside) + outside.var() * len(outside)) / n + 1e-12
            )
            score = diff / pooled
            if score > best_score:
                best_score = score
                best_a, best_b = a, b

    return {
        "z_scores": z,
        "anomalous_layers": anomalous,
        "changepoint_score": float(best_score),
        "changepoint_range": [best_a, best_b],
        "changepoint_mean_sr": float(srs[best_a:best_b].mean()) if best_b > best_a else 0.0,
        "overall_mean_sr": float(mu),
        "overall_std_sr": float(sigma),
    }


def plot_results(stats, anomaly, model_tag, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = [s["layer"] for s in stats]
    srs = [s["mean_stable_rank"] for s in stats]
    z = anomaly["z_scores"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(layers, srs, "o-", color="steelblue", markersize=5)
    ax.set_xlabel("Layer"); ax.set_ylabel("Mean Stable Rank")
    ax.set_title("Weight Stable Rank per Layer")
    for i in anomaly["anomalous_layers"]:
        ax.axvline(i, color="crimson", alpha=0.3, linewidth=3)
    a, b = anomaly["changepoint_range"]
    if b > a:
        ax.axvspan(a - 0.5, b - 0.5, alpha=0.1, color="crimson", label="detected unlearning region")
        ax.legend()

    ax = axes[1]
    colors = ["crimson" if abs(zi) > 2 else "steelblue" for zi in z]
    ax.bar(layers, z, color=colors)
    ax.set_xlabel("Layer"); ax.set_ylabel("Z-score")
    ax.set_title("Stable Rank Z-scores (|z| > 2 = anomalous)")
    ax.axhline(2, color="crimson", linewidth=0.5, linestyle="--")
    ax.axhline(-2, color="crimson", linewidth=0.5, linestyle="--")

    fig.suptitle(f"Weight Analysis: {model_tag}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, f"{model_tag}_weights.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  plot -> {path}")


def main():
    p = argparse.ArgumentParser(description="Weight-only unlearning detection.")
    p.add_argument("--model", default=None, help="Key from MODELS registry in common.py")
    p.add_argument("--model_path", default=None, help="Direct HF model path")
    p.add_argument("--model_tag", default=None, help="Tag for output files")
    p.add_argument("--output_dir", default="results/weights")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--device", default=None)
    args = p.parse_args()

    if args.model_path:
        model_path = args.model_path
        tag = args.model_tag or args.model_path.split("/")[-1]
    elif args.model:
        model_path = MODELS[args.model]
        tag = args.model_tag or args.model
    else:
        p.error("Specify --model or --model_path")

    device = args.device or get_device()
    dtype = DTYPE_MAP[args.dtype]

    print(f"[weights] model={tag}  path={model_path}")
    print(f"[weights] device={device}  dtype={args.dtype}")

    model, _ = load_model(model_path, dtype=dtype, device=device)

    print("[weights] computing per-matrix stable rank...")
    stats = compute_weight_stats(model)
    print(f"  {len(stats)} layers, {sum(len(s['matrices']) for s in stats)} matrices")

    anomaly = detect_anomalies(stats)

    results = {
        "model_tag": tag,
        "model_path": model_path,
        "n_layers": len(stats),
        "stats": stats,
        "anomaly": anomaly,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, f"{tag}_weights.json")
    save_json(results, json_path)
    print(f"[weights] saved -> {json_path}")

    plot_results(stats, anomaly, tag, args.output_dir)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"\n[weights] summary for {tag}:")
    print(f"  anomalous layers (|z|>2): {anomaly['anomalous_layers']}")
    print(f"  changepoint score: {anomaly['changepoint_score']:.4f}")
    print(f"  changepoint range: layers {anomaly['changepoint_range']}")
    if anomaly["changepoint_score"] > 1.0:
        print(f"  -> localized weight anomaly detected (potential unlearning region)")
    else:
        print(f"  -> no strong localized anomaly (uniform or no unlearning)")


if __name__ == "__main__":
    main()
