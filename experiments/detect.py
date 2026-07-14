"""E6: Combined unlearning detection — ROC across all models.

Loads spectral + weight results for all models, computes detection scores,
and generates:
  1. Overlay plot of Δerank per layer for all models (the headline figure)
  2. ROC curve: unlearned (positive) vs retain/original (negative)
  3. Per-method detection score bar chart
  4. Summary table (JSON + CSV)

The detection score combines:
  - spectral_score = -min(Δerank)  (high when forget domain rank collapsed)
  - weight_score   = max(|z|)      (high when weight outliers exist)
  - cosine_score   = max(Δcosine)  (high when forget activations collapsed)

Usage:
  # After running spectral.py and weights.py on all models:
  python detect.py --spectral_dir results/spectral/ --weight_dir results/weights/ --output_dir results/detection/
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_json, save_json, UNLEARNED_METHODS, REFERENCE_MODELS


def load_all_results(spectral_dir, weight_dir):
    """Load all spectral and weight JSON results."""
    spectral = {}
    for path in glob.glob(os.path.join(spectral_dir, "*_spectral.json")):
        data = load_json(path)
        spectral[data["model_tag"]] = data

    weights = {}
    for path in glob.glob(os.path.join(weight_dir, "*_weights.json")):
        data = load_json(path)
        weights[data["model_tag"]] = data

    return spectral, weights


def compute_scores(spectral, weights):
    """Compute per-model detection scores."""
    results = {}

    for tag, sdata in spectral.items():
        delta = sdata["delta"]
        spectral_score = -min(d["delta_erank"] for d in delta)
        cosine_score = max(d["delta_cosine"] for d in delta)
        min_erank_layer = min(delta, key=lambda d: d["delta_erank"])["layer"]

        wdata = weights.get(tag)
        if wdata:
            anomaly = wdata["anomaly"]
            weight_score = max(abs(z) for z in anomaly["z_scores"])
            cp_score = anomaly["changepoint_score"]
            cp_range = anomaly["changepoint_range"]
        else:
            weight_score = 0.0
            cp_score = 0.0
            cp_range = [0, 0]

        results[tag] = {
            "spectral_score": float(spectral_score),
            "cosine_score": float(cosine_score),
            "weight_score": float(weight_score),
            "changepoint_score": float(cp_score),
            "min_erank_layer": min_erank_layer,
            "changepoint_range": cp_range,
        }

    return results


def normalize_scores(scores, key):
    """Min-max normalize a score across all models."""
    vals = [s[key] for s in scores.values()]
    lo, hi = min(vals), max(vals)
    rng = hi - lo
    if rng < 1e-12:
        return {tag: 0.5 for tag in scores}
    return {tag: (s[key] - lo) / rng for tag, s in scores.items()}


def plot_delta_erank_overlay(spectral, output_path):
    """Overlay Δerank per layer for all models — the headline figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    colors_unlearned = plt.cm.Set1(np.linspace(0, 1, 10))
    color_idx = 0

    for tag, data in spectral.items():
        layers = [d["layer"] for d in data["delta"]]
        delta_erank = [d["delta_erank"] for d in data["delta"]]

        if tag in REFERENCE_MODELS:
            color = "black"
            linewidth = 2.5
            linestyle = "--"
        else:
            color = colors_unlearned[color_idx % len(colors_unlearned)]
            color_idx += 1
            linewidth = 1.5
            linestyle = "-"

        ax.plot(layers, delta_erank, "o-", label=tag, color=color,
                linewidth=linewidth, linestyle=linestyle, markersize=4)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Δ Effective Rank (forget − retain)", fontsize=12)
    ax.set_title("Per-Layer Rank Collapse: Unlearned vs Reference Models", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, ncol=2)
    ax.axhline(0, color="gray", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  overlay plot -> {output_path}")


def plot_roc(scores, output_path):
    """ROC curve: unlearned (positive) vs retain+original (negative)."""
    from sklearn.metrics import roc_curve, auc

    tags = list(scores.keys())
    y_true = np.array([1 if t in UNLEARNED_METHODS else 0 for t in tags])

    score_keys = ["spectral_score", "cosine_score", "weight_score", "changepoint_score"]
    combined_raw = {}
    norm_spectral = normalize_scores(scores, "spectral_score")
    norm_cosine = normalize_scores(scores, "cosine_score")
    norm_weight = normalize_scores(scores, "weight_score")
    for tag in tags:
        combined_raw[tag] = 0.4 * norm_spectral[tag] + 0.3 * norm_cosine[tag] + 0.3 * norm_weight[tag]
    scores["combined"] = combined_raw
    score_keys.append("combined")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))

    for key in score_keys:
        y_score = np.array([scores[t].get(key, 0.0) if isinstance(scores[t], dict) else scores[t][key]
                            for t in tags])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{key} (AUC={roc_auc:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Unlearning Detection ROC:\nUnlearned vs Reference Models", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ROC plot -> {output_path}")


def plot_score_bars(scores, output_path):
    """Per-method detection score bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tags = [t for t in scores if t not in ("combined",)]
    tags_sorted = sorted(tags, key=lambda t: scores[t]["spectral_score"], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(tags_sorted))
    width = 0.25

    spectral_vals = [scores[t]["spectral_score"] for t in tags_sorted]
    cosine_vals = [scores[t]["cosine_score"] for t in tags_sorted]
    weight_vals = [scores[t]["weight_score"] for t in tags_sorted]

    colors = ["crimson" if t in UNLEARNED_METHODS else "steelblue" for t in tags_sorted]

    ax.bar(x - width, spectral_vals, width, label="Spectral (-min Δerank)", color=colors, alpha=0.8)
    ax.bar(x, cosine_vals, width, label="Cosine (max Δcos)", color=colors, alpha=0.5)
    ax.bar(x + width, weight_vals, width, label="Weight (max |z|)", color=colors, alpha=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(tags_sorted, rotation=45, ha="right")
    ax.set_ylabel("Detection Score")
    ax.set_title("Per-Method Detection Scores", fontsize=14, fontweight="bold")
    ax.legend()

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="crimson", alpha=0.7, label="unlearned"),
        Patch(facecolor="steelblue", alpha=0.7, label="reference"),
    ]
    ax.legend(handles=legend_elements + ax.get_legend().get_patches()[:3], fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  score bars -> {output_path}")


def main():
    p = argparse.ArgumentParser(description="Combined unlearning detection.")
    p.add_argument("--spectral_dir", default="results/spectral")
    p.add_argument("--weight_dir", default="results/weights")
    p.add_argument("--output_dir", default="results/detection")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("[detect] loading results...")
    spectral, weights = load_all_results(args.spectral_dir, args.weight_dir)
    print(f"  spectral: {list(spectral.keys())}")
    print(f"  weights:  {list(weights.keys())}")

    if not spectral:
        print("No spectral results found. Run spectral.py first.")
        return

    print("[detect] computing scores...")
    scores = compute_scores(spectral, weights)

    for tag, s in scores.items():
        label = "UNLEARNED" if tag in UNLEARNED_METHODS else "reference"
        print(f"  {tag:15s} [{label:9s}]  spectral={s['spectral_score']:.3f}  "
              f"cosine={s['cosine_score']:.3f}  weight={s['weight_score']:.3f}  "
              f"cp={s['changepoint_score']:.3f}")

    save_json(scores, os.path.join(args.output_dir, "scores.json"))

    print("\n[detect] generating plots...")
    plot_delta_erank_overlay(spectral, os.path.join(args.output_dir, "delta_erank_overlay.png"))
    plot_roc(scores, os.path.join(args.output_dir, "roc.png"))
    plot_score_bars(scores, os.path.join(args.output_dir, "score_bars.png"))

    csv_path = os.path.join(args.output_dir, "scores.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "category", "spectral_score", "cosine_score",
            "weight_score", "changepoint_score", "min_erank_layer", "changepoint_range",
        ])
        writer.writeheader()
        for tag, s in scores.items():
            writer.writerow({
                "model": tag,
                "category": "unlearned" if tag in UNLEARNED_METHODS else "reference",
                "spectral_score": f"{s['spectral_score']:.4f}",
                "cosine_score": f"{s['cosine_score']:.4f}",
                "weight_score": f"{s['weight_score']:.4f}",
                "changepoint_score": f"{s['changepoint_score']:.4f}",
                "min_erank_layer": s["min_erank_layer"],
                "changepoint_range": str(s["changepoint_range"]),
            })
    print(f"  CSV -> {csv_path}")

    print(f"\n[detect] done. Results in {args.output_dir}/")


if __name__ == "__main__":
    main()
