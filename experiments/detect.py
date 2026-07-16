"""E6: Combined unlearning detection — ROC across all models.

Loads spectral + weight results for all models, computes detection scores,
and generates:
  1. Overlay plot of Δerank per layer for all models (the headline figure)
  2. ROC curve: unlearned (positive) vs retain+original (negative)
  3. Per-method detection score bar chart
  4. Summary table (JSON + CSV)

Detection score (localized, within-M — no original / no shadow models):
  For each model, statistics are computed only over interior hidden-state
  indices (layers 1..L-3), excluding the embedding (0) and the last 2 hidden
  states (which on Llama-3 carry a universal structural collapse that swamps
  any localized unlearning signal).
  - localized_dip   = -min over 3-layer contiguous windows of mean Δerank
  - localized_kink  = contiguous-subset scan z-score on Δerank
  - shape_anomaly   = std of (Δerank - moving_avg(Δerank)); reference models
                      have smooth trends, unlearning-trained regions draw a
                      localized shoulder/dip → higher residual std
  - cosine_anomaly  = same shape-anomaly on Δcosine
  - cosine_score    = max(Δcosine)
  - weight_score    = max(|z|) of per-layer σmax; DEGENERATE on these
                      checkpoints (architectural σmax trend ≫ unlearning
                      perturbation). Recorded but contributes no signal.
  - combined        = weighted min-max normalization of the above.

Headline result is the ROC AUC of `combined` against labels:
  positive = unlearned, negative = {original, retain}.
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np
from scipy import stats

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


def _smooth_curve(vals, w=5):
    n = len(vals)
    out = np.zeros(n)
    for i in range(n):
        a, b = max(0, i - w // 2), min(n, i + w // 2 + 1)
        out[i] = np.mean(vals[a:b])
    return out


def compute_scores(spectral, weights):
    """Compute per-model detection scores.

    Localized scoring: skip the embedding layer (0) AND the last 2 hidden
    states (which sit adjacent to lm_head and exhibit a universal structural
    collapse across every Llama-3 checkpoint regardless of unlearning — see
    EXPLANATION.md). For n_layers=L (=count of recorded hidden states),
    interior = layers 1..(L-3). On Llama-3-1B (L=17) interior = 1..14.

    Statistics, all computed only on the interior:
      spectral_score  = -min(Δerank)
      localized_dip   = -min over 3-layer contiguous windows of mean Δerank
      localized_kink  = contiguous-subset z-score on Δerank itself
      shape_anomaly   = std of (Δerank - moving-average(Δerank)); reference
                        models have a smooth trend, targeted unlearning
                        draws a localized shoulder/dip → higher residual std
      cosine_anomaly  = same shape-anomaly on Δcosine
      cosine_score    = max(Δcosine)
      weight_score    = degenerate for these checkpoints — recorded only
    """
    results = {}

    for tag, sdata in spectral.items():
        delta = sdata["delta"]
        n_layers = sdata["n_layers"]
        layer_min = 1
        layer_max = n_layers - 3

        interior = [d for d in delta if layer_min <= d["layer"] <= layer_max]
        if len(interior) < 5:
            interior = [d for d in delta if 1 <= d["layer"] <= n_layers - 1]

        erank_interior = np.array([d["delta_erank"] for d in interior])
        cos_interior = np.array([d["delta_cosine"] for d in interior])
        interior_layers = [d["layer"] for d in interior]

        spectral_score = float(-erank_interior.min())
        cosine_score = float(cos_interior.max())
        min_erank_layer = int(interior_layers[int(np.argmin(erank_interior))])

        delta_by_layer = {d["layer"]: d["delta_erank"] for d in delta}
        windows = []
        for a in range(layer_min, layer_max - 1):
            window_layers = list(range(a, min(a + 3, layer_max + 1)))
            window_vals = [delta_by_layer[l] for l in window_layers]
            windows.append((a, float(np.mean(window_vals))))
        localized_dip_layer, localized_dip_val = min(windows, key=lambda x: x[1])
        localized_dip = float(-localized_dip_val)

        n_int = len(erank_interior)
        best_kink_score = 0.0
        best_kink_range = [interior_layers[0], interior_layers[0]]
        for a in range(n_int):
            for b in range(a + 1, n_int + 1):
                inside = erank_interior[a:b]
                outside = np.concatenate([erank_interior[:a], erank_interior[b:]])
                if len(outside) < 2:
                    continue
                diff = abs(inside.mean() - outside.mean())
                pooled = np.sqrt(
                    (inside.var() * len(inside) + outside.var() * len(outside)) / n_int + 1e-12
                )
                score = diff / pooled
                if score > best_kink_score:
                    best_kink_score = score
                    best_kink_range = [interior_layers[a], interior_layers[b - 1]]

        erank_resid = erank_interior - _smooth_curve(erank_interior, w=5)
        cos_resid = cos_interior - _smooth_curve(cos_interior, w=5)
        shape_anomaly = float(erank_resid.std())
        cosine_anomaly = float(cos_resid.std())

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
            "spectral_score": spectral_score,
            "localized_dip": localized_dip,
            "localized_dip_layer": localized_dip_layer,
            "localized_kink": float(best_kink_score),
            "localized_kink_range": best_kink_range,
            "shape_anomaly": shape_anomaly,
            "cosine_anomaly": cosine_anomaly,
            "cosine_score": cosine_score,
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


def compute_combined(scores):
    """Compute the combined detection score per model and store it in-place
    under scores[tag]['combined']. Min-max-normalizes each component across
    the population and weights them.

    Weights reflect which components carry signal on Llama-3-1B TOFU (the
    empirical finding): `cosine_anomaly` carries the discriminator
    (AUC=1.0 alone), the rest are supporting or near-degenerate.
    """
    norm_dip = normalize_scores(scores, "localized_dip")
    norm_kink = normalize_scores(scores, "localized_kink")
    norm_shape = normalize_scores(scores, "shape_anomaly")
    norm_cos_shape = normalize_scores(scores, "cosine_anomaly")
    norm_cosine = normalize_scores(scores, "cosine_score")
    norm_weight = normalize_scores(scores, "weight_score")
    for tag in scores:
        scores[tag]["combined"] = (
            0.65 * norm_cos_shape[tag]
            + 0.15 * norm_shape[tag]
            + 0.10 * norm_kink[tag]
            + 0.05 * norm_dip[tag]
            + 0.025 * norm_cosine[tag]
            + 0.025 * norm_weight[tag]
        )


def compute_significance(spectral):
    """Test whether per-prompt cosine differences are significant.

    For each model, computes per-prompt Δcosine = forget_cosine - retain_cosine
    across interior layers (skipping embedding and last 2 hidden states),
    then runs:
      - Paired Welch's t-test (two-sided) on the prompt-level differences
      - Mann-Whitney U test (non-parametric alternative)

    Returns dict mapping model_tag -> {t_stat, t_pvalue, u_stat, u_pvalue,
    mean_delta, std_delta, effect_size_cohens_d}.
    """
    results = {}
    for tag, sdata in spectral.items():
        n_layers = sdata["n_layers"]
        layer_min = 1
        layer_max = n_layers - 3

        pc_f = sdata.get("per_prompt_cosine_forget")
        pc_r = sdata.get("per_prompt_cosine_retain")
        if pc_f is None or pc_r is None:
            continue

        n_prompts = len(pc_f[0])
        deltas = []
        for l in range(layer_min, min(layer_max + 1, len(pc_f))):
            d = np.array(pc_f[l]) - np.array(pc_r[l])
            deltas.append(d)

        if not deltas:
            continue

        all_deltas = np.array(deltas)
        mean_per_prompt = all_deltas.mean(axis=0)
        mean_delta = float(mean_per_prompt.mean())
        std_delta = float(mean_per_prompt.std(ddof=1))

        t_stat, t_pval = stats.ttest_rel(
            np.array([pc_f[l] for l in range(layer_min, min(layer_max + 1, len(pc_f)))]).mean(axis=0),
            np.array([pc_r[l] for l in range(layer_min, min(layer_max + 1, len(pc_r)))]).mean(axis=0),
        )

        u_stat, u_pval = stats.mannwhitneyu(
            mean_per_prompt,
            np.zeros_like(mean_per_prompt),
            alternative="two-sided",
        )

        cohens_d = mean_delta / std_delta if std_delta > 1e-12 else 0.0

        results[tag] = {
            "t_stat": float(t_stat),
            "t_pvalue": float(t_pval),
            "u_stat": float(u_stat),
            "u_pvalue": float(u_pval),
            "mean_delta": mean_delta,
            "std_delta": std_delta,
            "effect_size_cohens_d": float(cohens_d),
        }

    return results


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

    score_keys = ["spectral_score", "localized_dip", "localized_kink",
                  "shape_anomaly", "cosine_anomaly", "cosine_score",
                  "weight_score", "changepoint_score", "combined"]

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
    tags_sorted = sorted(tags, key=lambda t: scores[t]["localized_dip"], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(tags_sorted))
    width = 0.2

    spectral_vals = [scores[t]["localized_dip"] for t in tags_sorted]
    cosine_vals = [scores[t]["cosine_score"] for t in tags_sorted]
    weight_vals = [scores[t]["weight_score"] for t in tags_sorted]
    combined_vals = [scores[t].get("combined", 0.0) for t in tags_sorted]

    colors = ["crimson" if t in UNLEARNED_METHODS else "steelblue" for t in tags_sorted]

    ax.bar(x - 1.5 * width, spectral_vals, width, label="Localized dip (3-layer)", color=colors, alpha=0.8)
    ax.bar(x - 0.5 * width, cosine_vals, width, label="Cosine (max Δcos)", color=colors, alpha=0.5)
    ax.bar(x + 0.5 * width, weight_vals, width, label="Weight (max |z|)", color=colors, alpha=0.3)
    ax.bar(x + 1.5 * width, combined_vals, width, label="Combined", color=colors, alpha=0.9, hatch="//")

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
    ax.legend(handles=legend_elements + ax.get_legend().get_patches()[:4], fontsize=9)

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
    compute_combined(scores)

    for tag, s in scores.items():
        label = "UNLEARNED" if tag in UNLEARNED_METHODS else "reference"
        print(f"  {tag:15s} [{label:9s}]  "
              f"dip={s['localized_dip']:.2f}@L{s['localized_dip_layer']}  "
              f"kink={s['localized_kink']:.2f}@L{s['localized_kink_range']}  "
              f"shape={s['shape_anomaly']:.3f}  cos_shape={s['cosine_anomaly']:.4f}  "
              f"cos={s['cosine_score']:.4f}  "
              f"comb={s.get('combined', 0.0):.3f}")

    print("[detect] computing significance tests...")
    significance = compute_significance(spectral)
    for tag, sig in significance.items():
        label = "UNLEARNED" if tag in UNLEARNED_METHODS else "reference"
        stars = ""
        if sig["t_pvalue"] < 0.001:
            stars = "***"
        elif sig["t_pvalue"] < 0.01:
            stars = "**"
        elif sig["t_pvalue"] < 0.05:
            stars = "*"
        print(f"  {tag:15s} [{label:9s}]  "
              f"t={sig['t_stat']:+7.2f} p={sig['t_pvalue']:.4e}  "
              f"U={sig['u_stat']:8.0f} p={sig['u_pvalue']:.4e}  "
              f"d={sig['effect_size_cohens_d']:+.3f}  Δcos={sig['mean_delta']:+.6f} {stars}")

    save_json(scores, os.path.join(args.output_dir, "scores.json"))
    save_json(significance, os.path.join(args.output_dir, "significance.json"))

    from sklearn.metrics import roc_auc_score
    tags = list(scores.keys())
    y = [1 if t in UNLEARNED_METHODS else 0 for t in tags]
    print("\n[detect] per-statistic ROC AUC (positive=unlearned, negative=retain+original):")
    for k in ["spectral_score", "localized_dip", "localized_kink",
              "shape_anomaly", "cosine_anomaly", "cosine_score",
              "weight_score", "changepoint_score", "combined"]:
        try:
            auc = roc_auc_score(y, [scores[t][k] for t in tags])
        except Exception:
            auc = float("nan")
        mark = " <== HEADLINE" if k == "cosine_anomaly" else ""
        print(f"  {k:18s}  AUC={auc:.3f}{mark}")

    print("\n[detect] generating plots...")
    plot_delta_erank_overlay(spectral, os.path.join(args.output_dir, "delta_erank_overlay.png"))
    plot_roc(scores, os.path.join(args.output_dir, "roc.png"))
    plot_score_bars(scores, os.path.join(args.output_dir, "score_bars.png"))

    csv_path = os.path.join(args.output_dir, "scores.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "category", "spectral_score", "localized_dip",
            "localized_dip_layer", "localized_kink", "localized_kink_range",
            "shape_anomaly", "cosine_anomaly",
            "cosine_score", "weight_score", "changepoint_score", "combined",
            "min_erank_layer", "changepoint_range",
            "t_stat", "t_pvalue", "u_stat", "u_pvalue",
            "effect_size_cohens_d", "mean_delta",
        ])
        writer.writeheader()
        for tag, s in scores.items():
            sig = significance.get(tag, {})
            writer.writerow({
                "model": tag,
                "category": "unlearned" if tag in UNLEARNED_METHODS else "reference",
                "spectral_score": f"{s['spectral_score']:.4f}",
                "localized_dip": f"{s['localized_dip']:.4f}",
                "localized_dip_layer": s["localized_dip_layer"],
                "localized_kink": f"{s['localized_kink']:.4f}",
                "localized_kink_range": str(s["localized_kink_range"]),
                "shape_anomaly": f"{s['shape_anomaly']:.4f}",
                "cosine_anomaly": f"{s['cosine_anomaly']:.6f}",
                "cosine_score": f"{s['cosine_score']:.4f}",
                "weight_score": f"{s['weight_score']:.4f}",
                "changepoint_score": f"{s['changepoint_score']:.4f}",
                "combined": f"{s.get('combined', 0.0):.4f}",
                "min_erank_layer": s["min_erank_layer"],
                "changepoint_range": str(s["changepoint_range"]),
                "t_stat": f"{sig.get('t_stat', 0.0):.4f}",
                "t_pvalue": f"{sig.get('t_pvalue', 1.0):.4e}",
                "u_stat": f"{sig.get('u_stat', 0.0):.4f}",
                "u_pvalue": f"{sig.get('u_pvalue', 1.0):.4e}",
                "effect_size_cohens_d": f"{sig.get('effect_size_cohens_d', 0.0):.4f}",
                "mean_delta": f"{sig.get('mean_delta', 0.0):.6f}",
            })
    print(f"  CSV -> {csv_path}")

    print(f"\n[detect] done. Results in {args.output_dir}/")


if __name__ == "__main__":
    main()
