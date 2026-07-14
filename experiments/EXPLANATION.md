# Experiment Guide

## Scripts

| Script | Experiment | What it does | GPU needed? |
|---|---|---|---|
| `spectral.py` | E1/E2/E3 | Per-layer effective rank, cosine, spectral entropy of activations on forget vs retain | Yes (forward passes) |
| `weights.py` | E4/E5 | Weight stable rank per layer, outlier detection, change-point localization | No (weights only, but loading is faster on GPU) |
| `detect.py` | E6 | Combines spectral + weight scores into ROC, overlay plots, per-method table | No (analysis only) |
| `common.py` | — | Shared utilities, model registry, statistics | — |

## How to run

```bash
cd experiments

# Day 1: spectral analysis (E1/E2/E3) — ~5 min per model on 1B
for m in original retain rmu npo graddiff altpo undial idknll; do
  python spectral.py --model $m --output_dir results/spectral/
done

# Day 2: weight analysis (E4/E5) — ~2 min per model
for m in original retain rmu npo graddiff altpo undial idknll; do
  python weights.py --model $m --output_dir results/weights/
done

# Day 3: combined detection (E6) — seconds
python detect.py --spectral_dir results/spectral/ --weight_dir results/weights/ --output_dir results/detection/
```

To use a custom model:
```bash
python spectral.py --model_path path/to/model --model_tag my_model --output_dir results/spectral/
python weights.py   --model_path path/to/model --model_tag my_model --output_dir results/weights/
```

## GPU requirements

**Minimum: any GPU with 4GB+ VRAM** (e.g., T4, GTX 1050 Ti).

The default model is **Llama-3.2-1B-Instruct** (1B params, ~2GB in bf16).
- Forward passes on 200 prompts × 2 domains: < 5 min per model.
- All 8 models: ~40 min total.
- Weight analysis: ~2 min per model (SVD on small matrices).

If you want to use 7B models instead (Llama-2-7B-chat):
- Need 24GB+ VRAM (3090, 4090, A5000, A100-40GB).
- ~14GB weights in bf16, ~18GB with activations.
- Forward passes: ~20 min per model.
- Add `--model_path open-unlearning/tofu_Llama-2-7b-chat_full` etc.

**Rented server recommendation:** A single A100 40GB or 3090 24GB.
For 1B models: even a 4GB T4 works. Total cloud cost: < $5.

## What each experiment shows

### spectral.py — E1/E2/E3: Activation spectral analysis

**What it measures:** For each transformer layer, collects last-token hidden
states on 200 forget-domain and 200 retain-domain prompts. Computes:

- **Effective rank** — `exp(H(eigvals))` of the activation covariance. Measures
  how many dimensions the activations span. Low = collapsed.
- **Mean cosine** — average pairwise cosine between activation vectors. High =
  all prompts produce similar hidden states (collapsed).
- **Participation ratio** — `1/Σp²` where `p` = normalized eigenvalues. Low =
  one direction dominates.

**Key output:** `Δ(l) = stat(l, forget) − stat(l, retain)` per layer.

**What you expect to see:**

| Model type | Δ erank | Δ cosine | Interpretation |
|---|---|---|---|
| Unlearned (RMU) | sharp negative dip at trained layers | positive spike | RMU forces forget hiddens toward a random vector → rank collapses, cosine spikes |
| Unlearned (NPO/GradDiff) | moderate negative | slight positive | Gradient-based methods reduce forget representation richness, less sharply |
| Retain-only | ≈ 0 across all layers | ≈ 0 | No domain is special — genuine ignorance is uniform |
| Original | ≈ 0 or slightly positive | ≈ 0 or slightly negative | Memorized data has richer (higher-rank) representations |

**Impact:** This is the **headline result**. If Δerank shows a localized dip
in unlearned models but not in the retain-only model, it proves:
1. Unlearning leaves a **domain-specific geometric trace** in activations.
2. The trace is **detectable from a single model** — no original model needed.
3. The trace is **localized** to specific layers, unlike genuine ignorance.

The overlay plot (`delta_erank_overlay.png`) is the single most compelling
figure: all models' Δerank curves on one plot, with reference models in black
dashed lines and unlearned models in color.

### weights.py — E4/E5: Weight-only analysis

**What it measures:** For each 2D weight matrix in each layer, computes
**stable rank** = `‖W‖²_F / ‖W‖²_2` (Frobenius² / spectral²). This is a
shape-invariant measure of effective rank that doesn't need SVD of the full
matrix.

Then:
- **Z-score** each layer's mean stable rank against the model-wide distribution.
  Layers with |z| > 2 are anomalous.
- **Contiguous-subset scan**: find the layer block [a, b) whose mean stable
  rank differs most from the rest. This localizes the unlearning-trained region.

**What you expect to see:**

| Model type | Z-scores | Changepoint | Interpretation |
|---|---|---|---|
| Unlearned (RMU) | outliers at layers 3-7 (trained layers) | high score, narrow range | RMU trains 3 layers → those layers' weights are anomalous |
| Unlearned (NPO/GradDiff) | outliers spread across more layers | moderate score, wider range | Full-FT methods modify more layers diffusely |
| Retain-only | no outliers | low score | Naturally trained → uniform weight statistics |
| Original | no outliers | low score | Same as retain — no unlearning perturbation |

**Impact:** This is the **cheapest signal** — no forward passes, no data needed.
If it works, an adversary can detect unlearning by just downloading the weights
and computing SVD norms. The change-point localization also **identifies which
layers were unlearning-trained**, which is a step toward method fingerprinting.

### detect.py — E6: Combined detection

**What it measures:** Combines spectral and weight scores into a unified
detector. For each model:

- `spectral_score = -min(Δerank)` — high when forget domain rank collapsed
- `cosine_score = max(Δcosine)` — high when forget activations collapsed
- `weight_score = max(|z|)` — high when weight outliers exist
- `changepoint_score` — from contiguous-subset scan
- `combined = 0.4 * norm(spectral) + 0.3 * norm(cosine) + 0.3 * norm(weight)`

Then generates:
- **ROC curve**: unlearned models (positive) vs retain + original (negative)
- **Per-method bar chart**: scores for each model, colored by category
- **Overlay plot**: Δerank per layer for all models on one figure
- **CSV table**: all scores for the paper

**What you expect to see:**

- ROC AUC > 0.8 for the combined score → unlearning is reliably detectable
- AUC ≈ 0.5 → unlearning is stealthy (also a publishable result: "current
  methods are undetectable from single-model weight/activation statistics")
- Per-method breakdown: which methods are stealthier than others

**Impact:** This is the **practical deliverable**. The ROC answers the question
"can an adversary detect unlearning?" with a single number. The per-method
table shows which methods are more detectable, guiding future unlearning
research toward stealthier approaches (or showing that current methods are
already stealthy enough).

## Output structure

```
results/
  spectral/
    original_spectral.json     # per-layer stats
    original_spectral.png      # forget vs retain plots
    rmu_spectral.json
    rmu_spectral.png
    ...
  weights/
    original_weights.json      # per-layer stable rank + anomaly detection
    original_weights.png       # stable rank + z-score plots
    rmu_weights.json
    rmu_weights.png
    ...
  detection/
    delta_erank_overlay.png    # THE headline figure
    roc.png                    # ROC curve
    score_bars.png             # per-method comparison
    scores.json                # all scores
    scores.csv                 # for the paper table
```

## Models used (OpenUnlearning TOFU, Llama-3.2-1B-Instruct)

All from `open-unlearning` HuggingFace org:

| Tag | HF path | Role |
|---|---|---|
| `original` | `tofu_Llama-3.2-1B-Instruct_full` | Memorized all TOFU data |
| `retain` | `tofu_Llama-3.2-1B-Instruct_retain90` | Never saw forget10 — genuine ignorance baseline |
| `rmu` | `unlearn_..._RMU_lr1e-05_layer5_scoeff100_epoch10` | Representation misdirection |
| `npo` | `unlearn_..._NPO_lr5e-05_beta0.1_alpha2_epoch10` | Negative preference optimization |
| `graddiff` | `unlearn_..._GradDiff_lr1e-05_alpha5_epoch10` | Gradient difference |
| `altpo` | `unlearn_..._AltPO_lr5e-05_beta0.1_alpha1_epoch10` | Alternative preference optimization |
| `undial` | `unlearn_..._UNDIAL_lr0.0001_beta10_alpha1_epoch10` | Unlearning via dialing |
| `idknll` | `unlearn_..._IdkNLL_lr4e-05_alpha5_epoch10` | "I don't know" NLL |

Add more by editing `MODELS` in `common.py`.

## The central thesis (what to write up)

Unlearning is a **localized perturbation** (a few layers, a restricted
optimization); "never trained on D" is a **global property**. So an unlearned
model carries a **domain-localized geometric signature** on the forget domain
that genuine ignorance does not. An adversary with only the unlearned model
(no original, no shadow models) can detect this by comparing the forget
domain's representations against a retain domain **inside the model itself**.

The experiments test this thesis:
- E1/E2/E3 (spectral.py): activation geometry differs on forget vs retain
- E4/E5 (weights.py): weight statistics have localized anomalies
- E6 (detect.py): combining these detects unlearning without the original model

The **novelty** over prior MIA-against-unlearning work: self-referential
intra-model detection. No shadow models, no original model, no reference
model. The null comes from the model's own retain domain and untouched layers.
