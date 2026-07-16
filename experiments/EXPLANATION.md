# Experiment Guide

## Scripts

| Script | Experiment | What it does | GPU needed? |
|---|---|---|---|
| `spectral.py` | E1/E2/E3 | Per-layer effective rank, cosine, spectral entropy of activations on forget vs retain. Saves per-prompt cosine for significance testing. | Yes (forward passes) |
| `weights.py` | E4/E5 | Weight stable rank per layer, outlier detection, change-point localization | No (weights only, but loading is faster on GPU) |
| `detect.py` | E6 | Combines spectral + weight scores into ROC, overlay plots, per-method table. Includes significance tests (Welch's t, Mann-Whitney U) on per-prompt cosine. | No (analysis only) |
| `mia.py` | E7 | MINT-inspired token-level MIA metrics: loss, Min-K%++, log-rank, entropy. Paired significance tests on forget vs retain. | Yes (forward passes) |
| `samples.py` | E8 | Generate and save text completions from each model on forget/retain prompts for qualitative inspection | Yes (generation) |
| `common.py` | — | Shared utilities, model registry, statistics, MIA helpers | — |

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

# Day 3: combined detection (E6) — includes significance testing
python detect.py --spectral_dir results/spectral/ --weight_dir results/weights/ --output_dir results/detection/

# Day 4: MINT-style MIA metrics (E7) — ~5 min per model
for m in original retain rmu npo graddiff altpo undial idknll; do
  python mia.py --model $m --output_dir results/mia/
done

# Day 5: model output samples (E8) — ~2 min per model (50 samples)
for m in original retain rmu npo graddiff altpo undial idknll; do
  python samples.py --model $m --output_dir results/samples/ --num_samples 50
done
```

To use a custom model:
```bash
python spectral.py --model_path path/to/model --model_tag my_model --output_dir results/spectral/
python mia.py     --model_path path/to/model --model_tag my_model --output_dir results/mia/
python samples.py --model_path path/to/model --model_tag my_model --output_dir results/samples/
```

## GPU requirements

**Minimum: any GPU with 4GB+ VRAM** (e.g., T4, GTX 1050 Ti).

The default model is **Llama-3.2-1B-Instruct** (1B params, ~2GB in bf16).
- Forward passes on 200 prompts × 2 domains: < 5 min per model.
- All 8 models: ~40 min total.
- Weight analysis: ~2 min per model (SVD on small matrices).
- MIA metrics: ~5 min per model (4 forward-pass metrics).
- Samples: ~2 min per model (50 generations).

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
- **Per-prompt cosine** — cosine of each prompt to the centroid (new in v2).
  Enables paired significance testing across forget/retain sets.

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

### detect.py — E6: Combined detection + significance testing

**What it measures:** Combines spectral and weight scores into a unified
detector. For each model:

- `spectral_score = -min(Δerank)` — high when forget domain rank collapsed
- `cosine_score = max(Δcosine)` — high when forget activations collapsed
- `weight_score = max(|z|)` — high when weight outliers exist
- `changepoint_score` — from contiguous-subset scan
- `combined = 0.65 * norm(cosine_anomaly) + 0.15 * norm(shape_anomaly) + ...`

**Significance testing (new):** Uses per-prompt cosine values saved by
`spectral.py` to run paired statistical tests:

- **Welch's t-test** (paired, two-sided): tests whether per-prompt Δcosine
  is significantly different from zero
- **Mann-Whitney U** (non-parametric): distribution-free alternative
- **Cohen's d**: effect size (small=0.2, medium=0.5, large=0.8)
- Significance stars: `*` p<0.05, `**` p<0.01, `***` p<0.001

**What you expect to see:**

- ROC AUC > 0.8 for the combined score → unlearning is reliably detectable
- Significant p-values (p < 0.05) for unlearned models → the Δcosine gap is
  not due to chance
- Large Cohen's d (> 0.5) → practically meaningful effect sizes
- Per-method breakdown: which methods are stealthier than others

**Impact:** This is the **practical deliverable**. The ROC answers the question
"can an adversary detect unlearning?" with a single number. The significance
tests add statistical rigor — we're not just eyeballing score differences.

### mia.py — E7: MINT-inspired MIA metrics

**What it measures:** Adapts token-level likelihood metrics from MINT
(Koike et al., ICMLW 2026) to detect unlearning. For each prompt, computes:

| Metric | What it measures | MINT source |
|--------|------------------|-------------|
| `loss` | Negative cross-entropy (higher = model predicts better) | Baseline |
| `min_k_plus` | Mean standardized log-prob of lowest 20% tokens | Min-K%++ |
| `logrank` | Mean log-rank of true tokens (lower rank = more predictable) | LogRank |
| `entropy` | Mean next-token entropy (higher = more uncertain) | Entropy |

**Key insight from MINT:** Membership inference and machine-text detection
share the same optimal metric — the likelihood of a sample under the model.
We apply this to unlearning: if a model has truly forgotten, its likelihood
on forget prompts should drop (higher loss, lower probability tokens).

**Significance testing:** Paired Welch's t-test and Mann-Whitney U on
per-prompt scores between forget and retain sets. Reports Cohen's d.

**What you expect to see:**

| Metric | Properly unlearned | Not unlearned | Interpretation |
|--------|-------------------|---------------|----------------|
| `loss` | significantly higher on forget | ≈ same on both | Model lost knowledge of forget domain |
| `min_k_plus` | significantly lower on forget | ≈ same | Forget tokens are less predictable |
| `logrank` | significantly lower on forget | ≈ same | True tokens rank lower in distribution |
| `entropy` | significantly higher on forget | ≈ same | Model is more uncertain on forget |

**Impact:** This provides a **complementary detection signal** to spectral.py.
While spectral.py looks at activation geometry (hidden states), mia.py looks
at output likelihood (next-token predictions). If both agree, the evidence
for unlearning detection is much stronger.

### samples.py — E8: Model output inspection

**What it measures:** Generates text completions from each model on forget
and retain prompts. Saves full prompt + completion pairs to JSON.

**Usage:**
```bash
python samples.py --model rmu --num_samples 50 --output_dir results/samples/
```

**What to look for in the output JSON:**

- **Forget prompts**: Does the unlearned model refuse to answer? Produce
  irrelevant responses? Give wrong answers? This tells you *how* unlearning
  manifests in generation quality.
- **Retain prompts**: Do unlearned models maintain quality on non-forget data?
  Degradation here would indicate catastrophic forgetting.
- **Cross-model comparison**: Do different unlearning methods produce different
  failure modes (refusal vs. hallucination vs. noise)?

**Impact:** Qualitative sanity check. Numbers (AUC, p-values) tell you
*whether* unlearning is detectable; samples tell you *why* and *how*.

## Output structure

```
results/
  spectral/
    original_spectral.json     # per-layer stats + per_prompt_cosine
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
    scores.csv                 # for the paper table (includes t/p values)
    significance.json          # per-model significance test results
  mia/
    original_mia.json          # loss/min_k/logrank/entropy per prompt
    rmu_mia.json
    ...
  samples/
    original_samples.json      # prompt + completion pairs
    rmu_samples.json
    ...
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
- E7 (mia.py): token-level likelihood also differs on forget vs retain (MINT-inspired)
- E8 (samples.py): qualitative inspection of generation quality differences

The **novelty** over prior MIA-against-unlearning work: self-referential
intra-model detection. No shadow models, no original model, no reference
model. The null comes from the model's own retain domain and untouched layers.

The **MINT connection** (E7): Koike et al. proved that MIA and machine-text
detection share the same optimal metric. We apply this to unlearning detection
— the forget domain plays the role of "non-member" data, and the retain domain
plays the role of "member" data. If the model has been properly unlearned, it
should treat forget prompts like non-members (higher loss, lower probability).
