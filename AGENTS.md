# AGENTS.md

Notes for OpenCode sessions working in this repo. High-signal facts only.

## What this repo is

Research repo on detecting whether a model has unlearned a domain, using only
the model itself (no original, no shadow models) against the TOFU suite
(`locuslab/TOFU`) and OpenUnlearning's Llama-3.2-1B-Instruct checkpoints.

- No CI, no tests, no lint, no typecheck, no codegen. Do not invent commands.
- Python 3.12, torch 2.8 + CUDA. Install: `uv sync` (preferred) or `pip install -r requirements.txt`.
- `PROPOSAL.md` = thesis/design. `experiments/EXPLANATION.md` = canonical runbook — read it before running anything.

## Layout (which dirs matter)

- `experiments/` — **detection pipeline (current main work).** `common.py` (model registry + stats + I/O), `spectral.py` (E1–E3, GPU forward passes), `weights.py` (E4–E5, weights only), `detect.py` (E6, combines prior outputs), `mia.py` (E7, MINT-inspired token-level MIA metrics), `samples.py` (E8, text completions for qualitative inspection). Run from this dir. `common.py` also has `STRONG_MODELS` registry for strengthened-finetune models — use `--registry strong` with detect.py/spectral.py/weights.py.
- `train_pipeline.py` — **training pipeline (weak finetuning).** Resumable script that clones `open-unlearning/`, finetunes base models on TOFU, unlearns with 5 methods (GradAscent, GradDiff, NPO, RMU, SimNPO), uploads checkpoints to HF, generates samples. Targets Qwen2.5-1.5B, Phi-3.5-mini, Qwen2.5-3B (NOT the Llama-3.2-1B used by experiments/). Tracked in `train_manifest.json`; output goes to `train_output/samples/`. Uses `FINETUNE_EPOCHS=5`, `lr=1e-5` — produces weak learning (loss plateau ~4.5, model_utility ~0.34-0.44).
- `train_strong_pipeline.py` — **training pipeline (strong finetuning).** Same structure but with `FINETUNE_EPOCHS=15`, `lr=5e-5`, `warmup_epochs=2`. Tracked in `train_strong_manifest.json`; output goes to `train_output/strong_samples/`. Checkpoints uploaded as `erkenovaj/strong_tofu_*` and `erkenovaj/unlearn_strong_tofu_*`.
- `fast_experiments/` — **legacy**, older Phi-1.5 / Zephyr-7b probe approach. Not the current line of work; don't update unless explicitly asked.
- `logs_tofu/` — pre-cached `.npz`/`.json` TOFU splits and a prior MLP-probe run. Read-only cache.
- `experiments/results/` — detection pipeline output (gitignored). Subdirectories: `spectral/`, `weights/`, `detection/` (original); `spectral_v2/`, `weights_v2/`, `detection_v2/` (v2 with chat-template); `spectral_v2_seed99/` (seed-99 ablation); `mia/`, `samples/` (E7/E8); `*_phi/` variants (Phi-1.5 models).

## Canonical commands

### Detection pipeline (experiments/)

Run from the repo root with `workdir=/root/unlearning_fingertips/experiments`:

```
# IMPORTANT: requires HF_HOME=/root/.cache/huggingface/ HF_HUB_DISABLE_XET=1
# (see GOTCHA section — default HF cache path fills quota after a few downloads).

# spectral (E1/E2/E3), ~5 min/model on 1B
for m in original retain rmu npo graddiff altpo undial idknll; do
  python spectral.py --model $m --output_dir results/spectral/
done

# weights (E4/E5), ~8 min/model (full SVD on MLP matrices)
for m in original retain rmu npo graddiff altpo undial idknll; do
  python weights.py --model $m --output_dir results/weights/
done

# detection (E6) — must run AFTER spectral+weights complete for all 8
python detect.py --spectral_dir results/spectral/ --weight_dir results/weights/ --output_dir results/detection/

# MINT-style MIA metrics (E7), ~5 min/model
for m in original retain rmu npo graddiff altpo undial idknll; do
  python mia.py --model $m --output_dir results/mia/
done

# model output samples (E8), ~2 min/model
for m in original retain rmu npo graddiff altpo undial idknll; do
  python samples.py --model $m --output_dir results/samples/ --num_samples 50
done
```

Key constraints:
- `spectral.py` / `mia.py` / `samples.py` need a CUDA GPU.
- `weights.py` is weights-only; full SVD takes ~8 min per model (CPU-bound numpy).
- `detect.py` globs `*_spectral.json` / `*_weights.json` in its input dirs. If you rename a `--model_tag`, both pipelines must use the same tag or detect.py silently drops that model.
- `spectral.py` accepts `--seed N` (default 42) and `--raw_prompts` (legacy `Question:..\\nAnswer:` format — do NOT use for primary runs; the models were trained with the chat template, raw prompts mask the unlearning signal).

### Training pipeline (train_pipeline.py)

```
python train_pipeline.py --status              # show manifest summary
python train_pipeline.py --phase 0             # base model samples only
python train_pipeline.py --phase 1             # finetune originals (TOFU full)
python train_pipeline.py --phase 2             # finetune retains (TOFU retain90)
python train_pipeline.py --phase 23            # eval retains (for unlearning metrics)
python train_pipeline.py --phase 3             # unlearn all methods
python train_pipeline.py --phase 3 --model Qwen2.5-1.5B-Instruct --method RMU  # single combo
python train_pipeline.py --reset STEP_KEY      # mark a step as pending
```

Key constraints:
- Requires `open-unlearning/` directory (cloned automatically on first run, or `git clone` manually).
- Requires CUDA GPU and HF upload credentials (checkpoints are uploaded to `erkenovaj/*`).
- `--phase` runs sequentially through phases if omitted (0→1→2→23→3); phases check `train_manifest.json` and skip completed steps.
- RMU layer targets are model-specific (set in `RMU_LAYER` dict in train_pipeline.py).

### Strengthened training pipeline (train_strong_pipeline.py)

```
python train_strong_pipeline.py --status       # show manifest summary
python train_strong_pipeline.py --phase 1      # finetune originals (strong: 15ep, lr=5e-5)
python train_strong_pipeline.py --phase 2      # finetune retains
python train_strong_pipeline.py --phase 23     # eval retains
python train_strong_pipeline.py --phase 3      # unlearn all methods
python train_strong_pipeline.py --reset STEP_KEY
```

Key constraints:
- Same dependencies as train_pipeline.py; shares the same `open-unlearning/` clone.
- Uses its own manifest (`train_strong_manifest.json`) and output dir (`train_output/strong_samples/`).
- Checkpoints uploaded as `erkenovaj/strong_tofu_*` / `erkenovaj/unlearn_strong_tofu_*`.
- `open-unlearning/` has a patch in `src/evals/metrics/utils.py` (`.float()` before `.cpu()`) — see GOTCHA section.

## Headline result (on disk under `experiments/results/`)

Verified across two seeds (42, 99) with the v2 pipeline (chat-template + localized/dip/kink/shape scores):

| Statistic | seed 42 AUC | seed 99 AUC | Notes |
|---|---|---|---|
| `cosine_anomaly` | **1.000** | **0.917** | HEADLINE: residual std of Δcosine after 5-wide MA; separates all unlearned from {original, retain} cleanly except RMU on seed 99 |
| `shape_anomaly` (Δerank) | 0.500 | 0.833 | Less reliable; catches AltPO and IdkNLL only |
| `localized_kink` | 0.417 | 0.833 | Contiguous-block scan on Δerank |
| `combined` | 0.833 | 0.917 | Weighted blend dominated by `cosine_anomaly` |
| `weight_score` / `changepoint_score` | 0.167 | 0.167 | **Negative**: per-layer weight statistics (stable_rank AND σmax) are degenerate on Llama-3-1B TOFU — architectural trend ≫ unlearning perturbation; do not rely on E4/E5 alone. |

Caveat: with only 2 reference models (original, retain), AUC variance is high; multi-seed runs (e.g. `--seed 99`) are the minimum sanity check before claiming a separation.

## v2 pipeline changes

- `common.py`: `load_tofu_prompts` returns raw questions; `format_prompts(questions, tokenizer, raw=False)` applies chat template. `collect_hidden_states` uses left-padding so last real token is always at position `-1`. `max_length` bumped to 512. `spectral_metrics(W)` returns `(stable_rank, sigma_max, sigma_min)`.
- `spectral.py`: consumes `format_prompts`; `--seed N` and `--raw_prompts` flags.
- `weights.py`: per-matrix `sigma_max` + `sigma_min`; `detect_anomalies` runs on `mean_sigma_max` by default.
- `detect.py`: interior-only layer range (skip embedding + last 2 hidden states). Per-model scores: `localized_dip`, `localized_kink`, `shape_anomaly`, `cosine_anomaly`. `combined` weighted by `cosine_anomaly` (0.65 weight).
- Original pipeline output preserved in `experiments/results/` (raw prompts, global-min Δerank, stable_rank weights only). v2 writes to `spectral_v2/`, `weights_v2/`, `detection_v2/`.

## GOTCHA: HF cache disk-quota error masquerades as "model not found"

`HF_HOME=/workspace/.cache/huggingface/` lives on a shared network mount
(`mfs#...:/workspace`, ~1PB, per-user-quota'd). The `hf-xet` downloader
(`HF_XET_HIGH_PERFORMANCE=1`, `hf-xet 1.5.1`) errors with:

```
RuntimeError: ... IO Error: Disk quota exceeded (os error 122)
OSError: Can't load the model for 'open-unlearning/...'. ... make sure '...' is the correct path
```

The wrapping `OSError` quotes huggingface_hub's stale "wrong repo id" message —
don't chase the HF path. Symptom signature: the first few models download &
cache fine, then the rest fail identically. Real cause: per-user quota on the
network share is exhausted by the accumulated xet cache.

Fix options (verify space first with `df -h /workspace`):
- Move cache off the share: `export HF_HOME=/root/.cache/huggingface/` (local overlay, enough for the 8 ~2GB 1B checkpoints one at a time).
- Disable xet: `export HF_HUB_DISABLE_XET=1` and `unset HF_XET_HIGH_PERFORMANCE` — uses plain HF transfer, sometimes avoids the per-file reconstruction quota trigger.
- Clear stale xet chunks: `rm -rf /workspace/.cache/huggingface/xet` (safe; re-downloads what's missing).

After the cache is healthy the download resumes without touching HF repo ids.

**`train_pipeline.py` still hardcodes `HF_HOME=/workspace/.cache/huggingface`** in `OU_ENV` (line ~112) and `generate_samples` (line ~222). Either fix these or set `HF_HOME` in your shell before running.

## GOTCHA: `open-unlearning` needs a patch for bfloat16 eval

`src/evals/metrics/utils.py:98-99` calls `.cpu().numpy()` on bfloat16 tensors
which crashes. Patch: add `.float()` before `.cpu()`:

```python
avg_losses = avg_losses.float().cpu().numpy().tolist()
normalized_probs = normalized_probs.float().cpu().numpy().tolist()
```

Patch saved as `/root/unlearning_fingertips/open-unlearning.patch`.
Apply after re-cloning: `cd open-unlearning && git apply ../open-unlearning.patch`

Also, `transformers_modules/.../modeling_phi3.py` in the HF cache needs
DynamicCache compat shims for Phi-3.5 — see cached file at
`/workspace/.cache/huggingface/modules/transformers_modules/microsoft/Phi_hyphen_3_dot_5_hyphen_mini_hyphen_instruct/.../modeling_phi3.py`

## GOTCHA: `idknll` checkpoint discrepancy between `common.py` and `EXPLANATION.md`

| Source | `idknll` HF id ends in |
|---|---|
| `experiments/common.py` (committed) | `IdkNLL_lr3e-05_alpha5_epoch10` |
| `experiments/EXPLANATION.md` table | `IdkNLL_lr4e-05_alpha5_epoch10` |

**Both ids return HTTP 200 on `/api/models/...`** — both checkpoints exist on HF.
Don't "fix" one to match the other without confirming which checkpoint the
researcher intends; the two `lr` values are different unlearning runs. Pick one,
run spectral/weights with it, and note the choice in results. The actual code
uses `common.py` — `EXPLANATION.md` is stale.

## Finetuning quality finding

The standard TOFU finetuning setup (5 epochs, lr=1e-5) produces **weak learning**:
- Loss plateaus at ~4.5 after epoch 1 (from 11.66 start)
- `model_utility` ~0.34-0.44 (vs 0.1 random baseline)
- `forget_Q_A_Prob` / `retain_Q_A_Prob` ~0.27-0.34 (model assigns ~30% probability to correct answers)
- `retain_Q_A_ROUGE` ~0.35-0.40

This is consistent across Qwen2.5-1.5B, Phi-3.5-mini, and Qwen2.5-3B.
The strong pipeline (15 epochs, lr=5e-5) is intended to fix this.
See `train_strong_pipeline.py`.

On the existing Llama-3.2-1B detection results: `retain` (finetuned on retain90 only)
has `cos_anomaly=0.002027, combined=0.1543`. `rmu` (unlearned) has
`cos_anomaly=0.002086, combined=0.0900` — RMU is closer to original than to retain.
Only aggressive methods (altpo=0.9234, npo=0.6418) clearly separate from retain.
The detection may be measuring weight perturbation magnitude rather than knowledge removal.

## Repo code conventions to preserve

- `common.py` does `sys.path.insert(0, dirname(__file__))`; the experiment
  scripts rely on being run as `python spectral.py` from inside `experiments/`
  (or with `experiments/` on `PYTHONPATH`). Imports use `from common import ...`.
  Don't refactor to a package layout without rewiring this.
- matplotlib always uses `Agg` backend inside plot functions (headless).
- No inline code comments anywhere — match that style.

## Doing verification here

There is no test suite. To verify edits to the pipeline:
1. Re-run `spectral.py` / `weights.py` on one model (e.g. `--model rmu`) and
   diff the resulting JSON against the prior run in `experiments/results/` —
   statistics are deterministic for a fixed seed (built-in `seed=42` in
   `load_tofu_prompts`), modulo float nondeterminism on GPU.
2. Re-run `detect.py` and check `results/detection/*.png` regenerate and
   `scores.csv` matches prior shape (8 rows).
`AGENTS.md` never substitutes for running these — there's nothing else.

## Env / GPU reference

- NVIDIA A100-SXM4-80GB, CUDA 12.8, torch 2.8.0+cu128.
- `DEVICE` prefers `cuda` (no XPU on this host — the `*_xpu.py` legacy scripts'
  `xpu` branch never triggers here).
- bf16 natively supported — set `--dtype bf16` (default).
- `uv` available (`uv 0.9.0`) — preferred for dependency management (`uv sync`).
