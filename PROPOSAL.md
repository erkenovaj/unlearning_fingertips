# TOFU unlearning detection — proposal

## What's wrong now

`forget_acc=1.0` is trivial: any two checkpoints differ in activations; an MLP
separates them. It detects "which model am I in", not "which domain was
unlearned". Needs the original model. Useless to an adversary.

`within_acc=0.5` is the real (null) result and is **bugged**: it compares
forget10 prompts vs *other* forget10 prompts (seed 99) inside the unlearned
model. Both are forgotten → indistinguishable by construction. The correct
null is **forget-domain vs retain-domain within the same unlearned model**.

Also: per-prompt mean-pooled pre-logit vectors are the wrong unit. Unlearning
traces live in *domain-aggregated geometric/spectral structure*, not in one
2560-dim vector per prompt.

## Central thesis

Unlearning is a **localized perturbation** (a few layers, a rank-1 steering
vector, a restricted optim); "never trained on D" is a **global property**.
So an unlearned model carries a *domain-localized geometric signature* on D
that genuine ignorance does not. An adversary with only M (no original, no
shadow models) can detect this by comparing D's representations against a
retain domain R **inside M itself** — using spectral/geometric statistics
and tests calibrated on M's own untouched regions.

## Framing: dataset inference, self-referential

- Task: given (M, candidate domain D), decide H0 "M never saw D" vs H1
  "M saw D then unlearned it". Domain-level, not per-example MIA.
- Reference nulls come **from M itself**: (a) a matched retain domain R,
  (b) layers of M that the unlearning optim did not touch. No original M0,
  no shadow models.
- This is the novelty over the two cited MIA-against-unlearning works: they
  need shadow models / the original; we need only M plus a candidate D.

## Hypotheses

- **H1 (rank collapse).** RMU forces forget hidden states toward a random
  vector u → effective rank of forget-domain activation covariance drops at
  trained layers, on D only. Predict localized, sharp.
- **H2 (differential geometry).** For any unlearning method, the geometry
  of M's representations on D is statistically anomalous vs on R within the
  same M. Anomaly type differs per method; no single universal statistic.
- **H3 (weight-space localization).** Targeted methods (RMU/NAO/NPO-restricted)
  leave outlier weight statistics at trained layers; full-FT methods (GA)
  leave diffuse change. Outlier location → method fingerprint.
- **H4 (Pareto).** Evading these tests (e.g. rank-restoration regularizer on
  the unlearning loss) costs forget quality. Detectability vs forget-quality
  traces a frontier; no method is both stealthy and effective.

## Statistics (all computable from M alone, no M0)

- **S1 effective rank.** `erank(Σ_{l,D}) = exp(H(normalized eigvals))` per
  layer l, per domain. Compare D vs R via z-score against retain layers.
- **S2 spectral entropy / participation ratio** of activation SVD per layer.
  RMU → one dominant sing. direction (PR spikes on D).
- **S3 cosine collapse.** `⟨cos⟩_D − ⟨cos⟩_R` of last-token hiddens across
  prompts. Single scalar; cheapest; power against collapsing methods.
- **S4 stable rank of weights.** `s(W) = ‖W‖_F² / ‖W‖_2²` per matrix per
  layer. No forward passes. Outlier layers vs model's own median.
- **S5 within-model weight two-sample test.** Trained-layer weight stats vs
  untouched-layer weight stats (the latter are M's built-in "before" sample).
  Scan contiguous layer subsets; argmax localizes trained region.
- **S6 forget-loss gap / per-token entropy.** `H(p_D) − H(p_R)`. Cheap, but
  overlaps prior calibration work — keep as a baseline, not the headline.

## Experiments

- **E1 — intra-model eff-rank anomaly.** Models: original, retain-only,
  RMU/NPO/GA/SimNPO/NAO (OpenUnlearning TOFU). Per layer × {forget, retain,
  holdout} erank. Expect: unlearned → localized drop on D only; retain-only
  → no anomaly; original → *higher* rank on D (memorized = richer).
  Decision rule: flag D if z(l,D) < −τ over a contiguous layer block.
- **E2 — spectral signature taxonomy.** Same setup, statistic = PR + spectral
  entropy. Output: `[method × statistic × layer]` detectability matrix. The
  practical deliverable: which statistic catches which method.
- **E3 — cosine-collapse p-value.** Null = Δ over many random retain splits
  of M. Per-candidate-domain p-value. Characterize per-method power.
- **E4 — weight stable-rank outlier (zero-GPU).** S4 across all layers;
  flag outliers; argmax-subset localizes trained layers (E5 confirms).
- **E5 — within-model weight two-sample test.** S5 with contiguous-subset
  scan. Compare test statistic distribution: original (null) vs unlearned.
  Localizes unlearning-trained layers *with no data and no original*.
- **E6 — headline dataset-inference ROC.** Combine S1,S3,S4,S5 into one
  score (likelihood ratio or logistic on retain-only vs unlearned labels,
  calibrated on TOFU forget5/10/20 × seeds). Three groups: unlearned-on-D,
  retain-only, original. The separability that matters: **unlearned vs
  retain-only**. Report AUC + per-method breakdown.
- **E7 — stealth-vs-quality Pareto.** Re-run RMU/NPO with a rank-restoration
  regularizer at increasing strength. Plot forget-quality (TOFU forget ROC
  / retain BLEU) vs E6 detectability. Confirm frontier H4.
- **E8 — method fingerprinting.** Outlier-layer location (E4/E5) + anomaly
  shape (E1/E2) → identify method, not just detect. Confusion matrix over
  methods.

## Novelty

Self-referential intra-model dataset inference: no original M0, no shadow
models. Nulls come from M's own retain domain and untouched layers.
Cross-method spectral taxonomy (E2/E8) and the stealth-vs-quality frontier
(E7) are not in the cited prior work.

## Deliverables

1. A battery of single-model statistics that detect unlearning on TOFU
   without M0, with per-method power curves.
2. A `[method × statistic]` detectability table — practical guidance for
   which statistics to run against which suspected method.
3. The Pareto frontier (E7): evidence that stealth implies weak forgetting.
4. (If signals vanish for a method) a positive safety result for that method.

## Cost

- Models are small (Phi-1.5 / Llama-7B), TOFU is tiny.
- E1–E5: ~1 GPU-day total (forward passes on forget+retain, 200 prompts each,
  per method × forget fraction).
- E6: same forward passes, just recombined.
- E7: re-running unlearning with a regularizer, ~1 GPU-day per method.
- E8: analysis only on top of E1–E5 outputs.
- Total: ~3–5 GPU-days on a single rented A100/40GB.

## Anti-goals (explicitly out of scope)

- Per-example membership inference (we do dataset/domain-level only).
- Requiring the original model or shadow models.
- WMDP / cyber / bio — TOFU only.
- New unlearning methods (we only *detect*; E7 reuses existing methods with
  one regularizer).
