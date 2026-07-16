# Mathematical Background

## 1. Effective rank & rank collapse

**Setup.** For each layer *l*, you collect *n* hidden vectors
`h₁, ..., hₙ ∈ ℝᵈ` (one per prompt). Stack them into matrix `H ∈ ℝⁿˣᵈ`
(rows = prompts, columns = dimensions). Center: subtract the mean row so
`H` has zero-mean columns.

**Eigenvalues** (needed here; full definition in §3). For the *covariance
matrix* `Σ = (1/n) · HᵀH ∈ ℝᵈˣᵈ`, eigenvalues `λ₁ ≥ λ₂ ≥ ... ≥ λₐ ≥ 0`
tell you how much variance the activations have in each principal direction.
Large `λᵢ` = activations spread a lot along that direction. Small `λᵢ` =
activations barely move there.

If only `λ₁` is large and the rest ≈ 0: all prompts produce nearly the same
hidden vector (just scaled). The activations live in a 1-dimensional
subspace. That's **rank collapse**.

**Effective rank** quantifies this as a continuous number between 1 and *d*:

1. Normalize: `pᵢ = λᵢ / Σⱼλⱼ`. Now `p` is a probability distribution
   (sums to 1).
2. Shannon entropy: `H(p) = -Σᵢ pᵢ ln(pᵢ)`.
3. `erank = exp(H(p))`.

**Why exp(H)?** Entropy measures how uniform the distribution is:
- All variance in one direction: `p₁=1`, rest 0 → `H=0` → `erank=1`.
- Variance equally spread across *d* directions: `pᵢ=1/d` → `H=ln(d)` →
  `erank=d`.
- Something in between: `erank` is between 1 and *d*, measuring the
  "effective number of active dimensions."

**Example.** If `λ = [100, 0.1, 0.1, 0.1]` then `p ≈ [0.997, 0.001, ...]`,
`H ≈ 0.02`, `erank ≈ 1.02`. Nearly collapsed. If `λ = [25, 25, 25, 25]`
then `p = [0.25, ...]`, `H = ln(4)`, `erank = 4`. Full rank.

**Rank collapse in unlearning.** RMU forces forget hidden states toward a
fixed random vector `u`. After training, all forget prompts produce hidden
states clustered around `u` — they span ≈1 dimension → `erank` drops
sharply at trained layers, on forget domain only.

**Computational note.** We compute the *n×n Gram matrix* `G = (1/n)·HHᵀ`
instead of the *d×d* covariance `Σ = (1/n)·HᵀH`. They share the same
nonzero eigenvalues (a standard linear algebra fact), but when *n* = 200
and *d* = 2048, `G` is 200×200 — much cheaper to eigendecompose.

---

## 2. Mean cosine

For two vectors `hᵢ, hⱼ`, cosine similarity is:

```
cos(hᵢ, hⱼ) = (hᵢ · hⱼ) / (‖hᵢ‖ · ‖hⱼ‖)
```

This is the **cosine of the angle** between them. Range: [-1, 1].
- cos = 1: same direction
- cos = 0: orthogonal (perpendicular)
- cos = -1: opposite direction

**Mean cosine** = average over all pairs `(i, j)` where `i < j` (upper
triangle of the similarity matrix, excluding the diagonal):

```
mean_cos = (2 / n(n-1)) · Σᵢ<ⱼ cos(hᵢ, hⱼ)
```

**What it captures.** If all hidden vectors point in the same direction
(collapsed), every pair has cos ≈ 1, so mean_cos ≈ 1. If prompts produce
diverse, spread-out representations, pairwise cosines are low.

**Relationship to effective rank.** They measure the same phenomenon
(collapse) from different angles. Effective rank looks at the eigenvalue
spectrum; mean cosine looks at pairwise angles. They usually agree but can
diverge: if vectors form two tight clusters in different directions, mean
cosine is moderate but effective rank is ~2. Mean cosine is cheaper (no
eigendecomposition) but less informative about *how many* directions are
active.

**Why it's a good statistic for unlearning detection.** RMU explicitly
pushes all forget representations toward one vector `u`. This maximizes
mean cosine on the forget domain. The retain domain is unaffected. So
`Δcos = mean_cos(forget) − mean_cos(retain)` spikes at trained layers — a
direct, cheap signal.

---

## 3. Participation ratio & eigenvalues

**Eigenvalues — full definition.** For a square matrix `A ∈ ℝⁿˣⁿ`, a
scalar `λ` is an eigenvalue if there exists a nonzero vector `v` (the
eigenvector) such that:

```
A · v = λ · v
```

Meaning: `A` only *stretches* `v` by factor `λ`, without rotating it. The
eigenvectors are the "natural axes" of the transformation; the eigenvalues
are the "stretch factors" along those axes.

For a **symmetric** matrix (like a covariance matrix `Σ`), all eigenvalues
are real and non-negative, and eigenvectors are orthogonal. They're found
by solving `det(A − λI) = 0`.

**Geometric intuition.** Think of the covariance matrix as defining an
ellipsoid (a stretched sphere). The eigenvectors are the axes of the
ellipsoid; the eigenvalues are the lengths of those axes. A long thin
cigar shape has one large eigenvalue and many small ones. A round sphere
has all eigenvalues equal.

**Participation ratio (PR):**

```
PR = 1 / Σᵢ pᵢ²     where pᵢ = λᵢ / Σⱼ λⱼ
```

This measures how many directions "participate" significantly.

| Eigenvalue pattern | `p` | `Σpᵢ²` | PR | Meaning |
|---|---|---|---|---|
| `[1, 0, 0, 0]` | `[1, 0, 0, 0]` | 1 | 1 | One direction dominates |
| `[0.5, 0.5, 0, 0]` | `[0.5, 0.5, 0, 0]` | 0.5 | 2 | Two directions participate |
| `[0.25, 0.25, 0.25, 0.25]` | all 0.25 | 4·(0.0625) = 0.25 | 4 | All directions equal |

**PR vs effective rank.** Both measure dimensionality. PR is simpler (no
log/exp) but less sensitive to the distribution *shape* — it's dominated
by the largest components. Effective rank (entropy-based) captures more
nuance. We compute both as a robustness check: if they tell the same
story, the result is solid.

---

## 4. Unlearning methods

All methods start from the **original model** `π₀` (fine-tuned on full
TOFU). They produce `π` (unlearned) by optimizing a loss on forget data
`D_f` and retain data `D_r`.

### RMU — Representation Misdirection for Unlearning

**Idea:** Don't touch the output probabilities directly. Instead, corrupt
the intermediate representations so the model can no longer process
forget-domain information.

**Loss:**

```
L = ‖h_l(x_f) − c·u‖² + α · ‖h_l(x_r) − h_l^0(x_r)‖²
```

- `h_l(x)` = hidden state at layer *l* for input *x*
- `u` = fixed random vector (same dimension as hidden state)
- `c` = steering coefficient (controls how far to push)
- `h_l^0` = hidden state from the **frozen original model**
- Only 3 layers around *l* are trained; everything else is frozen.

**Intuition:** For forget data, push hidden states toward random noise
`u`. For retain data, keep hidden states close to the original model. The
model "forgets" forget-data because its representations become garbage,
but retain-data processing is preserved.

**Why it causes rank collapse:** All forget hidden states get pushed
toward the *same* vector `u`. So they cluster → rank drops to ≈1 at the
trained layer.

### NPO — Negative Preference Optimization

**Idea:** Frame unlearning as a preference problem: the model should
*prefer to not generate* forget-domain text. Based on DPO (Direct
Preference Optimization).

**Loss:**

```
L_forget = (2/β) · E_{x∈D_f}[ softplus(β · (log π(x) − log π₀(x))) ]
```

where `softplus(z) = ln(1 + eᶻ)`.

**How it works:**
- `log π(x)` = log-probability the current model assigns to forget text
  *x*
- `log π₀(x)` = log-probability the **original** model assigns
- `log_ratio = log π(x) − log π₀(x)`: is the model still assigning high
  probability?
  - If `log_ratio > 0` (model still "knows" it): `softplus` is large →
    penalized
  - If `log_ratio < 0` (model has reduced probability): `softplus(β·neg)
    ≈ 0` → no penalty
- `β` controls sharpness: large β → harder penalty for any residual
  knowledge.
- Plus `α · L_retain` (standard cross-entropy on retain data).

**Intuition:** "If you still assign higher probability to forget data
than the original model did, that's bad. If you've reduced it, you're
good." The softplus makes this a smooth gradient signal, unlike gradient
ascent which just says "maximize the loss."

### GradDiff — Gradient Difference

**Idea:** Simultaneously *increase* loss on forget data (forget it) and
*decrease* loss on retain data (keep it).

**Loss:**

```
L = L_retain − α · L_forget
```

where:
- `L_retain = −E_{x∈D_r}[log π(x)]` (standard negative log-likelihood —
  minimize this → stay good at retain)
- `L_forget = −E_{x∈D_f}[log π(x)]` (NLL on forget — the **minus sign**
  means we *maximize* the NLL → make the model bad at forget)

**Intuition:** Minimizing `L` means minimizing `L_retain` (good at
retain) and maximizing `L_forget` (bad at forget). The `α` controls the
trade-off. This is essentially gradient ascent on forget data + gradient
descent on retain data, done jointly.

**Vs. plain gradient ascent (GA):** GA only does `−L_forget` and hopes
retain doesn't degrade. GradDiff explicitly preserves retain. But both
modify many layers diffusely (full fine-tuning), unlike RMU which is
layer-targeted.

### AltPO — Alternative Preference Optimization

**Idea:** A variant of NPO with a flipped sign in the softplus, changing
which regime is penalized.

**Loss:**

```
L_forget = (2/β) · E_{x∈D_f}[ softplus(−β · log_ratio) ]
```

where `log_ratio = log π(x) − log π₀(x)`.

**Difference from NPO:**
- NPO penalizes when `log_ratio > 0` (model still knows the data)
- AltPO penalizes when `log_ratio < 0` (model has already reduced
  probability)

This seems counterintuitive — why penalize successful forgetting? AltPO
is designed to prevent *over-unlearning*: it acts as a regularizer that
stops the model from drifting too far from the original on forget data,
while still encouraging some reduction. The net effect is a gentler, more
stable unlearning.

### UNDIAL — Unlearning via Dialing

**Idea:** "Dial down" the model's response to forget data by
interpolating between the trained response and a neutral/uninformative
response.

**Loss:**

```
L_forget = E_{x∈D_f}[ −log π(y_dial | x) ]
```

where `y_dial` is a "dialed" target — a mixture between the true answer
and an "I don't know" response, controlled by parameter `β`:
- Large `β` → `y_dial` is closer to "I don't know"
- Small `β` → `y_dial` is closer to the original answer

Plus retain cross-entropy with weight `α`.

**Intuition:** Rather than violently destroying forget knowledge (like
GA or RMU), UNDIAL gradually redirects the model toward non-informative
outputs. The `β` parameter "dials" the degree of forgetting.

### IdkNLL — "I Don't Know" Negative Log-Likelihood

**Idea:** Replace the forget-data targets with "I don't know" responses
and train the model to produce those.

**Loss:**

```
L = α · L_forget + L_retain
```

where:
- `L_forget = −E_{(q,a)∈D_f}[ log π("I don't know" | q) ]` — train model
  to output "I don't know" when asked forget questions
- `L_retain = −E_{(q,a)∈D_r}[ log π(a | q) ]` — standard training on
  retain

**Intuition:** The simplest conceptual approach: just teach the model to
say "I don't know" for forget questions. The knowledge may still be in
the weights (unlike RMU which corrupts representations), but the model
is trained to not express it. This might make it harder to detect via
output-level signals, but the weight-level traces of the retraining are
still there.

### Summary table

| Method | Mechanism | Layer scope | Expected spectral trace |
|---|---|---|---|
| RMU | Push forget hiddens → random vector | 3 layers | Sharp rank collapse, cosine spike at trained layers |
| NPO | Reduce log-prob vs original (DPO-style) | Full FT | Moderate rank reduction, diffuse |
| GradDiff | Maximize forget NLL, minimize retain NLL | Full FT | Moderate, diffuse |
| AltPO | NPO variant with over-unlearning regularization | Full FT | Gentler, possibly stealthier |
| UNDIAL | Interpolate toward "I don't know" | Full FT | Moderate, diffuse |
| IdkNLL | Retrain to output "I don't know" | Full FT | Moderate, diffuse |

**Key distinction for detection:** RMU is *layer-targeted* (trains 3
layers) → sharp, localized spectral trace. The others are *full
fine-tuning* → diffuse, harder to localize. This is exactly what E4/E5
(weight analysis) tests: can we detect the difference in spatial pattern
of the perturbation?

---

## 5. Stable rank

For a weight matrix `W ∈ ℝᵐˣⁿ`, stable rank is:

```
srank(W) = ‖W‖²_F / ‖W‖²_2
```

**Two norms defined:**

**Frobenius norm** — the "total energy" of the matrix:
```
‖W‖_F = sqrt( Σᵢ Σⱼ Wᵢⱼ² ) = sqrt( Σₖ σₖ² )
```
where `σ₁ ≥ σ₂ ≥ ... ≥ σᵣ` are the **singular values** of `W` (singular
values are the square roots of eigenvalues of `WᵀW`; for symmetric
positive-semidefinite matrices they equal the eigenvalues).

**Spectral norm** (operator 2-norm) — the "peak energy":
```
‖W‖_2 = max( σₖ ) = σ₁  (the largest singular value)
```

Geometrically: `W` maps the unit sphere to an ellipsoid. `σ₁` is the
longest axis; `‖W‖_F` is the total "size" of the ellipsoid.

**So:**
```
srank(W) = Σₖ σₖ² / σ₁²
```

| Singular values | `Σσ²` | `σ₁²` | srank | Meaning |
|---|---|---|---|---|
| `[10, 0, 0, 0]` | 100 | 100 | 1 | Rank-1: one direction dominates |
| `[10, 10, 0, 0]` | 200 | 100 | 2 | Two equal directions |
| `[10, 10, 10, 10]` | 400 | 100 | 4 | All directions equal |

**Why stable rank instead of effective rank for weights?**
- Effective rank requires eigendecomposition of `WᵀW` (expensive for
  large matrices).
- Stable rank only needs `‖W‖_F` (sum of squares — trivial) and `‖W‖_2`
  (largest singular value — one Lanczos iteration, fast).
- Both measure the same thing: how concentrated the matrix's energy is
  in its top direction.
- "Stable" because it's less sensitive to tiny singular values than the
  numerical rank (count of nonzero σᵢ).

**What it detects in unlearning.** Unlearning fine-tuning modifies
weight matrices. The modified layers have different singular value
distributions → different stable rank. Layers that *weren't* trained
keep their original stable rank. So scanning stable rank across layers
reveals which layers were touched — without any forward passes or data.

---

## 6. Z-score

For a set of values `x₁, ..., xₙ` with mean `μ` and standard deviation
`σ`:

```
zᵢ = (xᵢ − μ) / σ
```

**What it measures:** How many standard deviations `xᵢ` is from the
mean. It's a **standardized distance**.

**Properties:**
- `z = 0`: the value is exactly at the mean
- `z = 1`: one standard deviation above the mean
- `z = -2`: two standard deviations below the mean
- For normally distributed data: `|z| > 2` occurs ~5% of the time,
  `|z| > 3` occurs ~0.3%

**How we use it.** For each layer *l*, compute `x_l` = mean stable rank
of that layer's weight matrices. Then:

```
z_l = (x_l − μ) / σ
```

where `μ`, `σ` are computed across *all layers* of the same model. This
makes each layer its own "sample" from the model's layer distribution.

**The self-referential trick.** We don't need an external baseline — the
model's own layers serve as the null distribution. In a naturally-trained
model, all layers have similar stable rank statistics → no outliers. In
an unlearned model, the few trained layers deviate → they show up as
`|z| > 2` outliers against the model's own untouched layers.

**Example.** Say a model has 16 layers with mean stable ranks:

```
[3.1, 3.0, 3.2, 3.1, 5.8, 5.9, 5.7, 3.0, 3.1, 3.2, 3.0, 3.1, 3.2, 3.1, 3.0, 3.2]
```

Mean `μ ≈ 3.4`, `σ ≈ 1.1`. Layers 4, 5, 6 get `z ≈ [2.2, 2.3, 2.1]` —
flagged as anomalous. Those are the RMU-trained layers. The rest have
`|z| < 0.2` — normal. No original model needed; the anomaly is visible
from the model alone.
