# P1-16 — Raw-Source-Detail / Learned Colour-Residual Fusion

**Status:** 🔄 IN PROGRESS (2026-08-27) — F3 parameter search underway on
internal validation, first results promising. Script implemented
(`src/eval/fuse_source_detail.py` + `slurm/fuse_source_detail.slurm`).

Running directly from stock P1-11 (P1-15 not yet passed — see Upstream Base
Output below).

**Prerequisite generated (2026-08-27):** P1-11 had never been run on
internal-validation (training-domain) crops before — only held-out. Ran
both required arms on the same 20 training-pair crops, inversion-fraction
1.0 (P1-11's own established best setting): identity mode (job 47257 →
`eval/p1_10_ddim_inversion/identity_pairs_inv100/`, supplies A_raw + A_V)
and translate mode, source-mode=correct (job 47258 →
`eval/p1_10_ddim_inversion/translate_pairs_inv100_correct/`, supplies
H_pred). Both completed cleanly, no NaN/crash. Required adding `--pairs-dir`
support to `slurm/infer_p1_10_ddim_inversion.slurm` (env var `PAIRS_DIR=1`,
additive, non-breaking — every prior invocation of that launcher targeted
held-out only and is unaffected).

**F3 parameter search (2026-08-27), 20 internal-validation crops, 1 seed,
upstream P1-11 reference: SSIM=0.1644, LAB total=21.46 (job 47258 scored
via `score_p1_10_ablation.slurm`, jobid 47268):**

| sigma | beta | SSIM | LAB total | Job (fuse/score) |
|---|---|---|---|---|
| 4 | 0.75 | 0.185 | 21.74 | 47261/47265 |
| 8 | 0.75 | 0.187 | 21.75 | 47262/47266 |
| 16 | 0.75 | 0.187 | 21.75 | 47263/47267 |
| 8 | 0.50 | **0.1866** | **21.44** | 47269/47273 |
| 8 | 1.00 | 0.1865 | 22.50 | 47270/47274 |

**Reading:** SSIM improves over upstream P1-11 by +0.020 to +0.023
(12–14% relative) at every tested (sigma, beta) combination — a real,
consistent effect, unlike P1-17's null result on the same kind of grid.
SSIM is essentially flat across sigma (saturates by sigma=8) and across
beta (0.1865–0.187 throughout) — the structural win doesn't depend much on
either parameter once sigma is large enough to smooth the residual field
at all. LAB total, by contrast, degrades monotonically as beta increases
(21.44 → 21.75 → 22.50 from beta=0.50 to 1.00) — counterintuitive at
first (more residual should mean more colour recovery), but explained by
F3's own design: the residual is heavily blurred before being added, so
high beta injects more of a *smoothed approximation* of P1-11's colour
shift rather than reproducing it, which is worse than P1-11's own
unsmoothed result. **Best point so far: sigma=8, beta=0.50** — same SSIM
gain as every other config, LAB (21.44) essentially tied with (marginally
better than) upstream P1-11 itself (21.46), i.e. the structural
improvement currently looks close to free.

**Not yet done:** beta below 0.50 untested (SSIM looks saturated and LAB
keeps improving as beta drops, so there may be a still-better point before
it degrades toward the beta=0 raw-source limit); F1/F2 diagnostic variants
not run; no visual artifact check (haloing/seams) yet; only one slide
(A03), one seed; canonical held-out run not started (correctly — no
config has been frozen yet, and this project's own guardrail is to never
escalate to held-out before internal validation shows a real, frozen
effect).

## Motivation

P1-11 strongly improves scanner-colour recovery but remains below deterministic classical methods on SSIM.

The classical methods have a fundamental structural advantage: they recolour the original pixels rather than resynthesising the tissue through a VAE + generative denoising path.

P1-16 tests whether the learned diffusion model can be used primarily as an estimator of the target scanner colour transformation while the final microscopic detail comes directly from the untouched Aperio source.

## Scientific Question

> Can the learned P1-11/P1-15 scanner transformation be projected back onto the original Aperio pixels so that diffusion supplies the A→H colour change while the source supplies the fine tissue structure?

Desired separation:

```text
diffusion -> colour/style transformation
raw source -> microscopic structure
```

rather than:

```text
diffusion -> entire final image resynthesis
```

## Hard Guardrail — No Target Leakage

The fusion algorithm may use only:

```text
raw Aperio source A
VAE reconstruction of A
P1-11/P1-15 predicted H output
```

It must **never use the real held-out Hamamatsu target to construct the fused image**.

Hamamatsu is used only afterward for evaluation.

Do not derive any of the following from held-out target images:

```text
pixel residual
colour field
mask
histogram
local statistic
fusion coefficient
blur scale
registration transform chosen specifically for fusion
```

Otherwise the experiment is invalid.

## No Training Initially

The first P1-16 implementation is deterministic post-processing only.

Do not:

- retrain P1-10;
- retrain the VAE;
- train a fusion network;
- add a new LoRA;
- modify DDIM inversion.

Use existing saved images where possible.

## Upstream Base Output

Define:

```text
H_pred
```

as the best frozen quality path available when P1-16 begins:

```text
P1-15 if P1-15 passes
otherwise P1-11
```

Record the exact upstream model/evaluation tag in every output manifest.

# Variant F1 — Source-Luminance Diagnostic

Convert:

```text
A_raw
H_pred
```

to CIELAB.

Let:

\[
A=(L_A,a_A,b_A)
\]

\[
H=(L_H,a_H,b_H)
\]

Construct:

\[
F_1=(L_A,a_H,b_H)
\]

Interpretation:

```text
luminance / much structural content = raw Aperio
chroma                            = learned translated output
```

This is a diagnostic, not assumed to be the final method.

Purpose:

> How much of the SSIM penalty is due to regenerated luminance/detail rather than the learned chromatic transformation?

# Variant F2 — High-Frequency Luminance Reinjection

Preserve P1-11/P1-15's broad target-scanner luminance while restoring source microscopic detail.

For Gaussian low-pass operator \(G_\sigma\):

\[
L_A^{HF}=L_A-G_\sigma(L_A)
\]

\[
L_H^{LF}=G_\sigma(L_H)
\]

Construct:

\[
L_F=L_H^{LF}+lpha L_A^{HF}
\]

Then:

\[
F_2=(L_F,a_H,b_H)
\]

Interpretation:

```text
low-frequency luminance = learned translated output
high-frequency detail   = untouched source
chroma                   = learned translated output
```

# Variant F3 — VAE-Relative Learned Colour-Residual Projection

This is the primary P1-16 residual-fusion experiment.

Let the stock-VAE reconstruction of the exact source be:

\[
A_V=D(E(A)).
\]

Let the translated prediction be:

\[
H_{	ext{pred}}.
\]

Estimate the learned change relative to the model's own reconstruction of the source:

\[
\Delta=LAB(H_{	ext{pred}})-LAB(A_V)
\]

Remove high-frequency generative variation:

\[
\Delta_{	ext{colour}}=G_\sigma(\Delta)
\]

Apply the smooth learned transformation to the untouched raw source:

\[
LAB(F_3)=LAB(A_{	ext{raw}})+eta\Delta_{	ext{colour}}
\]

Convert back to RGB with correct clipping and gamut handling.

## Why Use `H_pred - A_V` Rather Than `H_pred - A_raw`?

P1-11's output is downstream of the VAE.

Using:

\[
H_{	ext{pred}}-A_V
\]

attempts to cancel VAE reconstruction characteristics shared by the translated and source-reconstruction paths before transferring the learned scanner change onto the untouched source.

This is an empirical hypothesis and must be tested rather than assumed.

## New Script

Create:

```text
src/eval/fuse_source_detail.py
```

Suggested interface:

```text
--source-dir
--prediction-dir
--vae-reconstruction-dir
--method
--sigma
--alpha
--beta
--output-dir
```

The script should consume existing outputs.

Do not rerun diffusion unless required to generate missing validation predictions.

## Parameter Selection

Do **not** tune:

```text
sigma
alpha
beta
```

on:

```text
A06
A08
A09
A13
A16
```

Use P1-10's internal training-domain validation frames.

Generate the upstream P1-11/P1-15 outputs for those internal validation images if needed.

Freeze all fusion parameters before held-out evaluation.

## Initial Parameter Grid

Keep the first grid small.

### F2

```text
alpha = {0.25, 0.50, 0.75, 1.00}
```

### F3

```text
beta = {0.50, 0.75, 1.00}
```

Use only a few sensible `sigma` values appropriate to the crop resolution.

Do not launch a huge blind search.

## Selection Objective

P1-16 must not win merely by copying raw Aperio.

Therefore optimize a **joint colour/structure objective**, not SSIM alone.

Track:

```text
SSIM
windowed LAB
recovery Δlab
```

and define:

\[
R_{	ext{retained}}
=
rac{\Delta LAB_{	ext{fusion}}}
{\Delta LAB_{	ext{upstream}}}
\]

Interpretation:

```text
R = 1.0 -> 100% of upstream colour recovery retained
R = 0.8 -> 80% retained
R ≈ 0   -> fusion has effectively reverted to raw
```

## Proposed Smoke Gate

A fusion candidate should proceed to the full held-out run only if:

1. SSIM clearly improves over the upstream P1-11/P1-15 model;
2. colour recovery stays positive;
3. preferably at least ~80% of the upstream colour-recovery gain is retained on internal validation;
4. no obvious halos, ringing, seams or clipping are visible.

The ~80% criterion is a supplementary practical threshold, not a proposal-defined requirement.

## Mandatory Internal Controls

Compare the same validation samples across:

```text
raw source
VAE-only reconstruction
upstream P1-11/P1-15
F1
best F2
best F3
```

This prevents a trivial regression toward raw from being misreported as a structural win.

## Full Held-Out Evaluation

After parameter selection is frozen, evaluate on the canonical:

```text
496 held-out crops
```

If the upstream P1-11/P1-15 outputs exist for three seeds, fuse each seed separately and preserve the existing three-seed protocol.

Report:

```text
SSIM
PSNR
MAE
windowed LAB
LAB total
dE2000
recovery Δlab
colour-recovery retention ratio
```

## Required Comparison Table

Include:

```text
Raw
Macenko
Reinhard
Histogram Matching
P1-10
P1-11
P1-15 if applicable
F1
best F2
best F3
```

## Classical-Method Comparison

This ticket is explicitly intended to attack the structural advantage of deterministic colour remapping.

Therefore directly compare the best fusion SSIM against:

```text
Macenko             0.628
Histogram Matching  0.651
Reinhard             0.681
```

while checking that the fusion retains substantial P1-11 colour recovery.

A high SSIM that simply reverts to raw is not success.

## Per-Slide Reporting

Always report:

```text
A06
A08
A09
A13
A16
ALL
ALL excluding A06
```

A06 is particularly important because an overly conservative fusion could preserve structure while throwing away the large A06 colour correction achieved by P1-11.

## Qualitative Panel

For representative A06 and A08 examples include:

```text
Raw Aperio
Real Hamamatsu
upstream P1-11/P1-15
F1
best F2
best F3
Macenko
```

Inspect full crop and zoomed nuclear regions.

Look for:

- nuclear-edge preservation;
- chromatin detail;
- stromal fibres;
- halos;
- ringing;
- blur;
- over-smoothing;
- colour seams;
- LAB→RGB gamut artifacts;
- clipping.

## Optional Downstream Structural Check

If a fusion configuration produces a genuinely large SSIM improvement while retaining strong colour recovery, rerun:

```text
Relative Dice / HoVer-Net
```

for **only the final selected fusion**.

Do not rerun Relative Dice for every fusion hyperparameter.

This is valuable because the project's earlier Relative Dice result showed that diffusion structural loss corresponded to a real nucleus-detection penalty, not only a pixel-metric effect.

## Acceptance Criteria

A strong P1-16 result would satisfy:

```text
SSIM substantially > upstream P1-11/P1-15
AND
windowed LAB approximately preserved or improved
AND
positive colour recovery on most/all slides
```

The ideal outcome is to move toward the classical methods' structural range while retaining P1-11's learned scanner normalization.

The ticket does **not** pass if high SSIM is achieved mainly by reverting toward raw Aperio.

## Interpretation

### If fusion works

> The remaining structural gap was not primarily a failure to learn the target-scanner transformation. P1-11 learned a useful colour mapping, but full latent resynthesis unnecessarily replaced high-frequency morphology. Reapplying the learned low-frequency scanner transformation to the original source pixels preserves substantially more structure.

### If fusion fails

> The learned target-scanner transformation is not cleanly separable into a smooth colour residual that can simply be projected back onto the raw source. A more principled pixel-preserving translation architecture would be required.

Either outcome is scientifically useful.
