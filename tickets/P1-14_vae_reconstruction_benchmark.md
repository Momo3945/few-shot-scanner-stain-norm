# P1-14 — VAE Reconstruction Benchmark: Stock SD1.5 vs `sd-vae-ft-mse`

**Status:** ✅ CLOSED/POSITIVE (2026-08-28) — Stage A and Stage B both
confirm `stabilityai/sd-vae-ft-mse` is a true drop-in reconstruction-fidelity
improvement over the stock SD1.5 VAE. Canonical 496-crop held-out result:
mean ΔSSIM = **+0.0422** (stock 0.5759 → ft_mse 0.6182), positive and
consistent on every one of the 5 held-out slides individually (range
+0.0403 to +0.0462, no domain trade-off, A06 outlier included). Gate to
P1-15 (already open on Stage A alone) is now confirmed by the full
held-out run. See Results below.

## Motivation

The canonical P1-10 VAE-only control found that pure stock-SD1.5 encode→decode reconstruction already introduces a large structural penalty:

```text
SSIM          = 0.5393
LAB total     = 31.24
windowed LAB  = 31.66
PSNR          = 17.72
MAE           = 25.96
```

This occurs with:

```text
no U-Net
no denoising
no noise
no LoRA
no ControlNet
```

P1-11 has already recovered much of the additional denoising-related structural loss. The remaining quality gap is therefore increasingly dominated by the VAE reconstruction path itself.

This ticket asks whether a reconstruction-focused SD1.x-compatible VAE improves histopathology fidelity without changing the diffusion model.

## Scientific Question

> Does `stabilityai/sd-vae-ft-mse` preserve histopathology structure measurably better than the stock SD1.5 VAE currently used by the project under a pure deterministic encode→decode test?

Primary comparison:

```text
stock stable-diffusion-v1-5 VAE
vs
stabilityai/sd-vae-ft-mse
```

## Scope

This ticket performs **only VAE self-reconstruction**.

Do not:

- load P1-10;
- load P1-11;
- load a U-Net;
- load ControlNet;
- load colour LoRA;
- load LCM-LoRA;
- add diffusion noise;
- run DDIM;
- run DDIM inversion;
- perform scanner translation.

The point is to isolate decoder/autoencoder reconstruction quality.

## New Code

Create a standalone benchmark:

```text
src/eval/benchmark_vae_reconstruction.py
slurm/benchmark_vae_reconstruction.slurm
```

Do not modify any existing validated P1-10/P1-11 inference script.

## VAE Configurations

Benchmark:

### A. Current stock VAE

```text
stable-diffusion-v1-5/stable-diffusion-v1-5
subfolder="vae"
```

### B. Alternate VAE

```text
stabilityai/sd-vae-ft-mse
```

Inspect the actual cached configs before running.

Record for each:

```text
latent channels
encoder architecture
decoder architecture
parameter count
dtype
effective scaling factor
Diffusers version
checkpoint path/revision
```

Do not assume compatibility from the model name alone.

## Latent Scaling

The current project uses the stock SD1.5 latent convention with an effective factor of:

```text
0.18215
```

The cached legacy stock VAE config does not explicitly serialize `scaling_factor`, so the benchmark must not interpret an absent field as scale `1.0`.

The script must reproduce the actual project encode/decode convention exactly.

Log the effective scaling factor used for every tested VAE.

## Deterministic Encoding

Use:

```python
latent_dist.mode()
```

not stochastic sampling.

This benchmark must be deterministic.

## Preprocessing

Reuse the exact preprocessing used by the canonical P1-10 VAE-floor scorer:

- same resize/crop behavior;
- same normalization;
- same output range;
- same conversion back to RGB;
- same scorer.

Do not introduce a new image pipeline.

## Stage A — Internal Validation Benchmark

Use P1-10's non-held-out validation frames first.

Benchmark the two VAEs separately on:

```text
Aperio crops
Hamamatsu crops
```

This checks whether a candidate VAE improves one scanner domain while degrading the other.

For each domain report:

```text
SSIM(input, reconstruction)
PSNR
MAE
LAB identity distance
windowed LAB identity distance
```

## Stage B — Canonical Held-Out Confirmation

Only if `sd-vae-ft-mse` clearly improves Stage A should the canonical 496-crop held-out benchmark be run.

Use exactly the same canonical Aperio crops as the existing stock-VAE control.

Prefer rerunning **both** stock and alternate VAEs through the same new benchmark script rather than comparing results produced by slightly different historical code paths.

## Statistics

This experiment is deterministic, so three-seed reporting is unnecessary.

Use paired crop-level analysis.

Report:

```text
mean ΔSSIM
median ΔSSIM
mean ΔPSNR
mean ΔMAE
```

and, if convenient in the existing analysis stack, a paired bootstrap confidence interval for the SSIM difference.

## Proposed Meaningful-Gain Gate

Treat the alternate VAE as meaningfully better if it achieves approximately:

```text
>= +0.01 absolute SSIM
```

on internal validation **and** the paired improvement is consistently positive without materially worsening PSNR/MAE.

This is a supplementary practical threshold, not a proposal-defined pass criterion.

A smaller improvement can still be reported as marginal.

## Per-Slide Reporting

For the held-out confirmation report:

```text
ALL
ALL excluding A06
A06
A08
A09
A13
A16
```

Never rely on pooled results alone.

## Qualitative Check

Create representative A06 and A08 panels:

```text
original
stock-VAE reconstruction
ft-MSE reconstruction
```

Optionally add an absolute-error heatmap.

Inspect specifically:

- nuclear boundaries;
- chromatin texture;
- stromal fibres;
- blur;
- ringing;
- colour drift;
- checkerboard artifacts.

## Acceptance Criteria

P1-14 passes the go/no-go gate for P1-15 only if `sd-vae-ft-mse`:

1. improves reconstruction SSIM on internal validation;
2. does not improve one scanner domain by materially degrading the other;
3. does not introduce obvious artifacts;
4. remains compatible with the SD1.5 latent shape/scaling expected by P1-10/P1-11.

## Results (2026-08-28)

Both VAEs confirmed architecturally identical before scoring (read from the
loaded model configs, not assumed from the repo name): `AutoencoderKL`,
`latent_channels=4`, identical `block_out_channels`/down/up block types,
identical param count (83,653,863), `scaling_factor=0.18215` for both.
Only `sample_size` differs (512 vs 256, a resolution hint, not a hard
shape constraint at this project's 512x512 crop size). True drop-in swap.

### Stage A — internal validation (job 47230, COMPLETED, 50 non-held-out
pairs, both scanner domains, deterministic `.mode()` encode)

| Domain | stock SSIM | ft_mse SSIM | mean ΔSSIM | 95% bootstrap CI | ΔPSNR | ΔMAE |
|---|---|---|---|---|---|---|
| Aperio | 0.5477 | 0.5957 | **+0.0480** | [+0.0470, +0.0490] | +1.38 dB | -3.24 |
| Hamamatsu | 0.5862 | 0.6312 | **+0.0450** | [+0.0443, +0.0457] | +1.30 dB | -2.81 |

Clears the meaningful-gain gate (>=+0.01 absolute SSIM) by nearly 5x in
both scanner domains almost equally — no domain trade-off, tight CIs
entirely above zero, PSNR/MAE improving alongside SSIM.

### Stage B — canonical held-out confirmation (job 47232, COMPLETED,
496-crop canonical Aperio held-out set, same script as Stage A)

Pooled: stock SSIM=0.5759, ft_mse SSIM=0.6182, mean ΔSSIM=+0.0422 (95% CI
[+0.0419, +0.0426]), ΔPSNR=+1.320 dB, ΔMAE=-2.507, n=496.

Per-slide breakdown (paired, computed from the 496-crop `per_crop.csv`;
never let the pooled ALL number stand alone per this project's methodology
guardrails):

| Slide group | n | stock SSIM | ft_mse SSIM | mean ΔSSIM | median ΔSSIM | 95% CI | ΔPSNR | ΔMAE |
|---|---|---|---|---|---|---|---|---|
| ALL | 496 | 0.5759 | 0.6182 | **+0.0422** | +0.0425 | [+0.0419, +0.0426] | +1.320 dB | -2.507 |
| ALL excl. A06 | 432 | 0.5931 | 0.6348 | **+0.0416** | +0.0420 | [+0.0412, +0.0420] | +1.311 dB | -2.502 |
| A06 | 64 | 0.4598 | 0.5061 | **+0.0462** | +0.0470 | [+0.0453, +0.0471] | +1.378 dB | -2.543 |
| A08 | 112 | 0.6062 | 0.6481 | **+0.0419** | +0.0422 | [+0.0412, +0.0426] | +1.308 dB | -2.698 |
| A09 | 96 | 0.5417 | 0.5856 | **+0.0439** | +0.0444 | [+0.0433, +0.0446] | +1.338 dB | -2.395 |
| A13 | 64 | 0.5884 | 0.6297 | **+0.0413** | +0.0412 | [+0.0405, +0.0420] | +1.282 dB | -2.590 |
| A16 | 160 | 0.6167 | 0.6570 | **+0.0403** | +0.0406 | [+0.0396, +0.0410] | +1.309 dB | -2.393 |

Every slide, including the A06 colour-gap outlier, shows a positive,
tightly-CI'd ΔSSIM gain of similar magnitude (+0.040 to +0.046) — the gain
is not being driven by one slide, and A06 (usually the hardest case in
this project) actually shows the *largest* gain, not a degraded one.
Source: `docs/results/p1_14_vae_decoder_swap/eval/stage_b/per_crop.csv`
(local copy of `/datasets/mhoosen/stain-norm/eval/p1_14_stage_b/` on the
cluster).

### Qualitative check (job 47259, A06 + A08 panels, `eval/
p1_14_stage_b_qualitative/qualitative/`)

Inspected `orig / stock / ft_mse` panels and abs-error heatmaps for both
slides directly (pulled and viewed, not just scored). Nuclear boundaries
and chromatin texture are visibly crisper under `ft_mse` than stock —
stock reconstruction shows mild softening/blur on fine chromatin detail
inside nuclei that `ft_mse` recovers more of; stromal fibre texture reads
similarly between the two. No checkerboard artifacts, no ringing, no
colour drift in either decoder. The abs-error heatmaps for both decoders
put error in the same locations (nuclear/stromal boundaries, expected for
any reconstruction), but the `ft_mse` heatmap is visibly lower-amplitude
overall than stock's, consistent with the quantitative MAE gain — no new
error concentrated anywhere `ft_mse` doesn't already share with stock.

### Acceptance criteria

1. Improves reconstruction SSIM on internal validation — **yes** (Stage A, +0.045 to +0.048).
2. Does not improve one scanner domain by materially degrading the other — **yes** (Stage A: both domains gain similarly; Stage B: all 5 slides gain, no domain/slide trade-off).
3. No obvious artifacts — **yes** (qualitative panel check: no checkerboarding, ringing, or colour drift).
4. Remains compatible with the SD1.5 latent shape/scaling used by P1-10/P1-11 — **yes** (identical `latent_channels`, `scaling_factor=0.18215`, param count).

All four criteria pass. **Gate to P1-15 is confirmed.**

## Interpretation

### Positive (confirmed)

> A measurable portion of the remaining structural loss is attributable specifically to the stock SD1.5 VAE reconstruction path. `sd-vae-ft-mse` is a compatible, artifact-free, uniformly-positive drop-in improvement (+0.042 pooled, +0.040 to +0.046 per-slide) that P1-15 can use to decode P1-11's existing translation without retraining the source-conditioned denoiser.

### Negative

> The stock-VAE reconstruction penalty is not meaningfully improved by this pre-specified compatible alternative; further random VAE swapping is unlikely to be a good use of project time.

(Not the outcome reached — kept for record. Actual result: **Positive**, see Results above.)
