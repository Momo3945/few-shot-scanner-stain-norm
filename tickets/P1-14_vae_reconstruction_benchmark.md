# P1-14 — VAE Reconstruction Benchmark: Stock SD1.5 vs `sd-vae-ft-mse`

**Status:** ⏳ PLANNED — inference-only reconstruction diagnostic.

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

## Interpretation

### Positive

> A measurable portion of the remaining structural loss is attributable specifically to the stock SD1.5 VAE reconstruction path, and a compatible reconstruction-focused decoder may improve P1-11 without retraining the source-conditioned denoiser.

### Negative

> The stock-VAE reconstruction penalty is not meaningfully improved by this pre-specified compatible alternative; further random VAE swapping is unlikely to be a good use of project time.

If negative, move to P1-16 rather than starting a broad VAE search.
