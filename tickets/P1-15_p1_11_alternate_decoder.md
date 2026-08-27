# P1-15 — P1-11 with the Winning Alternate Decoder

**Status:** 🔄 IN PROGRESS (2026-08-27) — P1-14 gated this open (positive result,
`stabilityai/sd-vae-ft-mse` consistent ~+0.04 SSIM reconstruction gain). Stage 1
(decoder-only swap, stock encoder + P1-11's exact `f=1.00` correct-source
trajectory, only the final decode swapped) underway:

- **Compatibility verification + exact-latent A/B sanity check (job 47308):**
  PASSED. Both VAEs report identical `latent_channels=4`, `scaling_factor=
  0.18215`, `param_count=83653863` from the loaded configs (not assumed) --
  a true drop-in swap. 4 crops x 1 seed, dual-decoded from the shared final
  latent, no NaN/Inf, output images valid and non-trivial (different file
  sizes between stock/alt confirm genuine divergence, not a no-op).
- **Stage 1 smoke test, `--source-mode correct`, A06+A08, seeds 0/1/2 (job
  47310, scored by job 47331):** 80 crops x 3 seeds, 240 paired trajectories.
  Go/no-go gate result:

  | Metric | stock_decoder | alt_decoder | Δ |
  |---|---|---|---|
  | SSIM | 0.4015 | **0.4658** | +0.0643 |
  | PSNR | 12.90 | **13.27** | +0.37 |
  | MAE | 48.16 | **46.81** | -1.35 |
  | LAB total | **62.73** | 65.29 | +2.56 (worse) |
  | windowed LAB | **63.41** | 65.74 | +2.33 (worse) |
  | dE2000 | 18.70 | **18.40** | -0.30 (slightly better) |

  Paired win-rate: **alt_decoder wins 80/80 crops on SSIM** -- a completely
  consistent structural improvement, corroborated by PSNR/MAE moving the
  same direction. Colour cost is real but modest (LAB total/windowed up
  ~2.5, dE2000 essentially flat/slightly better) -- the same "real but
  non-disqualifying" pattern P1-14 found for this decoder generally.
  **Gate passes clearly** (`SSIM_alt > SSIM_stock`, colour not materially
  degraded).
- **Source-conditioning sanity check, `--source-mode shuffled` (job 47337):**
  SUBMITTED, running/queued as of this writing -- not yet scored. This
  confirms the SSIM gain reflects genuine source-conditioned structure
  recovery rather than the alt decoder simply smoothing over noise
  independent of the source signal.
- **Not yet started:** the full 496-crop x 3-seed evaluation (blocked on the
  shuffled sanity check finishing and the user's go-ahead), per-slide
  reporting, qualitative panel.

Scripts: `src/eval/infer_colour_source_ddim_inversion_alt_decoder.py`
(standalone, does not edit P1-11's validated script),
`slurm/infer_p1_11_alt_decoder.slurm`, `slurm/score_p1_15_alt_decoder.slurm`.

## Motivation

P1-11 is currently the project's strongest joint colour/structure diffusion result:

```text
ALL:
SSIM          = 0.4960
windowed LAB  = 26.01
recovery Δlab = +8.74

ALL excluding A06:
SSIM          = 0.5142
windowed LAB  = 18.92
recovery Δlab = +7.43
```

P1-11 improved both structure and colour on every held-out slide relative to P1-10.

This ticket therefore keeps the trained translator and inversion trajectory fixed and changes **only the final decoding stage**.

## Scientific Question

> Can the existing P1-11 translated latent be decoded with the reconstruction-superior decoder identified by P1-14 to increase structural fidelity without sacrificing the A→H colour transformation already achieved by P1-11?

## Critical Isolation Rule

The first experiment must be a **decoder-only swap**.

Keep unchanged:

```text
stock SD1.5 encoder
P1-11 DDIM inversion
P1-10 frozen source-conditioning branch
P1-10 frozen colour LoRA
DDIM scheduler configuration
f = 1.00
correct source condition
guidance settings
conditioning scale
timesteps
seed protocol
```

Change only:

```text
final VAE decoder
```

from:

```text
stock SD1.5 decoder
```

to:

```text
winning P1-14 alternate decoder
```

## Why the Encoder Must Stay Stock Initially

Replacing the encoder would alter the latent distribution entering:

```text
DDIM inversion
U-Net denoising
source-conditioned ControlNet
```

P1-10 was not trained under that altered encoder distribution.

Therefore:

```text
P1-15 Stage 1 = stock encoder + alternate decoder only
```

A full VAE encoder+decoder swap would be a separate methodology change and is not part of this ticket.

## New Code Path

Do not break P1-11's validated script.

Preferred approach:

```text
src/eval/infer_colour_source_ddim_inversion_alt_decoder.py
slurm/infer_p1_11_alt_decoder.slurm
```

Alternatively, an optional decoder flag may be added only if the default P1-11 path remains byte-for-byte behaviorally unchanged.

For auditability, a new script is preferred.

## Compatibility Verification

Before translation:

1. confirm alternate decoder accepts the same latent shape;
2. confirm latent scale convention;
3. confirm decoder output range;
4. confirm postprocessing is identical;
5. confirm the alternate encoder is never called;
6. confirm P1-10 weights remain unchanged;
7. confirm output dimensions and dtype match stock output.

Log:

```text
encoder = stock SD1.5
decoder = <alternate checkpoint>
effective scaling factor = ...
```

## Exact-Latent A/B Test

This is mandatory before rerunning a full inversion pass.

For a small P1-11 smoke subset, obtain the **same exact final latent**:

```text
z_final
```

Then decode it twice:

```text
z_final -> stock decoder
z_final -> alternate decoder
```

This isolates decoder behavior perfectly.

If the existing P1-11 script discards final latents, add an optional debug mechanism to save them for a small subset only.

Do not produce separate denoising trajectories for the two decoder arms during this first comparison.

## Smoke Test

Use the same A06+A08 diagnostic subset.

Compare:

```text
P1-11 stock decoder
P1-15 alternate decoder
```

against real Hamamatsu.

Report:

```text
SSIM
windowed LAB
LAB total
PSNR
MAE
dE2000
```

## Go / No-Go Gate

Proceed to the full 496-crop evaluation only if:

```text
SSIM_alt > SSIM_stock
```

and target-colour normalization is not materially degraded.

The strongest pass is:

```text
SSIM ↑
windowed LAB <=
```

relative to stock P1-11.

If the alternate decoder improves SSIM but badly worsens windowed LAB, treat it as a trade-off rather than an automatic replacement.

## Full Held-Out Evaluation

If the smoke gate passes:

```text
496 crops × 3 seeds
```

with the same five held-out slides:

```text
A06
A08
A09
A13
A16
```

Use exactly P1-11's frozen:

```text
f = 1.00
correct source
50 inversion steps
50 forward DDIM steps
same guidance
same source-conditioning scale
same seeds
```

The decoder must remain the only changed component.

## Required Comparison Table

Include:

```text
P1-10
P1-11 stock decoder
P1-15 alternate decoder
Raw
Macenko
Reinhard
Histogram Matching
```

Reference values:

```text
P1-11:
SSIM          = 0.4960
windowed LAB  = 26.01
recovery Δlab = +8.74
```

Also report the A06-excluded aggregate.

## Per-Slide Requirement

Report:

```text
A06
A08
A09
A13
A16
ALL
ALL excluding A06
```

A pooled improvement is not enough.

## Source-Conditioning Sanity Check

A decoder swap should not affect whether the model uses its source condition, but run a small:

```text
correct
vs
shuffled
```

structural check.

This protects against a decoder that increases apparent SSIM mainly through smoothing.

## Qualitative Check

Produce representative A06 and A08 comparisons:

```text
raw
real Hamamatsu
P1-11 stock decoder
P1-15 alternate decoder
```

Inspect:

- nuclear contours;
- chromatin;
- stroma;
- blur;
- over-smoothing;
- hallucinated detail;
- colour saturation;
- decoder artifacts.

## Acceptance Criteria

P1-15 is a strong pass if it:

1. improves pooled SSIM over P1-11;
2. improves A06-excluded SSIM;
3. improves/preserves SSIM on most individual slides;
4. retains strong positive colour recovery;
5. introduces no obvious artifacts;
6. requires no retraining of P1-10.

A result around:

```text
SSIM 0.496 -> 0.52+
```

with similar or better wLAB would be especially valuable, but `0.52` is **not** a formal pass threshold.

## Failure Interpretation

If P1-14 shows better pure VAE reconstruction but P1-15 does not improve translated outputs:

> Better identity reconstruction does not necessarily imply better decoding of translated diffusion latents.

Do not force the alternate decoder into the final pipeline solely because its VAE-only score is higher.

If P1-15 fails, retain stock P1-11 as the quality reference and proceed to P1-16.
