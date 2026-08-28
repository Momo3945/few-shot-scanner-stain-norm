# P1-15 — P1-11 with the Winning Alternate Decoder

**Status:** ✅ CLOSED (2026-08-28) — POSITIVE, strong pass. Decoding P1-11's
existing DDIM-inversion translation with `stabilityai/sd-vae-ft-mse` (P1-14's
winning VAE) instead of the stock decoder, with everything else held
byte-for-byte identical, raises pooled SSIM from 0.4960 to **0.5567**
(+0.0607, exceeding the ticket's own "especially valuable" 0.52+ bar) with
**alt_decoder winning all 496/496 held-out crops** on SSIM, while ALSO
improving pooled colour recovery (windowed LAB 26.01 -> 25.21, ΔE2000 11.16
-> 10.26) rather than trading it off. All 6 acceptance criteria pass. One
nuance: on A06 (the established colour-gap outlier) the SSIM gain holds
(+0.065) but LAB/windowed-LAB get modestly worse (73.82 -> 76.79) — every
other slide improves on both axes simultaneously. No retraining of P1-10
required (decoder-only swap, by construction). Full results below and in
`tickets/PHASE1-TICKETS.md` P1-15; qualitative panel at
`docs/results/p1_15_alt_decoder/qualitative_panel.png`.

Stage 1 (decoder-only swap, stock encoder + P1-11's exact `f=1.00`
correct-source trajectory, only the final decode swapped) work log:

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
- **Source-conditioning sanity check, `--source-mode shuffled` (job 47337,
  scored together with the correct-source smoke test by job 47391):**
  PASSED. Shuffling the source collapses SSIM from ~0.40-0.47 down to
  ~0.09-0.11 for BOTH decoders:

  | Arm | SSIM | LAB total | wLAB | PSNR | MAE |
  |---|---|---|---|---|---|
  | correct + stock | 0.4015 | 62.73 | 63.41 | 12.90 | 48.16 |
  | correct + alt | **0.4658** | 65.29 | 65.74 | 13.27 | 46.81 |
  | shuffled + stock | 0.0925 | 68.53 | 71.05 | 11.16 | 56.96 |
  | shuffled + alt | 0.1117 | 70.09 | 72.48 | 11.40 | 55.35 |

  Confirms the SSIM gain reflects genuine source-conditioned structure
  recovery, not the alt decoder simply smoothing over noise independent of
  the source signal -- alt decoder still edges out stock under shuffling
  too (0.1117 vs 0.0925), consistent with it being a generally-better
  reconstructor independent of conditioning, exactly the expected pattern.
- **Both Stage 1 gates now pass** -- proceeded to the full 496-crop x
  3-seed evaluation across all 5 held-out slides.
- **Full held-out evaluation, `--source-mode correct`, all 5 slides, seeds
  0/1/2 (job 47406, COMPLETED 03:41:01; scored by job 47453, COMPLETED
  02:00:39 -- 496 crops x 3 seeds, 2976 output rows):**

  | Metric | stock_decoder (= P1-11 canonical) | alt_decoder | Δ |
  |---|---|---|---|
  | SSIM | 0.4960 | **0.5567** | +0.0607 |
  | LAB total | 25.10 | **24.53** | -0.56 (better) |
  | windowed LAB | 26.01 | **25.21** | -0.80 (better) |
  | ΔE2000 | 11.16 | **10.26** | -0.90 (better) |
  | PSNR | 16.79 | **17.68** | +0.89 |
  | MAE | 28.97 | **26.26** | -2.70 |

  **Paired win-rate: alt_decoder wins 496/496 crops on SSIM** -- every
  single held-out crop. The stock_decoder row exactly reproduces P1-11's
  canonical baseline (SSIM 0.4960, wLAB 26.01), confirming a clean
  apples-to-apples replication. Unlike the smoke subset, on the full set
  alt_decoder improves EVERY metric including colour (no structure/colour
  trade-off on the pooled aggregate).

  **Per-slide (never let pooled ALL stand alone):**

  | Slide | stock SSIM | alt SSIM | ΔSSIM | stock LAB | alt LAB | stock wLAB | alt wLAB |
  |---|---|---|---|---|---|---|---|
  | A06 (outlier) | 0.3732 | **0.4382** | +0.0650 | **73.12** | 76.32 (worse) | **73.82** | 76.79 (worse) |
  | A08 | 0.5368 | **0.5963** | +0.0595 | 18.12 | **17.46** | 19.02 | **18.02** |
  | A09 | 0.4767 | **0.5408** | +0.0641 | 17.81 | **15.83** | 18.54 | **16.64** |
  | A13 | 0.4979 | **0.5575** | +0.0596 | 21.27 | **20.90** | 22.21 | **21.44** |
  | A16 | 0.5273 | **0.5855** | +0.0582 | 16.67 | **15.44** | 17.78 | **16.25** |
  | ALL | 0.4960 | **0.5567** | +0.0607 | 25.10 | **24.53** | 26.01 | **25.21** |
  | ALL excl. A06 | 0.5142 | **0.5742** | +0.0600 | 17.98 | **16.86** | 18.92 | **17.57** |

  Every slide's SSIM improves by a similar amount (+0.058 to +0.065, no
  outlier in the gain itself). On every non-outlier slide (A08/A09/A13/
  A16), alt_decoder improves BOTH structure and colour simultaneously. On
  A06 specifically, structure improves substantially but colour distance
  (LAB/windowed LAB) gets modestly worse -- the same slide that has always
  behaved differently in this project (LAB Wasserstein ~95 vs ~25 for
  others, per the methodology guardrails). Reported per-slide plus the
  outlier-excluded aggregate, per project convention -- pooled ALL does not
  stand alone.

  **Qualitative panel** (`docs/results/p1_15_alt_decoder/qualitative_panel.png`
  -- A06_00A and A08_00A, x=0/y=0 crops, seed 0: raw Aperio source / real
  Hamamatsu target / P1-11 stock decoder / P1-15 alt decoder, side by
  side): no checkerboarding, ringing, or obvious blur/over-smoothing in
  either decoder's output on visual inspection; nuclear contours and
  chromatin texture look comparable between stock and alt on both slides.
  On A06, the alt decoder's output visibly sits closer to the source
  Aperio's pink cast and further from the real Hamamatsu target's stronger
  magenta/purple cast than the stock decoder's output -- directly
  consistent with A06's measured colour-metric regression above, not a
  contradiction of it. No hallucinated detail observed.

  **Acceptance criteria (all 6 pass):**
  1. improves pooled SSIM over P1-11 -- YES (0.4960 -> 0.5567)
  2. improves A06-excluded SSIM -- YES (0.5142 -> 0.5742)
  3. improves/preserves SSIM on every individual slide -- YES, all 5
  4. retains strong positive colour recovery -- YES on the pooled/
     A06-excluded aggregate (wLAB improves); A06 alone shows a modest
     colour-metric regression, noted above, not disqualifying given the
     ticket's own framing (this criterion is about the aggregate, and A06
     is this project's known, expected outlier)
  5. introduces no obvious artifacts -- YES, per qualitative panel
  6. requires no retraining of P1-10 -- YES, by construction (decoder-only
     swap; encoder/UNet/ControlNet/colour-LoRA all held frozen and
     untouched)

Scripts: `src/eval/infer_colour_source_ddim_inversion_alt_decoder.py`
(standalone, does not edit P1-11's validated script),
`slurm/infer_p1_11_alt_decoder.slurm`, `slurm/score_p1_15_alt_decoder.slurm`.
Full-run data + qualitative panel archived at
`docs/results/p1_15_alt_decoder/`.

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
