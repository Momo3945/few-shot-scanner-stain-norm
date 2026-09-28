# P3-08 — SDXL VAE-Only Floor + P1-16-Style Fusion on SDXL

**What this tests:** two levers already proven on SD1.5, neither tried on
SDXL until now. (A) Does SDXL have its own VAE-only reconstruction ceiling,
the way SD1.5's was found to cap SSIM at 0.5393 regardless of backbone or
conditioning quality? (B) Does P1-16's post-hoc raw-source-detail/
colour-residual fusion — SD1.5's single biggest SSIM lever
([`../p1_16_source_fusion/`](../p1_16_source_fusion/)) — transfer to SDXL's
best full run (P3-07c, [`../p3_07c_sdxl_1024_overlap144/`](../p3_07c_sdxl_1024_overlap144/))?

**Status:** ✅ CLOSED, decisive positive result (2026-09-28, jobs 61318 VAE
floor, 61319 vae-only run, 61503 fusion, 61559 scoring).

## Part A — SDXL's own VAE-only floor

Measured for the first time (P3-06's own ticket had left this explicitly
open). SDXL's official VAE, run in float32 (its known NaN-under-fp16
failure mode, established earlier in this project):

| | SSIM | LAB total | PSNR | MAE |
|---|---|---|---|---|
| **SDXL VAE-only** (8 held-out crops, zero denoising) | **0.4924** | 6.69 | 20.23 | 18.74 |
| SD1.5 VAE-only (established earlier, 496 crops) | 0.5393 | — | — | — |

SDXL's own reconstruction ceiling is **lower** than SD1.5's, despite both
being 4-channel VAEs — a real, previously-unknown part of why SDXL's raw
SSIM numbers across P3-06/P3-07/P3-07c (0.392–0.446) have trailed SD1.5's.
The architecture itself has a lower reconstruction ceiling here, not just a
conditioning/data question.

## Part B — Fusion applied to P3-07c (SDXL's current best full run)

`infer_colour_translation_sdxl.py` has no "identity mode" the way P1-11's
script does and never saves the raw Aperio source crop to disk, so
`fuse_source_detail.py` was extended with a `--vae-only-manifest` convention
(alongside its existing SD1.5 `--identity-manifest` path, unchanged) that
re-derives the raw source crop via the same registration/grid-crop routine
`infer_colour_translation_sdxl.py` itself uses. Ran the frozen F3 config
(σ=8, β=0.50) unchanged from P1-16 — no new hyperparameter search.

| | P3-07c (pre-fusion) | **P3-08 (post-fusion)** | SD1.5 P1-16 | Classical (Macenko / Reinhard / HistMatch) |
|---|---|---|---|---|
| ALL SSIM | 0.4457 | **0.7325** | 0.729 | 0.628 / 0.681 / 0.651 |
| ALL_excl_outliers SSIM | 0.4617 | **0.7463** | 0.746 | — |
| ALL recovery Δlab | +1.85 | **+2.47** | +7.81 | — |
| A06 / A08 / A09 / A13 / A16 SSIM | 0.336 / 0.467 / 0.417 / 0.437 / 0.495 | **0.638 / 0.789 / 0.722 / 0.673 / 0.760** | — | — |

Fusion lifted SSIM by **+0.287 absolute (+64% relative)** — clears all three
classical baselines outright (a first for any SDXL configuration), and
essentially matches SD1.5's own best-ever result. Colour recovery
*improved* alongside structure (+1.85 → +2.47), not traded off, and every
per-slide SSIM/Δlab is positive with no exceptions.

**Honest nuance:** SDXL+fusion's SSIM (0.7325) marginally *exceeds* SD1.5's
own P1-16 (0.729), but its colour recovery (+2.47) is substantially
*weaker* than SD1.5's (+7.81). This is not an unambiguous "SDXL now beats
SD1.5" result — it is "fusion transfers cleanly to SDXL and closes the
classical-baseline gap on structure specifically," a real and significant
result in its own right, not the same claim.

**Files:** `eval/vae_floor/{per_crop.csv,summary.json,vae_config_inspection.json}`
(Part A), `eval/vae_only/eval_manifest.csv` (fusion input), `eval/fusion/
{eval_manifest.csv,eval_per_crop.csv,eval_summary.csv}` (table above).

**Full narrative:** `tickets/PHASE3-TICKETS.md` P3-08;
`docs/results/RESULTS_SUMMARY.md`.
