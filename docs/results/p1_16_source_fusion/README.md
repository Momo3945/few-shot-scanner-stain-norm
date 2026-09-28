# P1-16 — Source-Detail / Colour-Residual Fusion

**What this tests:** stop trusting the diffusion model's regenerated pixels
for structure. Use P1-11's output only to estimate a colour shift, blur
that shift so it carries no fine detail, and paste it onto the untouched
raw Aperio source. Full mechanism (F1/F2/F3 formulas): Pipeline Anatomy §10.

**Status:** ✅ CONFIRMED at full held-out scale — **the first configuration
in this project's entire history (37+ configs) to clear classical
stain-normalisation baselines on SSIM.**

| Scope | SSIM | LAB total | Recovery Δlab |
|---|---|---|---|
| **ALL (5 slides, n=496)** | **0.729** | 26.56 | **+7.81** |
| ALL excl. A06 | 0.746 | 18.11 | — |
| A06 (outlier) | 0.617 | 83.59 | +11.25 |
| A08 | 0.787 | 18.33 | +6.94 |
| A09 | 0.727 | 18.20 | +9.29 |
| A13 | 0.669 | 21.62 | +5.01 |
| A16 | 0.759 | 16.51 | +7.27 |

Four of five slides individually clear the classical baseline range; only
A06 falls narrowly short (0.617 vs. the weakest baseline, 0.628). Colour
recovery is ~89% of P1-11's own (+7.81 vs. +8.74) — a real, modest tradeoff,
confirmed not a revert-to-raw via a direct pixel-diff check. Two real bugs
were caught and fixed on the way here (a scoring baseline-weighting bug, and
a target-leakage bug that produced an invalid SSIM≈0.998 before being
caught and discarded) — see the ticket for the full writeup.

**Files:** `eval/f3_full_heldout/eval_summary.csv` (table above),
`eval_per_crop.csv`, `eval_manifest.csv`.

**Images:** `images/f3_fusion_process/{a06,a08}/` (raw Aperio → VAE
reconstruction → P1-11 prediction → F3 fusion result → real Hamamatsu, same
crop), `images/comparison_p1_16_f3_process_{a06,a08}.png`.

## Reproduced on H2A — structural/colour win, but no downstream clinical-utility gain

The same frozen F3 config (σ=8, β=0.50), rerun on the H2A checkpoint's
own DDIM-inversion output (P1-11's SSIM 0.5477 / Δlab −15.64, this
project's best-performing H2A configuration before fusion):

| | SSIM | Recovery Δlab |
|---|---|---|
| H2A, pre-fusion (P1-11) | 0.5477 | −15.64 |
| **H2A, post-fusion (P1-16)** | **0.7304** | **−5.34** |

Fusion delivers a large, genuine structural/colour improvement — SSIM
+0.183 absolute (~33% relative), colour regression roughly a third the
size, on every single slide including the A06 outlier (−12.84→−2.72). **But
this does not translate into any downstream clinical-utility gain**: the
frame-level atypia-classifier recovery delta is **0.0** (95% CI
[−0.025, +0.025], n=120) — an exact tie with the raw, unnormalised
Hamamatsu baseline, essentially unchanged from the pre-fusion result.
Fusion's benefit is real but structural/colour-only for H2A; the negative
atypia-classifier verdict holds across every H2A configuration tested to
date, including this project's best architecture and its post-hoc fusion
refinement.

**Files (H2A):** `eval/h2a_f3_s8_b0.50_full/{eval_summary.csv,
eval_per_crop.csv,eval_manifest.csv}`.

**Full narrative:** `docs/results/RESULTS_SUMMARY.md` ("Update
(2026-08-28): P1-16", "Update (2026-09-22): P1-16 ... rerun on H2A") and
`tickets/PHASE1-TICKETS.md` P1-16,
`tickets/P1-16_source_detail_colour_residual_fusion.md`,
`tickets/P2-12_atypia_classifier_evaluation_hardening.md` §8.5.
