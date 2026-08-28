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

**Full narrative:** `docs/results/RESULTS_SUMMARY.md` ("Update
(2026-08-28): P1-16") and `tickets/PHASE1-TICKETS.md` P1-16,
`tickets/P1-16_source_detail_colour_residual_fusion.md`.
