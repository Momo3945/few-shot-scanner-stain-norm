# P1-10 — Source-Conditioned Scanner Translation

**What this tests:** the colour LoRA never saw the source image during
training, only via inference-time img2img strength. P1-10 fixes that by
training a fresh, trainable ControlNet-style branch (cloned from the frozen
UNet, 6-channel source RGB + Canny conditioning) jointly with the colour
LoRA. Full mechanism: Pipeline Anatomy §05.

**Status:** ✅ DONE — a genuinely mixed result. Full 496-crop held-out:

| | SSIM | Windowed LAB | Recovery Δlab |
|---|---|---|---|
| Classical baselines (range) | 0.628–0.681 | 26–32 | +3.2 to +9.0 |
| SD1.5 prior best (A4/A5@0.20) | 0.454–0.459 | 32.9–33.3 | +1.5 to +1.9 |
| **P1-10** | **0.4485** | **31.60** | **+2.89** |

Beats SD1.5's prior best on colour recovery, roughly ties it on structure;
does not beat classical, capped by a VAE-only floor (SSIM 0.5393, measured
here — zero denoising, pure encode→decode) that later became the ceiling
every SD1.5 result in this project is measured against.

**Files:** `eval/full_heldout/eval_manifest.csv` (crop bookkeeping),
`eval/full_heldout_summary/summary.csv` (headline numbers above),
`per_crop.csv` (per-crop detail).

**Images:** `images/controlnet_input_p1_10_a03_00a_c000.png` (the actual
6-channel ControlNet input, source RGB + Canny), `images/
controlnet_input_canny_comparison_a03_00a_c000.png` (this project's
Otsu-adaptive Canny vs. a textbook fixed-threshold one), `images/
p1_10_vs_baselines_a08/` (P1-10 vs. every classical baseline vs. real
Hamamatsu, same crop).

**Full narrative:** `docs/results/RESULTS_SUMMARY.md` ("Update
(2026-08-20): P1-10") and `tickets/PHASE1-TICKETS.md` P1-10.
