# P2-08 — Relative Dice via HoVer-Net

**What this tests:** does normalisation help or hurt a pretrained
nucleus-detection model (HoVer-Net), scored against human-annotated ground
truth nucleus masks (Lizard dataset)? Relative Dice =
Dice(normalised, GT) / Dice(original, GT). HoVer-Net itself was first
validated against ground truth alone to confirm it's a sane instrument
before trusting the ratio. Proposal's H3/RQ2 pass threshold: ≥0.95,
pre-committed before the number came in.

**Status:** ❌ DONE — fails the threshold, at the P1-09 best
general-purpose operating point (colour LoRA + ControlNet + LCM, 1-step
LCM/strength 0.20):

| Check | Value |
|---|---|
| Dice(A, ground truth) — HoVer-Net validation gate | 0.6982 |
| Dice(B, ground truth) — normalised output | 0.6105 |
| **Relative Dice = Dice(B,G)/Dice(A,G)** | **0.8745** (need ≥0.95) |

Full 130-image runs on both sides, no shape-mismatch skips. Agrees with the
pixel-exact metrics (SSIM/PSNR/MAE) and round-trip consistency (P2-07), not
against them.

**Files:** `eval/relative_dice.csv`, `summary.csv` (table above),
`per_image.csv`, `manifest.csv`.

**Full narrative:** `docs/results/RESULTS_SUMMARY.md` ("Update
(2026-08-17): P2-08") and `tickets/PHASE2-TICKETS.md` P2-08.
