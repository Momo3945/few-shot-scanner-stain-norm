# P2-07 — Cycle Consistency (Round-Trip Reconstruction)

**What this tests:** A06 → colour-LoRA(A→H) → colour-LoRA(H→A) → compare to
the *original* A06 crop. No real Hamamatsu target involved at all — isolates
the pipeline's own structural drift from the colour-matching task entirely.
Deterministic 50-step DDIM (not LCM, to avoid sampling-randomness
contamination). Ground-truth direct-comparison test, `sec:experiments`.

**Status:** ❌ DONE — confirms the structural-drift pattern independently.
16-frame run, n=64 crops:

| Metric | Value |
|---|---|
| SSIM | **0.1429** |
| PSNR | 13.11 |
| MAE | 41.54 |

Lower even than the one-way normalisation-vs-real-Hamamatsu SSIM reported
elsewhere for the best diffusion rungs (0.27–0.46) — a third, independent
metric family pointing the same direction as SSIM/PSNR/MAE and Relative
Dice.

**Files:** `eval/per_crop.csv`, `eval/summary.csv`.

**Full narrative:** `docs/results/RESULTS_SUMMARY.md` ("Update
(2026-08-18): P2-07") and `tickets/PHASE2-TICKETS.md` P2-07.
