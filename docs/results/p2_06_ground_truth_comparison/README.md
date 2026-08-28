# P2-06 — Ground-Truth Direct Comparison (SSIM / PSNR / MAE)

**No new data pulled here** — this ticket needed no new compute.
`score_aligned_pair()` (`src/eval/metrics.py`) already computes
`ssim`/`psnr`/`mae` alongside `lab_wasserstein` for every scored crop
throughout this project, so these columns already exist in every rung's
`eval_summary.csv` in `../phase1_ablation/` and `../classical_baselines/`.
This folder's only job is documenting that finding, not holding a copy of
the data.

**What this tests:** normalised Aperio output vs. registered real
Hamamatsu, grayscale SSIM / PSNR / MAE, across every method in the project.

**Status:** ❌ DONE — this is where the sharpest disagreement in the whole
project shows up. Raw do-nothing (SSIM 0.733 pooled) and every classical
baseline (0.628–0.681) beat *every single diffusion rung* (0.27–0.46) —
including on A06, the slide diffusion is specifically supposed to help.
Mechanism: SSIM/PSNR/MAE require pixel-exact correspondence; classical
methods never move a pixel, diffusion resynthesises. This divergence from
the LAB-Wasserstein story is what motivated P2-08 (Relative Dice) as a
structural-safety check appropriate for a generative method.

**Where the data actually lives:** `../phase1_ablation/*/eval_summary.csv`
and `../classical_baselines/*/eval_summary.csv` — the `ssim`, `psnr`, `mae`
columns.

**Full narrative:** `docs/results/RESULTS_SUMMARY.md` ("P2-06:
ground-truth direct comparison") and `tickets/PHASE2-TICKETS.md` P2-06.
