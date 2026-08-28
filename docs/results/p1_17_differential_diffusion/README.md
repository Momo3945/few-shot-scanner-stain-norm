# P1-17 — Differential Diffusion Change-Map Inference

**What this tests:** instead of one global denoising strength for the whole
crop, give every pixel its own "unlock point" via a per-pixel change map
derived from the same Canny edge map already computed for P1-10's
conditioning — protect structurally busy regions, let flat regions change
freely. Full mechanism (incl. the map sign-convention gotcha): Pipeline
Anatomy §11.

**Status:** ❌ CLOSED — negative, internal validation only (no held-out run
was ever justified). All six tested (radius × strength-cap) configs landed
within about one seed-stdev of the plain global-strength baseline:

| radius | c_max | SSIM | LAB total |
|---|---|---|---|
| — (P1-10 baseline, 3 seeds) | — | 0.1616 ± 0.0057 | 22.44 ± 2.61 |
| 2 | 0.50 | 0.164 | 22.12 |
| 2 | 0.70 | 0.162 | 22.30 |
| 4 | 0.50 | 0.166 | 22.04 |
| 4 | 0.70 | 0.165 | 22.10 |
| 8 | 0.50 | 0.167 | 21.91 |
| 8 | 0.70 | 0.167 | 21.91 |

Likely cause: on densely cellular H&E tissue, the Canny-derived
"structure-dense" region reads as nearly the whole crop, so the UNet's
shared receptive field lets free-region generative influence bleed into
nominally protected pixels. Directly motivated P1-16's post-hoc approach,
which sidesteps this by never letting structure pass through the shared
diffusion computation at all.

**Files:** `eval/p1_10_baseline_same4/`, `eval/p1_10_baseline_summary/`
(the baseline), `eval/p1_17_smoke/` (the initial 4-crop smoke test),
`eval/param_search/r{2,4,8}_s2_cmin0.00_cmax{0.50,0.70}/` (the full grid,
table above).

**Full narrative:** `docs/results/RESULTS_SUMMARY.md`,
`tickets/PHASE1-TICKETS.md` P1-17, and
`tickets/P1-17_differential_diffusion_change_map.md`.
