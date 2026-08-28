# Classical baselines — Macenko, Reinhard, Histogram Matching

**What this tests:** three deterministic, non-learned colour-transfer
methods, run through the exact same registration/scoring pipeline as every
diffusion configuration in this project, fit against the same fixed
few-shot reference crop (`A03_00A_c000_hamamatsu.png`) the colour LoRA
trains on. These are what every diffusion rung is measured against
throughout `RESULTS_SUMMARY.md`.

**Status:** ✅ DONE. See `tickets/PHASE2-TICKETS.md` P2-11 for the full
comparison, the metric-construction confound found and checked (windowed
LAB / `wlab_mean`, added retroactively to all three here), and the
per-slide instability finding.

**Mechanism, briefly** (full detail in `tickets/P2-11` and the Pipeline
Anatomy artifact):
- **Macenko** — fits stain vectors (SVD on optical-density values) from
  source to reference, remaps by that transform.
- **Reinhard** — matches LAB channel-wise mean/std between source and
  reference. The most balanced of the three.
- **Histogram Matching** — per-channel CDF remap forcing the entire output
  histogram onto the reference image's histogram, independent of source
  content — the most volatile of the three (see the per-slide-consistency
  addendum in `RESULTS_SUMMARY.md`).

All three **remap existing pixels** rather than regenerating them — the
mechanical reason they preserve structure that diffusion resynthesis
cannot, and the reason P1-16 later converges on the same trick.

**Folders:** `macenko/`, `reinhard/`, `histogram_matching/` — each has
`eval_summary.csv` (per-slide numbers) and `eval_per_crop.csv`.

**Full narrative:** `docs/results/RESULTS_SUMMARY.md` ("Classical baseline
comparison (P2-11)" and the per-slide-consistency addendum) and
`tickets/PHASE2-TICKETS.md` P2-11.
