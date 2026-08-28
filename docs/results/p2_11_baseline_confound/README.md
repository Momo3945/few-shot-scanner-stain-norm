# P2-11 — Classical Baseline Comparison & the Metric-Construction Confound

**No new data pulled here** — the actual classical-method CSVs live in
`../classical_baselines/`; this ticket is about a check performed *on* that
existing data (a windowed-metric backfill), not a separate run.

**What this tests:** two things. (1) Macenko/Reinhard/Histogram Matching
vs. every diffusion rung, same scoring pipeline. (2) A specific confound
check: histogram matching's mechanism (forcing the whole output histogram
onto one fixed reference) could in principle "cheat" a global colour metric
while being locally wrong — tested via a windowed LAB Wasserstein
(`wlab_mean`, per 64×64 tile) backfilled onto every existing config.

**Status:** ❌ DONE — classical wins by 5–8× on colour-distribution
recovery, and the confound check confirms this is real, not a metric
artefact: the windowed vs. global gap is under ~1.2 LAB units for every
method, nowhere near large enough to explain the margin.

**Also see**: the "per-slide consistency" addendum in `RESULTS_SUMMARY.md`
— classical wins the average by more but is 3–4× more volatile slide-to-
slide than diffusion, a nuance this ticket's own tables don't surface on
their own.

**Where the data actually lives:** `../classical_baselines/*/
eval_summary.csv` (now includes `wlab_mean`/`de2000_mean` columns from the
backfill).

**Full narrative:** `docs/results/RESULTS_SUMMARY.md` ("Classical baseline
comparison (P2-11)" and its "Follow-up: closing the metric gap" section)
and `tickets/PHASE2-TICKETS.md` P2-11.
