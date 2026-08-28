# P2-10 — CAMELYON17 Cross-Hospital Generalisation

**What this tests:** does the colour LoRA — trained on exactly one scanner
pair (Aperio/Hamamatsu) — still help on five completely different hospital
centres it's never seen? Pairwise LAB Wasserstein between all 5 centres
(2,103 patches, 3 patients/centre, 100 patches/patient), before (D_pre) vs.
after (D_post) normalisation at the project's best general-purpose
operating point (colour LoRA + ControlNet + LCM, strength 0.20). Success
criterion: D_post < D_pre.

**Status:** ❌ DONE — FAIL. Only 1 of 10 centre pairs improved, and only by
0.23, a small fraction of typical pair-to-pair variation:

| | D_pre | D_post |
|---|---|---|
| Mean across 10 centre pairs | **56.70** | **57.52** |

A colour LoRA trained on one scanner pair, applied uniformly to 5 unseen
hospital centres, does not pull them closer together — if anything it adds
small, inconsistent per-centre drift.

**Files:** `eval/d_pre_post_comparison.csv` (table above), `eval/
pairwise.csv` (all 10 pair distances), `eval/summary.csv`.

**Images:** `images/camelyon/` (centre-0 raw vs. normalised examples).

**Full narrative:** `docs/results/RESULTS_SUMMARY.md` ("Update
(2026-08-20): P2-10") and `tickets/PHASE2-TICKETS.md` P2-10.
