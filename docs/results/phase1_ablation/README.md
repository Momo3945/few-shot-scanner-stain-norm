# Phase 1 — SD1.5 ablation ladder (A0–A5)

**What this tests:** the core ablation ladder — six increasingly complete
versions of the frozen-SD1.5 pipeline (frozen base → +ControlNet-Canny →
+colour LoRA → both together → +LCM-LoRA → +histopathology warm-start),
each rung isolating one component's contribution. Plus its post-gate
strength-sweep follow-ups (`*_ext_*` folders) and the raw do-nothing
baseline every `recovery_delta_lab` in this project is measured against.

**Status:** ✅ DONE — all six rungs complete, decision gate closed. See
`tickets/PHASE1-TICKETS.md` P1-01–P1-09 for the full build/debug history.

**Folders:**

| Folder | Rung |
|---|---|
| `raw_baseline/` | do-nothing comparison — the reference point |
| `a0/` | frozen SD1.5 base only |
| `a1/` | + ControlNet-Canny |
| `a2h_r4/`, `a2h_r8/` | + colour LoRA, rank 4 / rank 8 |
| `a3/` | ControlNet + colour LoRA (rank 8) |
| `a4/` | + LCM-LoRA (8-step) |
| `a5/` | + histopathology warm-start LoRA |
| `a3_ext_s02/`, `a4_ext_s02/`, `a5_ext_s02/` | P1-09 strength-sweep follow-up @ strength 0.20 |
| `a4_ext_s07/`, `a5_ext_s07/` | P1-09 follow-up @ strength 0.70 |
| `a4_ext_s67/`, `a5_ext_s67/` | P1-09 smoke test, strength 0.60/0.70 |

Each folder: `eval_manifest.csv` (crop-level path bookkeeping),
`eval_per_crop.csv` (per-crop metrics), `eval_summary.csv` (per-slide/
per-strength aggregates — the numbers quoted throughout `RESULTS_SUMMARY.md`).

**Images:** `images/` — the ladder's qualitative comparisons (same crop
across all six rungs, the A06-outlier strength-050 headline result, and the
P1-09 strength sweeps).

**Full narrative:** `docs/results/RESULTS_SUMMARY.md` (sections "Main
comparison", "A4/H4-RQ3", "A5/sec:hist_lora", "Post-gate follow-up") and
`tickets/PHASE1-TICKETS.md`.
