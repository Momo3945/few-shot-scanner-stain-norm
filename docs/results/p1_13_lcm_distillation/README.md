# P1-13 / P1-13b — Task-Specific LCM-LoRA Distillation

**What this tests:** distils a brand-new LCM-LoRA directly from the frozen
P1-10 model as its own teacher (consistency distillation), instead of using
the generic pretrained adapter — does matching P1-10's exact trajectory
beat a task-agnostic one? Two scopes: P1-13 (attention-only, rank 64) and
P1-13b (full-UNet, matching the generic adapter's actual module coverage).
Full mechanism, including the `G = w + 1` guidance-semantics correction:
Pipeline Anatomy §08.

**Status:** ❌ CLOSED — negative, under a precommitted hard-stop rule.
Final frozen A06+A08 comparison:

| Arm | SSIM | LAB total |
|---|---|---|
| Generic pretrained LCM-LoRA | **0.2958 ± 0.0019** | 71.43 ± 0.38 |
| P1-13 task-specific, attention-only | 0.2690 | 76.61 |
| P1-13b task-specific, full-UNet | 0.2841 ± 0.0013 | **71.98 ± 0.63** |

Full-UNet scope closed most of the LAB gap (within seed variance) but still
lost on SSIM (non-overlapping error bars) and paired win-rate (30/80 vs.
50/80 crops). Note: the proposal states the pretrained LCM-LoRA "is not
trained in this project" — this ticket is the one place that constraint is
explicitly crossed (see Pipeline Anatomy §13, scope map).

**Files:** `eval/final_comparison/summary.csv` (the table above),
`per_crop.csv`, `paired_win_rate.csv`; `eval/generic_lcm_baseline/
eval_manifest.csv` (the baseline arm reused throughout).

**Full narrative:** `docs/results/RESULTS_SUMMARY.md` and
`tickets/PHASE1-TICKETS.md` P1-13 / P1-13b,
`tickets/P1-13_task_specific_lcm_distillation.md`.
