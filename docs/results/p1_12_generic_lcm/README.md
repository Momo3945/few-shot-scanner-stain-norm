# P1-12 — P1-10 + Generic LCM-LoRA Acceleration

**What this tests:** can the P1-10/P1-11 checkpoint run fast (4–8 steps)
using the same off-the-shelf `latent-consistency/lcm-lora-sdv1-5` A4 already
uses, instead of P1-11's 50-step DDIM-inversion quality path? Self-directed
follow-up, not from the proposal. Full mechanism: Pipeline Anatomy §07.

**Status:** ✅ CLOSED — a real speed/quality tradeoff, not a substitute for
P1-11. Best config found (6-step, guidance 2.0, strength 0.70), A06+A08
diagnostic subset:

| | A06 SSIM | A06 Δlab | A08 SSIM | A08 Δlab |
|---|---|---|---|---|
| P1-11 (50-step DDIM inversion, quality reference) | 0.373 | +21.72 | 0.537 | +7.15 |
| **P1-12 best LCM config** | 0.334 | +9.73 | 0.515 | +2.98 |

Retains under half of P1-11's colour recovery at broadly comparable SSIM.
`guidance=0` and `guidance=1` were confirmed to produce byte-identical
output (diffusers only applies CFG above 1.0) — a quantisation gotcha, not
a bug.

**Files:** `eval/best_config_s6_g2/eval_summary.csv` (this config's
numbers), `eval_per_crop.csv`, `eval_manifest.csv`. The full
strength×steps×guidance grid and extended sweeps stay cluster-only per this
folder's "headline result only" scope — see the ticket for the complete
grid table.

**Full narrative:** `docs/results/RESULTS_SUMMARY.md` and
`tickets/PHASE1-TICKETS.md` P1-12.
