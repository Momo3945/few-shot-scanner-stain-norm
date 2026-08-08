# Probe — SD3.5 Feasibility

**Auxiliary. Not a phase.** A feasibility check on whether SD3.5 is a viable backbone
at all. Results are kept out of the main Phase 1–3 comparison and should not be
presented alongside them.

## Consumes

| Path | What |
|---|---|
| `pairs/train/` | A small subset of the MITOS paired crops — enough to smoke-test |
| `data/mitos/` | Source frames, if larger samples are needed |

Deliberately minimal. This probe does not consume CAMELYON17, Lizard, PanNuke, or
TCGA-BRCA, and does not run the Phase-2 evaluation battery.

## Produces

| Path | Contents |
|---|---|
| `probe_sd35/configs/` | Probe configs |
| `probe_sd35/outputs/` | Sample generations |
| `probe_sd35/results/` | Feasibility notes — does it train, does it converge, what does it cost |

No checkpoints directory: the probe is not expected to produce weights worth keeping.
