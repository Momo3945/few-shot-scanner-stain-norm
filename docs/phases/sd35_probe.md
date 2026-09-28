# Probe — SD3.5 Feasibility

**Documentation only.** This file holds no code, configs, or results — see
[`CLAUDE.md`](../../CLAUDE.md)'s "Layout" section for the canonical code location.

**Auxiliary. Not a phase.** A feasibility check on whether SD3.5 is a viable backbone
at all. Explicitly lowest-priority and descopable per the proposal's own risk table.
Results are kept out of the main Phase 1–3 comparison and should not be presented
alongside them.

## Status: partially complete, remainder descopable

Model-weight download, LoRA training-time/VRAM measurement, and the
colour-fidelity test on held-out patches (PR-00 through PR-02) are done — see
[`tickets/PROBE-SD35-TICKETS.md`](../../tickets/PROBE-SD35-TICKETS.md). The
structural-conditioning test, few-step reference, and outcome write-up
(PR-03 through PR-05) remain open but are explicitly the lowest-priority,
descopable item in the project's scope — not required to close out the main
research questions.

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
| `/datasets/mhoosen/stain-norm/lora/<direction>_r<rank>_sd35/` | SD3.5 colour-LoRA runs (PR-01) |
| `docs/results/` (SD3.5-tagged sub-dirs) | Archived colour-fidelity eval outputs (PR-02) |

The probe is not expected to produce weights worth keeping long-term beyond
this feasibility check.
