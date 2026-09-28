# Phase 3 — SDXL Portability

**Documentation only.** This file holds no code, configs, checkpoints, or results —
see [`CLAUDE.md`](../../CLAUDE.md)'s "Layout" section for the canonical code location.

Transfers the best-performing Phase-1 configuration to SDXL to test whether the
ablation findings are architecture-specific or portable.

## Status: ✅ complete

The core transfer (P3-01 through P3-05), the SDXL vs. SD1.5 comparison
(P3-04), the native-1024/expanded-pair follow-ups (P3-06/P3-07/P3-07b/P3-07c),
and the H2A extension are all done. See
[`tickets/PHASE3-TICKETS.md`](../../tickets/PHASE3-TICKETS.md) for the
per-ticket task board and [`docs/results/RESULTS_SUMMARY.md`](../results/RESULTS_SUMMARY.md)
for the full numbers and narrative.

## Consumes

| Path | What |
|---|---|
| `docs/results/phase1_ablation/` | Rung comparison — identifies the winning config |
| `docs/phases/phase1_ablation.md` | The Phase-1 config being ported |
| `data/mitos/mitos_atypia_2014_training_aperio/` | A03 training frames |
| `data/mitos/mitos_atypia_2014_training_hamamatsu/` | H03 training frames |
| `pairs/train/` + `pairs/train_manifest.csv` | The same paired crops Phase 1 trained on |
| `pairs/heldout_frames.csv`, `pairs/registered_heldout/` | Held-out MITOS slides for scoring |
| `pairs/baseline_metrics/` | Same baseline Phase 2 scores against |

MITOS-ATYPIA-14 only. CAMELYON17, Lizard, PanNuke, and TCGA-BRCA are not read here —
portability is tested on the primary dataset under an unchanged protocol, so that any
difference is attributable to the backbone rather than the data.

## Produces

| Path | Contents |
|---|---|
| `/datasets/mhoosen/stain-norm/lora/<direction>_r<rank>_sdxl/` | SDXL colour-LoRA runs |
| `/datasets/mhoosen/stain-norm/lora/a2h_cond_r8_sdxl*/` | Source-conditioned checkpoints (P3-06/P3-07 family), one dir per resolution/pair-count variant |
| `docs/results/p3_04_sdxl_a4/`, `p3_05_sdxl_a5_warmstart/`, and the other `docs/results/p3_*` dirs | Archived eval outputs for each sub-experiment |
