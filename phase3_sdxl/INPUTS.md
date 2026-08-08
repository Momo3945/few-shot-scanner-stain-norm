# Phase 3 — SDXL Portability

Transfers the best-performing Phase-1 configuration to SDXL to test whether the
ablation findings are architecture-specific or portable.

## Consumes

| Path | What |
|---|---|
| `phase1_ablation/results/` | Rung comparison — identifies the winning config |
| `phase1_ablation/configs/<best rung>` | The config being ported |
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
| `phase3_sdxl/configs/` | SDXL port of the winning Phase-1 config |
| `phase3_sdxl/checkpoints/` | SDXL LoRA / ControlNet weights |
| `phase3_sdxl/outputs/` | Generated normalised images |
| `phase3_sdxl/results/` | Metrics, directly comparable to the Phase-1 rung it ports |
