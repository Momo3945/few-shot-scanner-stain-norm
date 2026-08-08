# Phase 3 — SDXL Portability

**Documentation only.** This folder holds no code, configs, checkpoints, or results —
see CLAUDE.md's "Layout" section for the canonical code location.

Transfers the best-performing Phase-1 configuration to SDXL to test whether the
ablation findings are architecture-specific or portable.

## Status (2026-08-08)

**Blocked.** `/datasets/mhoosen/hf_cache/hub/models--stabilityai--stable-diffusion-xl-base-1.0`
contains only config/tokenizer files (~1.6MB), no `.safetensors` weights — the
`--include` filter in `slurm/fetch_models.slurm` needs fixing before this phase can
start. Phase 1 also hasn't produced a winning rung yet to port.

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

## Produces (aspirational — nothing produced yet)

Not yet run, so no cluster path convention is established. Follow the Phase-1 pattern
(`/datasets/mhoosen/stain-norm/<something>-sdxl/<run>/final/...`) when this phase
starts, and record the actual path here.
