# Phase 1 — SD1.5 Ablation Ladder (A0–A5)

**Documentation only.** This folder holds no code, configs, checkpoints, or results —
see CLAUDE.md's "Layout" section for the canonical code location and the cluster
`/datasets/mhoosen/stain-norm/` paths for real artifacts.

Colour-LoRA + ControlNet + LCM ablation rungs. Training and inference.

## Status (2026-08-08)

3 of 4 planned colour-LoRA legs completed on `bigbatch`: `a2h_r8`, `a2h_r4`, `h2a_r8`
(each 1000/1000 steps, loss ≈0.20–0.21, `final/pytorch_lora_weights.safetensors`
present). `h2a_r4` has not been run. ControlNet and LCM-LoRA rungs (A1–A5 beyond the
base colour LoRA) have not started.

## Consumes

| Path | What | Used by |
|---|---|---|
| `data/mitos/mitos_atypia_2014_training_aperio/` | A03 source frames | rungs A0–A5 |
| `data/mitos/mitos_atypia_2014_training_hamamatsu/` | H03 target frames | rungs A0–A5 |
| `pairs/train/` | Extracted paired A03/H03 crops — the actual training set | all rungs |
| `pairs/train_manifest.csv` | Crop manifest | all rungs |
| `pairs/extract_config.json` | Extraction parameters that produced `pairs/train/` | provenance |
| `data/tcga_brca/phase0/` + `phase0_patches/` | 15 slides, A5 warm-start | rung A5 |
| `data/pannuke/train/`, `data/pannuke/validate/` | 1,000 patches, A5 warm-start | rung A5 |
| `data/mitos/` (A5 pool) | Broader MITOS pool for the A5 rung | rung A5 |

Held-out MITOS slides (A06/A08/A09/A13/A16) are **not** read here — they belong to
Phase 2.

TCGA-BRCA slides used here are disjoint from the 10 slides Phase 2 uses as its LAB
colour reference.

PanNuke is **excluded from geometry evaluation** — it is a colour/appearance
warm-start source only.

## Code

- `src/data/extract_pairs.py` — builds `pairs/train/` from the MITOS training sets
- `src/data/inspect_mitos.py` — dataset inspection, surfaces the A03/H03 resolution mismatch
- `src/train/train_colour_lora.py` + `slurm/train_colour_lora.slurm` — colour-LoRA
  training (the code that produced the runs below)
- `src/models/` — pipeline and conditioning code (to be added, for ControlNet/LCM rungs)

## Produces

| Path | Contents |
|---|---|
| `/datasets/mhoosen/stain-norm/lora/<direction>_r<rank>/` | One dir per run, e.g. `a2h_r8/`, `h2a_r4/` |
| `.../final/pytorch_lora_weights.safetensors` | Trained LoRA weights |
| `.../loss_log.csv`, `.../training_config.json` | Training curve + run config |

The best-performing rung config is the input to **Phase 3** (`phase3_sdxl/`).
