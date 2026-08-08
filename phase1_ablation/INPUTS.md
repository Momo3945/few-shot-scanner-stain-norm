# Phase 1 — SD1.5 Ablation Ladder (A0–A5)

Colour-LoRA + ControlNet + LCM ablation rungs. Training and inference.

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
- `src/models/` — pipeline and conditioning code (to be added)

## Produces

| Path | Contents |
|---|---|
| `phase1_ablation/configs/` | One config per rung A0–A5 |
| `phase1_ablation/checkpoints/` | LoRA / ControlNet weights per rung |
| `phase1_ablation/outputs/` | Generated normalised images per rung |
| `phase1_ablation/results/` | Per-rung training logs and summary tables |

The best-performing rung config is the input to **Phase 3** (`phase3_sdxl/`).
