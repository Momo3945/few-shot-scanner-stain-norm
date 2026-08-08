# Phase 2 — Evaluation

Colour fidelity, structural preservation, clinical utility, cycle consistency, and
multi-centre generalisation.

## Consumes

### MITOS-ATYPIA-14 — held-out evaluation
| Path | What |
|---|---|
| `data/mitos/mitos_atypia_2014_testing_aperio/` | Held-out Aperio frames |
| `data/mitos/mitos_atypia_2014_testing_hamamatsu/` | Held-out Hamamatsu frames |
| `pairs/heldout_frames.csv` | Held-out frame inventory (A06 / A08 / A09 / A13 / A16) |
| `pairs/registered_heldout/` | Held-out frames after affine ECC registration |
| `pairs/baseline_metrics/` | Pre-normalisation baseline (`baseline_per_crop.csv`, `baseline_summary.csv`) — the reference point every rung is scored against |

Held-out slides are disjoint from the Phase-1 training slides (A03 / H03).

### CAMELYON17 — multi-centre generalisation
| Path | What |
|---|---|
| `data/camelyon17/raw/` | 15 `patient_*.zip` archives, **kept zipped** — extraction is a cluster job |
| `data/camelyon17/patient_0*/` | 5 already-extracted patients (000, 020, 040, 060, 080) |
| `data/camelyon17/phase2_patches/` | Derived patches, laid out as `centre_<i>_<patient>/` |
| `data/camelyon17/camelyon_patches.py` | Patch extractor; takes `--input_dir` / `--output_dir` |

Five centres, ~5 patients each. This dataset is Phase 2 only.

### TCGA-BRCA — LAB colour reference
| Path | What |
|---|---|
| `data/tcga_brca/phase2/` | 10 slides, disjoint from the Phase-1 warm-start set |
| `data/tcga_brca/phase2_patches/` | 1,000 derived patches |

### Lizard — structural safety
| Path | What |
|---|---|
| `data/lizard/lizard_images1/`, `lizard_images2/` | DigestPath / GlaS images |
| `data/lizard/lizard_labels/` | Instance / class labels |
| `data/lizard/overlay/` | Rendered overlays |

Used for HoVer-Net structural safety checks.

**PanNuke is not read here** — it is Phase-1 warm-start only and is excluded from
geometry evaluation.

## Code

- `src/eval/registration.py` — affine ECC registration of Hamamatsu into the Aperio grid
- `src/eval/metrics.py` — ΔE / colour fidelity, SSIM, PSNR, MAE
- `src/eval/progress.py` — shared progress bar (imported flat by the two above)

## Produces

| Path | Contents |
|---|---|
| `phase2_evaluation/configs/` | Evaluation protocol configs per axis |
| `phase2_evaluation/outputs/` | Normalised images and registration artifacts under evaluation |
| `phase2_evaluation/results/` | Metric tables per axis: colour, structural, clinical, cycle, multi-centre |
