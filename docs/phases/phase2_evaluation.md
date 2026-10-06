# Phase 2 — Evaluation

**Documentation only.** This file holds no code, configs, or results — see
[`CLAUDE.md`](../../CLAUDE.md)'s "Layout" section for the canonical code location and
the cluster `/datasets/mhoosen/stain-norm/` paths for real artifacts.

Colour fidelity, structural preservation, clinical utility, cycle consistency, and
multi-centre generalisation.

## Status: ✅ complete

All evaluation tickets (P2-01 through P2-13) are done, including the negative
results (the HoVer-Net/Lizard structural-safety gate and the CAMELYON17
multi-centre check both failed their acceptance bars — reported as genuine
findings, not hidden). See
[`tickets/PHASE2-TICKETS.md`](../../tickets/PHASE2-TICKETS.md) for the
per-ticket task board and [`docs/results/RESULTS_SUMMARY.md`](../results/RESULTS_SUMMARY.md)
for the full numbers and narrative.

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
- `src/eval/progress.py` — shared progress bar (imported flat by the others)
- `src/eval/infer_colour_lora.py` + `slurm/infer_colour_lora.slurm` — runs a trained
  LoRA over the held-out set
- `src/eval/score_outputs.py` + `slurm/score_outputs.slurm` — scores inference output
  against baseline (imports `metrics.py` as a flat sibling)
- **Learned baselines (P2-11b):** `src/baselines/models.py` (StainNet / ParamNet /
  CycleGAN definitions), `src/train/train_learned_baseline.py` +
  `slurm/train_learned_baseline.slurm` (trains on the same 50 LoRA pairs),
  `src/eval/infer_learned_baseline.py` + `slurm/infer_learned_baseline.slurm` (held-out
  inference, same manifest schema, so `score_outputs.py` is unchanged)
- **Lizard structure-safety for the baselines (P2-11b):**
  `src/eval/normalize_lizard_learned.py` / `normalize_lizard_classical.py` (+ slurm),
  then `slurm/infer_hovernet.slurm` and `slurm/score_lizard.slurm` as for the diffusion
  run; `src/eval/lizard_change_magnitude.py` + slurm measures how much each normaliser
  changes the images

## Produces

| Path | Contents |
|---|---|
| `/datasets/mhoosen/stain-norm/eval/<run>/eval_manifest.csv` | Inference run manifest |
| `/datasets/mhoosen/stain-norm/eval/<run>/eval_per_crop.csv` | Per-crop metrics |
| `/datasets/mhoosen/stain-norm/eval/<run>/eval_summary.csv` | Aggregate scores (per-slide + outlier-excluded, per CLAUDE.md guardrails) |
| `/datasets/mhoosen/stain-norm/baselines/<direction>_<method>/` | Trained StainNet / ParamNet / StainGAN weights (P2-11b) |
| `/datasets/mhoosen/stain-norm/eval/lizard_normalised_images_<method>/`, `eval/lizard_normalised_<method>/`, `eval/lizard_change_magnitude/` | Lizard normalised images, HoVer-Net predictions + Dice, and change-size tables for every normaliser |
