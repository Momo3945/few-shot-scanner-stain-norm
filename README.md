# Histopathology Stain Normalisation

Diffusion-based stain normalisation for histopathology, evaluated across scanners
and centres.

## Phase model

| Phase | Name | What it does |
|---|---|---|
| **Phase 1** | SD1.5 ablation ladder | Ablation rungs A0–A5 (colour-LoRA + ControlNet + LCM). Training and inference. |
| **Phase 2** | Evaluation | Colour, structural, clinical, cycle-consistency, and multi-centre generalisation. |
| **Phase 3** | SDXL portability | Transfers the best Phase-1 configuration to SDXL. |
| **Probe** | SD3.5 feasibility | Auxiliary feasibility check. **Not a phase** — kept separate and out of the main results. |

## Why data is stored by dataset, not by phase

Every dataset here is consumed by **more than one phase**, or is deliberately scoped
to one phase but shares a lineage with data used elsewhere. MITOS-ATYPIA-14 alone
feeds all three phases: Phase 1 trains on the A03/H03 pairs, Phase 2 evaluates on the
held-out slides, and Phase 3 re-runs the same protocol under SDXL.

If the tree were organised by phase, those bytes would have to be duplicated or
symlinked. Instead:

- **`data/` is the single canonical copy of each raw dataset**, keyed by dataset name.
  Nothing under `data/` is phase-specific.
- **Each phase folder carries an `INPUTS.md`** that gives the phase-centric view:
  which datasets and paths that phase reads, and what it writes.

So `data/` answers *"where does this dataset live?"* and `phase*/INPUTS.md` answers
*"what does this phase need?"*.

## Dataset → phase map

| Dataset | Path | Phase 1 | Phase 2 | Phase 3 | Role |
|---|---|:--:|:--:|:--:|---|
| **MITOS-ATYPIA-14** | `data/mitos/` | ✅ | ✅ | ✅ | P1: A03/H03 training pairs + A5 pool. P2: held-out A06/A08/A09/A13/A16. P3: same protocol under SDXL. |
| **CAMELYON17** | `data/camelyon17/` | — | ✅ | — | 5-centre generalisation only. |
| **TCGA-BRCA** | `data/tcga_brca/` | ✅ | ✅ | — | P1: 15 slides, A5 warm-start (`phase0/`). P2: 10 disjoint slides, LAB colour reference (`phase2/`). |
| **PanNuke** | `data/pannuke/` | ✅ | — | — | A5 warm-start, 1,000 patches. **Excluded from geometry evaluation.** |
| **Lizard** | `data/lizard/` | — | ✅ | — | DigestPath / GlaS, HoVer-Net structural safety. |

The TCGA-BRCA Phase-1 and Phase-2 slide sets are **disjoint** — no slide used for
warm-start appears in the LAB colour reference.

## Layout

```
data/                      canonical raw data, by dataset (shared across phases)
  mitos/                     4 extracted MITOS sets (train/test x aperio/hamamatsu)
  camelyon17/
    raw/                       15 patient_*.zip archives — KEPT ZIPPED, extraction is a cluster job
    patient_0*/                5 already-extracted patients
    phase2_patches/            derived Phase-2 patches
  tcga_brca/                 phase0/ + phase0_patches/ (P1), phase2/ + phase2_patches/ (P2)
  pannuke/                   train/, validate/
  lizard/                    lizard_images1/2, lizard_labels, overlay
raw_archives/              source archives for MITOS (4 zips + A03/H03 tarballs)
src/                       shared code, by function
  data/                      extract_pairs.py, inspect_mitos.py
  eval/                      registration.py, metrics.py, progress.py
  models/                    pipeline / conditioning (later)
  utils/                     seeding, io (later)
derived/                   generated intermediates
pairs/                     MITOS paired training crops + baseline metrics (pre-existing; see below)
phase1_ablation/           configs/ checkpoints/ outputs/ results/
phase2_evaluation/         configs/ outputs/ results/
phase3_sdxl/               configs/ checkpoints/ outputs/ results/
probe_sd35/                configs/ outputs/ results/
slurm/                     cluster job scripts
archive/                   stray and one-off scripts, superseded extractions, docs, slides
```

### Notes on specific paths

- **`pairs/`** is a derived artifact and conceptually belongs under `derived/`, but it
  is left at the root because existing manifests and scripts reference it by that path.
  Treat it as `derived/pairs/`.
- **`pairs/baseline_metrics/`** holds the pre-normalisation baseline. It belongs
  conceptually to `phase2_evaluation/results/` and should be referenced from there
  rather than recomputed.
- **`data/camelyon17/raw/*.zip` stay compressed.** Extracting all 15 patients is
  ~100 GB of WSI and is a cluster job, not a local step.
- **`src/eval/progress.py`** is a utility and would fit `src/utils/`, but `metrics.py`
  and `registration.py` import it as a flat sibling (`from progress import progress`).
  It stays in `src/eval/` until those imports are refactored.
