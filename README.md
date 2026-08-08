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

## Code layout

**See CLAUDE.md's "Layout" section — it is the canonical reference for where code
lives** (`src/data/`, `src/train/`, `src/eval/`, `slurm/`). This README does not
restate it, to avoid the two docs drifting out of sync again.

`phase1_ablation/`, `phase2_evaluation/`, `phase3_sdxl/`, and `probe_sd35/` are
**documentation-only** — each is just an `INPUTS.md` giving the phase-centric view
described above (which datasets/paths that phase reads and writes). They hold no
code, configs, checkpoints, or results; real artifacts live under `src/`, `slurm/`,
and the cluster `/datasets/mhoosen/stain-norm/` paths referenced from each `INPUTS.md`.

### Notes on specific data paths

- **`pairs/`** is a derived artifact and conceptually belongs under `derived/`, but it
  is left at the root because existing manifests and scripts reference it by that path.
  Treat it as `derived/pairs/`.
- **`pairs/baseline_metrics/`** holds the pre-normalisation baseline — the reference
  point Phase 2 scores every rung against (see `phase2_evaluation/INPUTS.md`).
- **`data/camelyon17/raw/*.zip` stay compressed.** Extracting all 15 patients is
  ~100 GB of WSI and is a cluster job, not a local step.
- **`src/eval/progress.py`** is a utility and would fit `src/utils/`, but `metrics.py`,
  `registration.py`, `infer_colour_lora.py`, and `score_outputs.py` import it as a flat
  sibling. It stays in `src/eval/` until those imports are refactored.
