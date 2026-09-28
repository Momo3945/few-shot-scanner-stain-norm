# Datasets

All datasets used are public research releases. No patient-identifiable
information is used at any stage. Raw data is not committed to this repo
(see `.gitignore`) — it lives locally and on the compute cluster's
`/datasets` storage.

| Dataset | Role | Used in |
|---|---|---|
| **[MITOS-ATYPIA-14](https://mitos-atypia-14.grand-challenge.org/)** | Physical Aperio *and* Hamamatsu scans of the *same* tissue — the only dataset here with true paired scanner ground truth. A03/H03 provides the ≤50-pair few-shot training set (Phase 1); A06/A08/A09/A13/A16 are the entirely held-out evaluation slides (Phase 2); the same protocol is re-run under SDXL (Phase 3). | Phase 1, 2, 3 |
| **[CAMELYON17](https://camelyon17.grand-challenge.org/)** | 5 clinical centres, multiple patients each — tests inter-centre colour-variance reduction independent of the MITOS-ATYPIA-14 split. | Phase 2 |
| **[TCGA-BRCA](https://portal.gdc.cancer.gov/projects/TCGA-BRCA)** (via the GDC) | Breast-cancer WSIs. A disjoint 15-slide subset provides A5's histopathology warm-start LoRA pool; a separate disjoint 10-slide subset is the LAB colour reference for a Phase-2 check. The two subsets never overlap. | Phase 1, 2 |
| **[PanNuke](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke)** | 1,000 patches, A5 warm-start only. Excluded from geometry/structural evaluation (it's an appearance source, not a registered pair). | Phase 1 |
| **[Lizard](https://warwick.ac.uk/fac/cross_fac/tia/data/lizard)** | Nucleus instance/class labels (DigestPath/GlaS) — drives the HoVer-Net structural-safety check (does normalisation distort nucleus geometry enough to break downstream detection?). | Phase 2 |

## Why data is stored by dataset, not by phase

Every dataset here is consumed by more than one phase, or is deliberately
scoped to one phase but shares a lineage with data used elsewhere.
MITOS-ATYPIA-14 alone feeds all three phases. Organising by phase would mean
duplicating or symlinking bytes across phase folders; instead `data/` is the
single canonical copy of each raw dataset, and each phase's reference doc in
[`docs/phases/`](phases/) gives the phase-centric view of which paths that
phase reads and writes.

## Notes on specific data paths

- **`pairs/`** is a derived artifact and conceptually belongs under
  `derived/`, but is left at the root because existing manifests and
  scripts reference it by that path — treat it as `derived/pairs/`.
  `pairs/baseline_metrics/` holds the pre-normalisation baseline that every
  Phase-2 rung is scored against.
- **`data/camelyon17/raw/*.zip` stay compressed.** Extracting all 15
  patients is ~100 GB of WSI and is a cluster job, not a local step.

## Guardrails

- Training pairs are coordinate-corresponding, not pixel-exact registered —
  affine registration is reserved for held-out evaluation metrics only.
- Held-out MITOS-ATYPIA-14 slides are never used in any training, at any
  phase — checked as a standing invariant.
- TCGA-BRCA's Phase-1 warm-start slides and Phase-2 colour-reference slides
  are disjoint sets.
- PanNuke is excluded from geometry/structural evaluation.
