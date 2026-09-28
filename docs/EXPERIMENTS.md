# Experimental design

## Phase model

| Phase | Name | What it does | Reference |
|---|---|---|---|
| **Phase 1** | SD1.5 ablation ladder | Ablation rungs A0–A5 (colour-LoRA + ControlNet + LCM), plus the source-conditioning/DDIM-inversion/fusion follow-ons (P1-10–P1-17) it led to. | [`docs/phases/phase1_ablation.md`](phases/phase1_ablation.md) |
| **Phase 2** | Evaluation | Colour, structural, clinical, cycle-consistency, and multi-centre generalisation checks, run against every Phase-1/3 configuration. | [`docs/phases/phase2_evaluation.md`](phases/phase2_evaluation.md) |
| **Phase 3** | SDXL portability | Transfers the best Phase-1 architecture to SDXL, testing whether the ablation findings are architecture-specific or portable. | [`docs/phases/phase3_sdxl.md`](phases/phase3_sdxl.md) |
| **Probe** | SD3.5 feasibility | Auxiliary feasibility check, explicitly out of the main Phase 1–3 comparison and lowest-priority/descopable per the proposal. | [`docs/phases/sd35_probe.md`](phases/sd35_probe.md) |

## Where to find things

- **Live task board** (per-ticket status, Slurm job IDs as evidence): [`tickets/`](../tickets/) — `tickets/README.md` explains the ID scheme and status legend. This is the authoritative "what's actually done" record.
- **Consolidated numeric results and narrative**: [`docs/results/RESULTS_SUMMARY.md`](results/RESULTS_SUMMARY.md).
- **Method/architecture detail**: [`docs/METHOD.md`](METHOD.md).
- **Datasets and their role per phase**: [`docs/DATASETS.md`](DATASETS.md).
- **Graded research proposal** (source of truth for scope — every ticket cites the proposal section it comes from): `docs/proposal_draft(6).tex`.

## Guardrails that protect experiment validity

These are treated as invariants throughout the project, not just Phase 1:

- Training pairs are coordinate-corresponding, never pixel-exact — affine
  registration is applied only to held-out evaluation data.
- Held-out slides are never used in any training, at any phase.
- One held-out slide (A06) is a genuine, confirmed colour-gap outlier —
  every result reports both a pooled aggregate and an outlier-excluded one,
  never the pooled number alone.
- Diffusion training loss is noisy and plateaus near a floor regardless of
  whether an adapter is useful — the only valid success signal is the
  held-out colour/structure recovery delta, never the training-loss curve.
