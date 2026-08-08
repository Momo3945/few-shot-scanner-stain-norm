# Phase 1 — SD 1.5 Ablation Ladder (A0–A5)

Source of truth: `docs/proposal.tex`, Chapter 3 §"Phase 1: Ablation Pipeline"
(`sec:phase1_sec`, `sec:ablation_methodology`, `tab:ablation_ladder`, `sec:training_order`).

Status as of this file's creation (Aug 2026) — verify against the cluster before
trusting; a job leaving the queue is not proof of success.

---

## P1-01 — A0 baseline: frozen SD1.5 img2img, no adapters
**Status:** TODO
**Source:** `tab:ablation_ladder` (row A0)
**Description:** Run the frozen SD1.5 base through img2img on the held-out set with
no colour LoRA, no ControlNet, no LCM. This is the zero-baseline ablation condition —
**distinct from** the raw do-nothing baseline already computed in
`pairs/baseline_metrics/` (which compares raw Aperio vs real Hamamatsu with no model
at all). A0 runs the *untrained diffusion pipeline* through the normalisation process.
**Acceptance criteria:** `eval/a0/eval_summary.csv` exists with per-slide LAB/SSIM/PSNR/MAE.

## P1-02 — A1: Base + ControlNet
**Status:** TODO — blocked on Canny edge extraction from source patches (not yet built)
**Source:** `tab:ablation_ladder` (row A1); `sec:hist_lora` subsection on conditioning signals
**Description:** SD1.5 + ControlNet-Canny (pretrained, `lllyasviel/sd-controlnet-canny`,
already cached), no colour LoRA. Isolates the structural-conditioning contribution alone.
**Depends on:** a Canny edge-map extraction step for source patches.
**Acceptance criteria:** `eval/a1/eval_summary.csv` exists.

## P1-03a — A2 colour LoRA training: A→H, rank 8
**Status:** ✅ DONE — Slurm job 3827 (re-run after 3809 was cancelled on a GPU-less node)
**Source:** `tab:ablation_ladder` (row A2); `sec:hist_lora`; rank experiment per proposal
("r ∈ {4,8}")
**Evidence:** `/datasets/mhoosen/stain-norm/lora/a2h_r8/final/pytorch_lora_weights.safetensors`

## P1-03b — A2 colour LoRA training: A→H, rank 4
**Status:** ✅ DONE — Slurm job 3810
**Evidence:** `/datasets/mhoosen/stain-norm/lora/a2h_r4/final/pytorch_lora_weights.safetensors`

## P1-03c — A2 colour LoRA training: H→A, rank 8 (cycle consistency)
**Status:** ✅ DONE — Slurm job 3811
**Evidence:** `/datasets/mhoosen/stain-norm/lora/h2a_r8/final/pytorch_lora_weights.safetensors`

## P1-03d — A2 colour LoRA training: H→A, rank 4
**Status:** TODO — the one gap in the rank-4/rank-8 × A2H/H2A matrix
**Source:** same as P1-03a–c
**Command:** `sbatch slurm/train_colour_lora.slurm H2A 4` (ask before submitting)
**Acceptance criteria:** `final/pytorch_lora_weights.safetensors` under `lora/h2a_r4/`

## P1-04 — A3: Base + ControlNet + colour LoRA
**Status:** TODO — blocked on P1-02, and on P1-03a/b (pick winning rank first)
**Source:** `tab:ablation_ladder` (row A3)
**Description:** Combine structural conditioning with colour adaptation, no LCM yet.
Tests the combination before acceleration is introduced.

## P1-05 — A4: Full pipeline (+ LCM-LoRA)
**Status:** TODO — blocked on P1-04
**Source:** `tab:ablation_ladder` (row A4); `sec:training_order` step on LCM-LoRA attachment
**Description:** Attach the pretrained `latent-consistency/lcm-lora-sdv1-5` (already
cached) AFTER colour LoRA and ControlNet are frozen. LCM-LoRA itself is never trained —
only its compatibility with the trained adapters is evaluated (4-step LCM vs 50-step
DDIM on held-out MITOS; SSIM/PSNR + artefact inspection per `sec:experiments`).

## P1-06 — A5 histopathology warm-start LoRA training
**Status:** TODO — data pool ready
**Source:** `sec:hist_lora`; rank 32, 3,000 steps, 3,000 patches
(1,000 PanNuke + 1,500 TCGA-BRCA + 500 MITOS)
**Evidence pool ready:** `/datasets/mhoosen/stain-norm/hist_lora_pool/` (3,000 files,
manifests present, held-out leak check passed at build time).
**Constraint (sec:training_order):** must be trained and FROZEN before colour-LoRA
training begins for this leg — do not train jointly.

## P1-07 — A5 full ablation: hist LoRA + colour LoRA + ControlNet + LCM
**Status:** TODO — blocked on P1-06 and P1-04
**Source:** `tab:ablation_ladder` (row A5)
**Description:** Primary comparison is A5 vs A4 (`sec:hist_lora`) — a positive result
quantifies the value of a combined tissue-and-target-domain prior; a negative result
supports that ControlNet + colour LoRA alone is sufficient. This decides whether
P3-05 (A5-on-SDXL) is attempted.

## P1-08 — Denoising-strength / LCM quality sweep infrastructure
**Status:** ✅ DONE — `infer_colour_lora.py` supports `--strengths` sweep;
`score_outputs.py` reports recovery delta per strength
**Source:** H4/RQ3, `tab:hyp_rq`; LCM quality window per `sec:experiments`

---

**Decision gate (proposal §"Time Plan"):** at the end of Phase 1, formally review
A0–A5 results and the SDXL compute-contingency evidence before proceeding to Phase 3
(see PHASE3-TICKETS.md, P3-01).
