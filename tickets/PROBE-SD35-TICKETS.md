# Auxiliary Probe — SD 3.5 Feasibility

Source of truth: `docs/proposal.tex`, §"Auxiliary SD~3.5 Feasibility Probe"
(`sec:sd35_probe`).

**This is NOT a phase and NOT part of A0–A5.** SD 3.5 uses an MMDiT-family
architecture (not the SD1.5/SDXL UNet family) and its few-step variant uses
adversarial diffusion distillation, not LCM-LoRA. It does not answer the main
LCM-LoRA research question. Treat as lowest priority — explicitly descopable per
the proposal's own risk table if compute or time is tight.

---

## PR-00 — Fix SD3.5 model weight download
**Status:** TODO — BLOCKING, and worse than P3-02's bug class. Two separate issues.
**Source:** infrastructure prerequisite
**Issue 1 (same `--include` bug as P3-02, fixed):** `slurm/fetch_models.slurm` now
uses repeated `--include` flags. Confirmed via `--dry-run` this is necessary but
NOT sufficient for this repo.
**Issue 2 (new, harder blocker):** `stabilityai/stable-diffusion-3.5-large` is a
**gated repo**. `--dry-run` returns `Error: Access denied. This repository requires
approval.` even with the include fix. The cluster also has **no HF token configured
at all** (`hf auth whoami` → `Not logged in`). Neither is fixable from a script —
requires manual action:
  1. Request/accept access at
     https://huggingface.co/stabilityai/stable-diffusion-3.5-large with your HF
     account (Muhammad, not Claude — this is an account-level approval).
  2. Generate a token at https://huggingface.co/settings/tokens and run
     `hf auth login` (or set `HF_TOKEN`) on the cluster.
**Once both are done:** re-run `sbatch slurm/fetch_models.slurm`. This is the large
T5-XXL-bearing variant — expect a genuinely large download (~20–30 GB).
**Also needed if the probe proceeds:** an SD3.5 Canny ControlNet
(e.g. `stabilityai/stable-diffusion-3.5-large-controlnet-canny`) and
`stabilityai/stable-diffusion-3.5-large-turbo` for the few-step reference (PR-04, PR-05)
— both likely gated too, same manual-approval requirement.

## PR-01 — LoRA training time + peak VRAM measurement
**Status:** TODO — blocked on PR-00
**Source:** `sec:sd35_probe`, first of the four restricted measurements
**Description:** Train an A→H LoRA on the same ≤50 A03/H03 crop pairs used for the
SD1.5 colour LoRA. Record wall-clock training time and peak VRAM. Attempt first on
`bigbatch` RTX3090 24GB nodes; `biggpu` only for mature, debugged code.

## PR-02 — Colour-fidelity test on held-out MITOS patches
**Status:** TODO — blocked on PR-01
**Source:** `sec:sd35_probe`, second measurement
**Description:** Does the A→H LoRA reduce LAB Wasserstein distance on 10–20 held-out
MITOS patches? Reuse `score_outputs.py`'s LAB Wasserstein function against a small
SD3.5 subset — do not build a parallel metrics implementation.

## PR-03 — Structural conditioning test
**Status:** TODO — blocked on PR-00 (ControlNet) + P2-08 (needs HoVer-Net boundary
maps, which P2-08 builds for the main structural-safety eval — reuse, don't duplicate)
**Source:** `sec:sd35_probe`, third measurement
**Description:** Can SD3.5's Canny ControlNet use HoVer-Net-derived nuclear boundary
maps without obvious structural drift? Qualitative/visual assessment per the proposal
(no formal Relative Dice threshold specified for the probe).

## PR-04 — SD3.5 Turbo few-step reference
**Status:** TODO — blocked on PR-00
**Source:** `sec:sd35_probe`, fourth measurement
**Description:** Whether SD3.5 Turbo provides a useful modern few-step reference.
**IMPORTANT (proposal is explicit):** this is NOT treated as an LCM-LoRA equivalent —
different distillation mechanism (adversarial diffusion distillation vs consistency
distillation). Do not conflate the two in reporting.

## PR-05 — Report outcome
**Status:** TODO — blocked on PR-01 through PR-04
**Source:** proposal §"Time Plan" — "A successful probe is reported as auxiliary
evidence... an unsuccessful probe is reported as a compute or compatibility
limitation and does not affect the success criteria of the main project."
**Description:** Write up the probe's outcome (feasible / infeasible) as a short,
clearly-scoped auxiliary section — it must not be framed as a fourth ablation
condition or as answering the main research questions.
