# Phase 2 — Experimental Protocol and Evaluation

Source of truth: `docs/proposal.tex`, Chapter 3 §"Phase 2: Experimental Protocol and
Evaluation" (`sec:experiments`, `tab:eval`, `tab:baselines`).

---

## P2-01 — Raw / do-nothing baseline
**Status:** ✅ DONE
**Source:** `tab:baselines` (Raw / do-nothing row); `tab:eval` (Colour accuracy — MITOS)
**Evidence:** `pairs/baseline_metrics/{baseline_per_crop.csv,baseline_summary.csv}`.
Per-slide + robust-outlier-excluded aggregate computed (A06 flagged, z≈33.8 vs ≈25.4
clean-slide LAB Wasserstein).

## P2-02 — Held-out registration pipeline (affine ECC)
**Status:** ✅ DONE
**Source:** `sec:experiments` Cycle Consistency subsection — "Affine registration is
applied before computing pixel-level metrics"
**Evidence:** `src/eval/registration.py`, validated against a known injected offset;
`pairs/registered_heldout/` audit overlays present for all 5 held-out slides.

## P2-03 — Colour-LoRA inference + scoring infrastructure
**Status:** ✅ DONE
**Source:** general Phase 2 infrastructure requirement
**Evidence:** `src/eval/infer_colour_lora.py` (img2img + strength sweep, DDIM 50-step
reference per proposal), `src/eval/score_outputs.py` (LAB/SSIM/PSNR/MAE + recovery
delta vs P2-01 baseline, robust per-slide outlier flagging).

## P2-04 — First evaluation run: A2H rank 8
**Status:** 🔄 IN PROGRESS — BLOCKED. Both jobs FAILED, not completed (verified against
`.err` logs, not assumed from submission — see CLAUDE.md rule).
**Source:** `tab:eval` (Colour accuracy — MITOS row); H2/RQ4 in `tab:hyp_rq`
**Acceptance criteria:** `eval/a2h_r8/eval_summary.csv` with per-strength, per-slide
LAB Wasserstein + recovery delta vs raw baseline. **Not met** — `eval/a2h_r8/` exists
but is empty.
**Evidence of failure:**
- `sbatch slurm/infer_colour_lora.slurm a2h_r8` → job 36462, FAILED, 2:22 elapsed.
  `ModuleNotFoundError: No module named 'cv2'` — `infer_colour_lora.py` imports
  `registration.py`, which needs `opencv-python-headless`, but
  `slurm/infer_colour_lora.slurm`'s dependency check only verifies
  `transformers, peft, accelerate, safetensors, diffusers`, never `cv2`.
  `opencv-python-headless` was installed on the cluster env at 2026-08-08 17:20:55 UTC —
  ~13 min *after* this job had already failed — so the fix landed but the job was
  never re-run.
- `sbatch slurm/score_outputs.slurm a2h_r8` → job 36508, FAILED, 1:40 elapsed.
  `FileNotFoundError` on `eval/a2h_r8/eval_manifest.csv` — cascading failure, this file
  was never written because inference (above) crashed first.
- **Separate latent bug, not yet triggered:** `slurm/score_outputs.slurm` line 25 sets
  `BASELINE=${DATA_ROOT}/eval/baseline_metrics/baseline_summary.csv`, but the real
  baseline is at `/datasets/mhoosen/stain-norm/pairs/baseline_metrics/baseline_summary.csv`
  (under `pairs/`, not `eval/`). Fails gracefully (scores without recovery delta) rather
  than crashing, but would silently violate this ticket's acceptance criteria if not
  fixed before the next run.
**Fixes applied (2026-08-08, not yet re-run):**
- `infer_colour_lora.slurm`: dependency check now includes `cv2`; `HELDOUT` corrected
  to `pairs/heldout_frames.csv` (was pointing at a nonexistent `scanner_lora_pairs/`).
- `score_outputs.slurm`: `BASELINE` corrected to `pairs/baseline_metrics/...`.
- **Third blocker found while fixing:** `MITOS_ROOT` pointed at `data/mitos`, which
  never existed on the cluster — the raw held-out MITOS testing frames were never
  uploaded (only derived crops in `pairs/`). Per `tab:split` in the proposal, the
  generalisation test needs "All ×20 frames" for A06/A08/A09/A13/A16, not just crops.
  Resolved: added canonical `mitos_heldout/` folder (see CLAUDE.md's cluster data
  layout section) and uploaded `mitos_atypia_2014_testing_{aperio,hamamatsu}/`
  (~10.4GB) there; `MITOS_ROOT` now points at it.
**Next step:** confirm the upload finished cleanly, then re-submit
`infer_colour_lora.slurm a2h_r8` → `score_outputs.slurm a2h_r8`.

## P2-05 — Evaluation run: A2H rank 4
**Status:** TODO — depends on P1-03b (done) + P2-03 (done); just needs submitting
**Source:** rank experiment, H1/RQ1 in `tab:hyp_rq`
**Purpose:** decide the winning rank empirically (held-out recovery delta), not from
training loss — see CLAUDE.md methodology guardrails.

## P2-06 — Cycle consistency: ground-truth direct comparison
**Status:** TODO — depends on P2-04
**Source:** `sec:experiments` "Ground Truth Direct Comparison"; equation for
A06_00A → Â H06_00A ↔ Real H06_00A
**Description:** Normalised Aperio output compared directly to registered real
Hamamatsu on held-out test slides. Metrics: grayscale SSIM, PSNR, MAE (post-registration).

## P2-07 — Cycle consistency: round-trip reconstruction
**Status:** TODO — depends on both A2H and H2A LoRAs at the chosen rank (P1-03a/c done
for rank 8)
**Source:** `sec:experiments` "Round Trip Reconstruction"
**Description:** A06 → (LoRA A→H) → Ĥ06 → (LoRA H→A) → Â06 ↔ Original A06.
**IMPORTANT (proposal is explicit):** this test uses a **deterministic 50-step DDIM
sampler**, NOT LCM — LCM's stochastic drift would contaminate the structural deviation
measurement. LCM is reserved strictly for the unidirectional deployment pipeline.

## P2-08 — Structural safety: HoVer-Net vs Lizard (Relative Dice)
**Status:** TODO — blocked: needs a HoVer-Net inference wrapper (not yet built) and
Lizard data on the cluster (currently local-only, 1.8 GB)
**Source:** `sec:experiments` "Structural Safety"; success threshold Relative Dice ≥ 0.95
**Description:** Dice(B,G)/Dice(A,G) where G=Lizard ground truth, A=HoVer-Net on
original patch, B=HoVer-Net on normalised patch. PanNuke excluded from this eval
(it appears in A5 training). HoVer-Net must first be validated against Lizard human
masks (Dice(A,G)) before being trusted as a measurement instrument.

## P2-09 — Clinical utility: downstream classifier delta
**Status:** TODO — not started; classifier training infra not yet built
**Source:** `sec:experiments` "Clinical Utility"
**Description:** Standard diagnostic classifier trained on raw Aperio, evaluated on
raw Hamamatsu + all baseline methods (P2-11) + A0–A5 outputs. Primary metric:
recovery delta = performance(Proposed) − performance(Raw Hamamatsu). Primary dataset
metric: atypia score accuracy at ×20 (macro-AUC/F1 secondary); mitosis F-measure at
×40 as cross-magnification generalisation check.

## P2-10 — CAMELYON17 multi-centre generalisation
**Status:** TODO — blocked: CAMELYON17 is local-only (232 GB, only 5/15 patients
extracted), no tiatoolbox patch-extraction script written yet, and this must run as
a cluster CPU job per the cluster guide (not on the login node)
**Source:** `sec:experiments` "Colour Accuracy" — CAMELYON17 subsection; `tab:camelyon`
**Description:** Pairwise LAB Wasserstein between all 5 centres, before (D_pre, 10
distances) and after (D_post, 10 distances) normalisation. Success = D_post < D_pre.
3 patients/centre, 100 random 512×512 patches/patient, pooled per centre.

## P2-11 — Baseline method comparisons
**Status:** TODO — not started
**Source:** `tab:baselines`
**Description:** Macenko, Reinhard, Histogram Matching, StainNet, (pretrained StainGAN
if reproducible), ParamNet — run against the same held-out set and clinical classifier
for a fair comparison table.

---

**Reminder (CLAUDE.md):** always report both the pooled `ALL` aggregate and the
robust-outlier-excluded aggregate. A06 is a confirmed genuine colour-gap outlier —
never let it silently dominate a headline number.
