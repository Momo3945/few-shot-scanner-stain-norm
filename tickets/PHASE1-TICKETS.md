# Phase 1 — SD 1.5 Ablation Ladder (A0–A5)

Source of truth: `docs/proposal.tex`, Chapter 3 §"Phase 1: Ablation Pipeline"
(`sec:phase1_sec`, `sec:ablation_methodology`, `tab:ablation_ladder`, `sec:training_order`).

Status as of this file's creation (Aug 2026) — verify against the cluster before
trusting; a job leaving the queue is not proof of success.

---

## P1-01 — A0 baseline: frozen SD1.5 img2img, no adapters
**Status:** ✅ DONE (2026-08-09) — full held-out run + scoring complete.
Infer job 38544 (COMPLETED 1:24:36, resubmit after first attempt 37406 was
cancelled by user's own account at 45:59 -- confirmed via `sacct -j 37406
--format=State%30 -X -n` showing "CANCELLED by 328600015", matching mhoosen's
own uid). 1488-row `eval/a0/eval_manifest.csv`, all 5 held-out slides present.
Score job 39010 (COMPLETED 24:30, exit 0). `eval/a0/eval_summary.csv` verified:
496 crops/strength (matches manifest).
**Source:** `tab:ablation_ladder` (row A0)
**Description:** Run the frozen SD1.5 base through img2img on the held-out set with
no colour LoRA, no ControlNet, no LCM. This is the zero-baseline ablation condition —
**distinct from** the raw do-nothing baseline already computed in
`pairs/baseline_metrics/` (which compares raw Aperio vs real Hamamatsu with no model
at all). A0 runs the *untrained diffusion pipeline* through the normalisation process.
**Code (2026-08-08):** `infer_colour_lora.py`'s `--lora` was hardcoded `required=True`
(unconditionally called `load_lora_weights`), which made A0 impossible to run. Now
optional -- skips LoRA loading when absent. Added `slurm/infer_a0_baseline.slurm` as
a dedicated launcher (no `LORA_TAG` concept applies here, so it doesn't reuse
`infer_colour_lora.slurm`'s tag-parsing).
**Results (2026-08-09), `eval/a0/eval_summary.csv`, LAB-Wasserstein / SSIM (strength 0.30 shown, all 3 strengths scored):**

| scope | n_crops | LAB total | SSIM | recovery_delta_lab |
|---|---|---|---|---|
| ALL | 496 | 32.89 | 0.285 | +0.95 |
| ALL_excl_outliers | 432 | 23.65 | 0.299 | — |
| A06 (outlier) | 64 | 95.22 | 0.186 | -0.39 |
| A08 | 112 | 23.42 | 0.298 | +1.85 |
| A09 | 96 | 26.24 | 0.269 | +1.24 |
| A13 | 64 | 21.69 | 0.283 | +4.94 |
| A16 | 160 | 23.05 | 0.325 | +0.73 |

Recovery delta degrades as strength increases (0.30 → 0.50: ALL delta +0.95 → -1.00) —
higher denoising strength pushes the untrained base further from the target without
any colour signal to guide it. A06 recovery delta is negative at every strength,
consistent with its confirmed colour-gap-outlier status (`robust_z` 26-31, well past
the outlier threshold) — the frozen base cannot close that gap on its own. This is
the expected zero-baseline signature: A2/A3/A5 (colour LoRA present) should show
materially better recovery deltas, which is the point of the ablation.
**Acceptance criteria:** ✅ `eval/a0/eval_summary.csv` exists with per-slide LAB/SSIM/PSNR/MAE.

## P1-02 — A1: Base + ControlNet
**Status:** ✅ DONE (2026-08-09) — full held-out run + scoring complete.
Job 37371 (infer smoke test, COMPLETED 7:22, 12 outputs) + job 37398 (score,
COMPLETED 31s) verified first. Visual check: structure (nuclei, glandular
architecture, folds) closely preserved between reference and A1 output, colour
correctly NOT shifted toward the Hamamatsu target (no colour LoRA active in A1)
-- confirms ControlNet conditioning is genuinely constraining generation, not
being silently ignored.
**Full run:** first attempt (job 39064) landed on a broken GPU node
(mscluster65, "Unable to determine the device handle for GPU0: Unknown Error")
-- fail-fast guard correctly aborted in 3:04 instead of crawling. Resubmit
(job 39085) COMPLETED 1:35:10, 1488-row manifest, all 5 slides confirmed.
Score job 39272 COMPLETED 10:10.
**Results (2026-08-09), `eval/a1/eval_summary.csv` (strength 0.30 shown):**

| scope | n_crops | LAB total | SSIM | recovery_delta_lab |
|---|---|---|---|---|
| ALL | 496 | 33.10 | 0.389 | +0.73 |
| ALL_excl_outliers | 432 | 23.88 | 0.408 | — |
| A06 (outlier) | 64 | 95.35 | 0.260 | -0.52 |
| A08 | 112 | 23.67 | 0.411 | +1.60 |
| A09 | 96 | 26.40 | 0.376 | +1.09 |
| A13 | 64 | 22.12 | 0.367 | +4.51 |
| A16 | 160 | 23.22 | 0.441 | +0.55 |

SSIM jumps substantially over A0 at every strength (0.30: 0.285→0.389, 0.40:
0.227→0.351, 0.50: 0.181→0.312) -- ControlNet's Canny conditioning is
genuinely preserving structure, not being ignored. LAB/colour numbers track
close to A0 (as expected -- A1 has no colour LoRA, so colour correction isn't
ControlNet's job). A06 still negative recovery delta at every strength, same
as A0, for the same reason: no colour-adaptation mechanism present yet.
**Source:** `tab:ablation_ladder` (row A1); `sec:hist_lora` subsection on conditioning signals
**Description:** SD1.5 + ControlNet-Canny (pretrained, `lllyasviel/sd-controlnet-canny`,
already cached), no colour LoRA. Isolates the structural-conditioning contribution alone.
**Code (2026-08-08):** `src/eval/canny.py` (new) -- Canny edge extraction, params
(`GaussianBlur(3,3)` + Otsu-adaptive thresholds) match
`archive/empty_stubs/canny_script.py`, the actual script that produced the
proposal's `fig:canny1` figure, reused rather than picking new thresholds.
`infer_colour_lora.py` extended (not duplicated) with optional `--controlnet`/
`--controlnet-scale` -- one script now composes {LoRA, ControlNet} independently,
covering A0 (neither), A1 (ControlNet only), A2 (LoRA only, unchanged default),
and setting up A3 (both) for free. `ControlNetModel`/
`StableDiffusionControlNetImg2ImgPipeline` API (`control_image=`,
`controlnet_conditioning_scale=`) verified against the actual current diffusers
source, not guessed -- an initial web search surfaced a legacy *community*
pipeline using a different parameter name (`controlnet_conditioning_image`),
caught before writing any code by checking the real core pipeline source.
Added `slurm/infer_a1_controlnet.slurm`. Verified: syntax-checked; `--help`
confirms both flags optional with no regression to A0; `canny.py` sanity-checked
against a real crop from `pairs/train/` -- edges visibly trace nuclear/tissue
boundaries, matching `fig:canny1`'s style.
**Next step:** `sbatch slurm/infer_a1_controlnet.slurm "0.3 0.4 0.5" 0` (full
held-out set), then `sbatch slurm/score_outputs.slurm a1`.
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
**Status:** ✅ DONE (2026-08-08) — completes the rank×direction matrix.
`lora/h2a_r4/final/pytorch_lora_weights.safetensors` (3,226,184 bytes) verified,
1000/1000 steps, final loss ~0.20.
**Source:** same as P1-03a–c
**Note:** first submission (job 37255) landed on a GPU-less node and silently
CPU-crawled for 24+ min before being cancelled -- `train_colour_lora.slurm` was
missing the fail-fast CUDA guard `infer_colour_lora.slurm` already had (same
failure mode as job 3809, previously documented). Fixed: added the guard, second
submission (37345) correctly aborted in 52s on another bad node, third (37349)
completed cleanly in 4:25 on a working node.
**Evidence:** `lora/h2a_r4/final/pytorch_lora_weights.safetensors`

## P1-04 — A3: Base + ControlNet + colour LoRA
**Status:** ✅ DONE (2026-08-09) — full held-out run + scoring complete (see below).
**Source:** `tab:ablation_ladder` (row A3)
**Description:** Combine structural conditioning with colour adaptation, no LCM yet.
Tests the combination before acceleration is introduced.
**Rank decision (2026-08-09):** per P2-05's rank 4 vs rank 8 comparison (see
PHASE2-TICKETS.md), rank barely matters (~0.1-0.3 LAB units apart everywhere).
Rank 8 has the marginal empirical edge (wins on pooled `ALL` recovery delta at
strengths 0.30/0.40, and on 3/5 slides at 0.30) — picked `a2h_r8` for A3.
**Code (2026-08-09):** `slurm/infer_a3_combined.slurm` (new) — mirrors
`infer_a1_controlnet.slurm`'s structure (bigbatch, fail-fast CUDA guard) but
passes both `--lora` (`lora/a2h_r8/final`) and `--controlnet`
(`lllyasviel/sd-controlnet-canny`) to the same `infer_colour_lora.py`, which
already supports composing both flags together (no new inference code needed —
this was the point of extending it in place for P1-02). Output to `eval/a3/`.
`score_outputs.slurm` is generic over the eval-dir tag, so `sbatch
slurm/score_outputs.slurm a3` works unchanged once inference completes.
Synced to cluster, verified byte-identical (78 lines).
**Smoke test (2026-08-09):** job 39084 (infer, COMPLETED 10:03, 96 output crops,
"LoRA + ControlNet(...)" confirmed in log) + job 39105 (score, COMPLETED 1:54).
`eval/a3/eval_summary.csv` valid -- positive recovery delta (+1.96/+2.01/+2.32 at
0.30/0.40/0.50) on the default-limit A06-only sample, schema compatible end-to-end
with no changes to `score_outputs.py`.
**Full run (2026-08-09):** infer job 39125 COMPLETED 1:38:58 (1488-row manifest, all 5 slides confirmed).
Score job 39310 COMPLETED 9:16.
**Results (2026-08-09), `eval/a3/eval_summary.csv` (strength 0.30 shown):**

| scope | n_crops | LAB total | SSIM | recovery_delta_lab |
|---|---|---|---|---|
| ALL | 496 | 32.52 | 0.386 | +1.32 |
| ALL_excl_outliers | 432 | 23.35 | 0.404 | — |
| A06 (outlier) | 64 | 94.39 | 0.259 | **+0.45** |
| A08 | 112 | 23.37 | 0.408 | +1.90 |
| A09 | 96 | 25.67 | 0.372 | +1.81 |
| A13 | 64 | 21.66 | 0.363 | +4.97 |
| A16 | 160 | 22.62 | 0.438 | +1.16 |

**A06 recovers for the first time in the ladder** — positive recovery delta at
every strength (+0.45/+0.56/+0.88 @0.30/0.40/0.50), versus negative at every
strength for both A0 (frozen base) and A1 (+ControlNet alone). SSIM is
essentially unchanged from A1 (structure preservation holds up), while LAB
recovery improves on every non-outlier slide too (e.g. A08 @0.30: A1 +1.60 →
A3 +1.90). ControlNet + colour LoRA together are doing real, additive work
here, not just each pulling in its own direction — the combination is the
best-performing rung of the ladder so far.
**Acceptance criteria:** ✅ `eval/a3/eval_summary.csv` exists with per-slide LAB/SSIM/PSNR/MAE.

## P1-05 — A4: Full pipeline (+ LCM-LoRA)
**Status:** TODO — unblocked (P1-04 done); not yet started
**Source:** `tab:ablation_ladder` (row A4); `sec:training_order` step on LCM-LoRA attachment
**Description:** Attach the pretrained `latent-consistency/lcm-lora-sdv1-5` (already
cached) AFTER colour LoRA and ControlNet are frozen. LCM-LoRA itself is never trained —
only its compatibility with the trained adapters is evaluated (4-step LCM vs 50-step
DDIM on held-out MITOS; SSIM/PSNR + artefact inspection per `sec:experiments`).

## P1-06 — A5 histopathology warm-start LoRA training
**Status:** ✅ DONE (2026-08-08) — job 37114 COMPLETED, 3000/3000 steps, 521.7s.
`lora/hist_r32/final/pytorch_lora_weights.safetensors` (25,548,064 bytes) verified
present on the cluster. Frozen per sec:training_order -- do not retrain jointly
with the A5 colour LoRA. Smoke test (job 37109) also verified beforehand.
**Source:** `sec:hist_lora`; rank 32, batch 1, 3,000 steps, 3,000 patches
(1,000 PanNuke + 1,500 TCGA-BRCA + 500 MITOS)
**Evidence pool ready:** `/datasets/mhoosen/stain-norm/hist_lora_pool/` (3,000 files,
manifests present, held-out leak check passed at build time; `composition.json`
confirms 1000/1500/500 split matches the proposal exactly).
**Code (2026-08-08):** `src/train/train_hist_lora.py` + `slurm/train_hist_lora.slurm`
written — the existing `train_colour_lora.py` can't do this (hardcoded to paired
A/H crops with a required `--direction`; the hist pool is unpaired, different
naming, no direction concept). Training loop mirrors `train_colour_lora.py`
(already validated by 4 real completed runs) — same `LoraConfig`, confirmed
identical to HF's own official `train_text_to_image_lora.py` example. New code
is `load_pool_manifest()`, which independently re-checks the pool for the 5
held-out MITOS slides (exact `slide_id` match, not substring) before training —
verified against real cluster data (exactly 3000 patches loaded, zero leaks, all
spot-checked files exist) and against a synthetic injected leak (correctly
aborts).
**Constraint (sec:training_order):** must be trained and FROZEN before colour-LoRA
training begins for this leg — do not train jointly.
**Next step:** none for this ticket. P1-07 (A5 full ablation) can proceed once P1-04
(ControlNet + colour LoRA, blocked on P1-02) is also done.

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
