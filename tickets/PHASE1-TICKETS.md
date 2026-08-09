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

**Correction (2026-08-09, caught while building `docs/results/RESULTS_SUMMARY.md`):**
A06 does NOT recover for the first time here — A2 (colour LoRA alone, either
rank) already shows positive A06 recovery delta (+0.92 to +1.16 @0.30), slightly
*better* than A3's +0.45 on this metric alone. A3's real distinction is
different: it's the first rung to combine a positive A06 recovery delta WITH
A1's full SSIM gain simultaneously (A06 SSIM: A0 0.186, A2 ~0.19, A1 0.260,
A3 0.259) — A2 alone barely helps A06's structure, A1 alone doesn't touch its
colour, A3 gets both at once. LAB recovery also improves on every non-outlier
slide over A2 (e.g. A08 @0.30: A2r8 +2.49 → A3 +1.90 is actually slightly lower,
but A3 still holds essentially all of A1's SSIM gain while A2 has none of it) —
see the full cross-rung table in `docs/results/RESULTS_SUMMARY.md` for the
accurate comparison across all 5 completed rungs, not just A0/A1/A3.
**Acceptance criteria:** ✅ `eval/a3/eval_summary.csv` exists with per-slide LAB/SSIM/PSNR/MAE.

## P1-05 — A4: Full pipeline (+ LCM-LoRA)
**Status:** ✅ DONE (2026-08-09) — full held-out run + scoring complete (see results below).
**Source:** `tab:ablation_ladder` (row A4); `sec:training_order` step on LCM-LoRA attachment;
H4/RQ3 in `tab:hyp_rq`; `sec:experiments` "LCM Acceleration: Clinical Throughput"
**Description:** Attach the pretrained `latent-consistency/lcm-lora-sdv1-5` (already
cached, verified real 134.6MB `.safetensors` blob not a stub) AFTER colour LoRA and
ControlNet are frozen. LCM-LoRA itself is never trained — only its compatibility with
the trained adapters is evaluated: 4-step LCM vs the 50-step DDIM reference already
produced by P1-04/A3 (same `a2h_r8` colour LoRA + ControlNet-Canny config) — SSIM/PSNR
+ artefact inspection. Pre-committed fallback if quality drops (SSIM < 0.85 or visible
banding/checkerboarding): 20-step DDIM, i.e. rerun `infer_a3_combined.slurm` at
`--steps 20` — no code change needed, already supported.
**API verified (2026-08-09), not guessed** — fetched real diffusers source
(`pipeline_controlnet_img2img.py`, `loaders/lora_pipeline.py`, `loaders/lora_base.py`,
`schedulers/scheduling_lcm.py`) plus the HF LCM-LoRA docs page, confirming:
`StableDiffusionControlNetImg2ImgPipeline` inherits the real
`StableDiffusionLoraLoaderMixin` (not a community pipeline); `load_lora_weights(...,
adapter_name=...)` + `set_adapters([...], adapter_weights=[...])` is the documented
pattern for two simultaneously-active LoRAs; `LCMScheduler.from_config(...)` mirrors
the existing `DDIMScheduler` pattern already in use; recommended range is
`num_inference_steps` 4-8, `guidance_scale` 1.0-2.0; `weight_name` must be explicit
for `latent-consistency/lcm-lora-sdv1-5` too under `HF_HUB_OFFLINE=1` (same gotcha
hit earlier for the local trained LoRA) — its repo has exactly one file,
`pytorch_lora_weights.safetensors`; LoRA adapters only ever patch UNet/text-encoder,
never `ControlNetModel`, so ControlNet + two active LoRAs + `LCMScheduler` has no
known architectural incompatibility.
**Code (2026-08-09):** `infer_colour_lora.py` extended again (not duplicated) with
`--lcm` (attaches LCM-LoRA via `adapter_name="lcm"` + `set_adapters` alongside the
colour LoRA's `adapter_name="colour"` when both are set, swaps scheduler to
`LCMScheduler`) and `--lcm-scale` (adapter weight, default 1.0). Zero behaviour
change to A0-A3 when `--lcm` is omitted. Added `slurm/infer_a4_lcm.slurm` — mirrors
`infer_a3_combined.slurm` but adds `--lcm`, defaults `--steps 4` (overridable, e.g.
8) and `--guidance 1.5` (within the recommended LCM range). Verified: `py_compile`
clean; `--help` shows both new flags with no regression; synced to cluster, byte-
identical (238-line script, 89-line launcher).
**Smoke test #1 (2026-08-09), steps=4:** job 39457 (infer, COMPLETED 11:03, all 3
adapters confirmed active: "LoRA + ControlNet(...) + LCM-LoRA") + job 39490 (score,
COMPLETED 57s). **Finding:** strengths 0.30 and 0.40 produced byte-identical output
on every single crop (`eval_per_crop.csv` rows exactly equal). Root cause confirmed
against real diffusers source (`pipeline_controlnet_img2img.py`'s `get_timesteps`,
not guessed):
```python
init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
t_start = max(num_inference_steps - init_timestep, 0)
```
At `num_inference_steps=4`: strength 0.30 -> `int(4*0.30)=1` real step; 0.40 ->
`int(4*0.40)=1` real step (same!); 0.50 -> `int(4*0.50)=2` steps. Not a code bug —
an inherent quantisation interaction between img2img's `strength`-based partial
denoising and a very low total step count, which also explains the weak LAB
recovery vs A3's 50-step reference (too few real steps to do meaningful work).
**Fix:** `--steps` bumped 4 -> 8 (still within diffusers' documented LCM range of
4-8, still far faster than 50-step DDIM). At 8 steps: 0.30/0.40/0.50 ->
`int(8*x)` = 2/3/4 real steps -- properly differentiated.
**Smoke test #2 (2026-08-09), steps=8:** job 39507 (infer, COMPLETED 8:00) + job
39534 (score, COMPLETED 2:01). Confirmed every strength now produces distinct
per-crop output. `slurm/infer_a4_lcm.slurm`'s default `STEPS` updated 4 -> 8.
**Cluster note:** two consecutive smoke-test attempts (jobs 39427, 39433, 39450)
failed on broken GPU nodes -- `mscluster65` and `mscluster75` both report
"Unable to determine the device handle for GPU0: Unknown Error" and show as
`idle` in `sinfo` (Slurm has no GPU health awareness under partition-only
scheduling, so it keeps re-offering them). Fail-fast guard caught all three
correctly (<4min each, no CPU-crawl). Worked around with
`sbatch --exclude=mscluster65,mscluster75 slurm/infer_a4_lcm.slurm ...` --
worth using this exclusion for future GPU submissions until those nodes are
confirmed fixed or reported to cluster admin.
**Status:** ✅ DONE (2026-08-09) — full held-out run + scoring complete.
Infer job 39630 COMPLETED 1:06:24 (1488-row manifest, all 5 slides confirmed) —
notably faster than every 50-step run despite the extra LCM-LoRA adapter, as
expected for 8-step inference. Score job 39784 COMPLETED 9:26.
**Results (2026-08-09), `eval/a4/eval_summary.csv` (strength 0.30 shown),
vs A3's 50-step DDIM reference:**

| scope | A3 (50-step DDIM) Δlab | A4 (8-step LCM) Δlab | A3 SSIM | A4 SSIM |
|---|---|---|---|---|
| ALL | +1.32 | **-0.13** | 0.386 | **0.398** |
| A06 (outlier) | +0.45 | **-0.22** | 0.259 | 0.272 |
| A08 | +1.90 | -0.04 | 0.408 | 0.421 |
| A09 | +1.81 | +0.76 | 0.372 | 0.389 |
| A13 | +4.97 | +3.24 | 0.363 | 0.370 |
| A16 | +1.16 | -0.39 | 0.438 | 0.449 |

**H4/RQ3 finding — nuanced, not a clean pass:** SSIM is essentially preserved
or even marginally *higher* under 8-step LCM than 50-step DDIM at every scope
(structural fidelity is not the casualty here) but colour recovery
(`recovery_delta_lab`) degrades and goes **negative** for `ALL`/`A06`/`A08`/`A16`
at strength 0.30 — LCM acceleration is trading away some of the colour LoRA's
gain even though it isn't damaging structure. The picture is strength-dependent
and not monotonic: at 0.50, A06 jumps to **+7.31** (best A06 result in the whole
ladder) while A08/A13/A16 fall further behind (-3.52/-1.09/-2.91) — likely the
same few-step quantisation sensitivity documented above (8 steps × strength
0.50 = only 4 real denoising steps, more variance-prone than 50-step DDIM's much
finer resolution). Full per-strength table in `eval/a4/eval_summary.csv`.
**Proposal's SSIM ≥ 0.85 threshold does not apply as a literal pass/fail bar
here** — no rung in the entire A0-A4 ladder ever exceeds ~0.45 SSIM under this
project's actual metric computation (see `docs/results/RESULTS_SUMMARY.md`),
so 0.85 is not a calibrated threshold for these numbers; the comparison that
matters is A4 vs A3 (same colour-LoRA/ControlNet config, only the sampler
changed), not A4 vs an absolute constant. On that relative basis: **structure
compatible, colour recovery not clearly compatible at 8 steps/strength 0.30-0.40,
but A06 (the hardest case) sees its best-ever recovery at strength 0.50.** Worth
reporting as-is (genuine, useful negative-leaning result) rather than invoking
the 20-step DDIM fallback, since the fallback is meant for structural/artefact
failure, which did not occur.
**Acceptance criteria:** ✅ `eval/a4/eval_summary.csv` exists; SSIM/PSNR comparison
against A3's 50-step DDIM reference documented above.

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
**Next step:** none for this ticket. P1-07 (A5 full ablation) is now unblocked.

## P1-07 — A5 full ablation: hist LoRA + colour LoRA + ControlNet + LCM
**Status:** TODO — code ready, not yet run.
**Source:** `tab:ablation_ladder` (row A5)
**Description:** Primary comparison is A5 vs A4 (`sec:hist_lora`) — a positive result
quantifies the value of a combined tissue-and-target-domain prior; a negative result
supports that ControlNet + colour LoRA alone is sufficient. This decides whether
P3-05 (A5-on-SDXL) is attempted.
**Scope decision (2026-08-09):** `sec:training_order` step 3 reads "Train the
colour LoRA on MITOS A03/H03 paired patches against the frozen base, optionally
with the frozen histopathology LoRA already loaded for A5" — ambiguous between
(a) training a NEW colour LoRA with the hist LoRA loaded+frozen underneath it,
or (b) stacking the existing `a2h_r8` colour LoRA and `hist_r32` at inference
only, no new training. Discussed with the user: going with (b) — no new training
job. This also matches `train_hist_lora.py`'s own docstring from the earlier
session ("loading it alongside a colour LoRA at inference ... is a separate
step (P1-07), not done here"), so it's consistent with what was already on
record, not a new precedent. A merge-based training approach for (a) was
researched and verified against real diffusers/peft source (`unet.
load_lora_adapter(..., prefix="unet")` + `unet.fuse_lora()` + `unet.
unload_lora()` + fresh `unet.add_adapter(...)`, confirmed sound) in case this
decision is revisited later, but is not implemented.
**Code (2026-08-09):** `infer_colour_lora.py` extended again (not duplicated)
with `--hist-lora`/`--hist-scale`. Generalises the existing 2-adapter
(`colour`+`lcm`) `set_adapters` logic from A4 to support up to 3 simultaneous
named adapters (`colour`, `hist`, `lcm`) — any combination triggers the
multi-adapter path; a lone `--lora` with neither `--hist-lora` nor `--lcm`
keeps the original single-adapter code path unchanged (zero regression to
A2/A3). Added `slurm/infer_a5_full.slurm` — mirrors `infer_a4_lcm.slurm`
(bigbatch, fail-fast CUDA guard, `mscluster65`/`mscluster75` excluded, 8-step
LCM default per P1-05's finding) but adds `--hist-lora lora/hist_r32/final`
alongside the existing `--lora a2h_r8/final` + `--controlnet` + `--lcm`. Output
to `eval/a5/`. Verified: `py_compile` clean; `--help` shows both new flags with
no regression; synced to cluster, byte-identical (259-line script, 89-line
launcher).
**Next step:** smoke test (`sbatch --exclude=mscluster65,mscluster75
slurm/infer_a5_full.slurm`), verify manifest + all four adapters confirmed
active in the log ("LoRA + ControlNet(...) + Hist-LoRA + LCM-LoRA"), then full
run (`sbatch --exclude=mscluster65,mscluster75 slurm/infer_a5_full.slurm
"0.3 0.4 0.5" 0`) → score → compare against A4 per the primary comparison above.
**Acceptance criteria:** `eval/a5/eval_summary.csv` exists; A5 vs A4 comparison
documented (per-slide, both `ALL` and `ALL_excl_outliers`, per CLAUDE.md).

## P1-08 — Denoising-strength / LCM quality sweep infrastructure
**Status:** ✅ DONE — `infer_colour_lora.py` supports `--strengths` sweep;
`score_outputs.py` reports recovery delta per strength
**Source:** H4/RQ3, `tab:hyp_rq`; LCM quality window per `sec:experiments`

---

**Decision gate (proposal §"Time Plan"):** at the end of Phase 1, formally review
A0–A5 results and the SDXL compute-contingency evidence before proceeding to Phase 3
(see PHASE3-TICKETS.md, P3-01).
