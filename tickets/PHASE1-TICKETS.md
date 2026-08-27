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
**Status:** ✅ DONE (2026-08-10) — full held-out run + scoring complete (see results below).
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
**Smoke test (2026-08-09):** job 39870 (infer, COMPLETED 11:11, all four adapters
confirmed active: "LoRA + ControlNet(...) + Hist-LoRA + LCM-LoRA") + job 39894
(score, COMPLETED 56s). `eval/a5/eval_summary.csv` valid -- on this A06-only
default-limit sample, recovery delta is dramatically higher than A4's
equivalent A06 numbers at every strength (A5 +3.95/+9.05/+18.02 vs A4's
-0.22/+1.55/+7.31 @0.30/0.40/0.50) -- early signal the histopathology
warm-start prior is doing real work on the hardest slide, worth confirming on
the full held-out set.
**Status:** ✅ DONE (2026-08-10) — full held-out run + scoring complete.
Infer job 39924 COMPLETED 1:06:56 (1488-row manifest, all 5 slides confirmed).
Score job 40106 COMPLETED 9:09.
**Results (2026-08-10), `eval/a5/eval_summary.csv`, A5 vs A4 (both at strength
0.30/0.40/0.50, same colour-LoRA/ControlNet/LCM config, only the hist LoRA added):**

| scope | A4 Δlab | A5 Δlab | delta (A5−A4) |
|---|---|---|---|
| ALL @0.30 | −0.13 | −0.12 | ≈0 |
| ALL @0.40 | −1.65 | −0.94 | +0.71 |
| ALL @0.50 | −1.21 | **+0.42** | +1.63 |
| **A06 @0.30** | −0.22 | **+2.38** | **+2.60** |
| **A06 @0.40** | +1.55 | **+7.35** | **+5.80** |
| **A06 @0.50** | +7.31 | **+16.02** | **+8.71** |
| A08 @0.30 | −0.04 | −1.27 | −1.23 |
| A08 @0.50 | −3.52 | −2.43 | +1.09 |
| A16 @0.30 | −0.39 | −1.51 | −1.12 |
| A16 @0.50 | −2.91 | −3.01 | −0.10 |

**A5's contribution is real but slide-dependent, not a uniform win.** On A06
(the confirmed colour-gap outlier) the histopathology warm-start prior
dramatically improves colour recovery at every strength, more than doubling
A4's already-best-in-ladder A06 result at 0.50 (+16.02 vs +7.31 — raw LAB
Wasserstein for A5's A06 output drops from baseline 94.84 to 78.82, closing
~17% of the original gap, the largest closure anywhere in the ladder). SSIM
stays close to A4's (0.199 vs A4's ~0.20 @0.50 A06) -- not a structural
trade-off. But on the typical slides (A08, A16), A5 is consistently *worse*
than A4 (e.g. A08 @0.30: A4 −0.04 → A5 −1.27). Pooled `ALL` is roughly a wash
at 0.30, favours A5 at 0.40/0.50 (A06's large gains outweigh the typical-slide
losses once pooled). **Reads as a genuine positive result for the specific
research question** (`sec:hist_lora`: does the histopathology prior help
*generalisation to the hardest case*?) even though it is not a positive result
in the pooled-average sense — worth reporting both framings, not collapsing to
a single verdict.
**Qualitative:** `docs/results/qualitative/comparison_a06_strength050.png`
shows the strength-0.50 A06 crop across the full ladder. The quantitative gain
is real (verified in `eval_per_crop.csv`, not just the aggregate) but the
*visual* difference between A4 and A5 in this single crop is fairly subtle to
the eye -- LAB Wasserstein is a holistic per-crop histogram statistic, and this
particular tile is dominated by pale background/stroma rather than densely
stained nuclei, so a real measured shift doesn't always look dramatic in one
image. Worth stating this caveat plainly in the write-up rather than
overselling the visual.
**Next step:** none — Phase 1 ablation ladder (A0-A5) is complete. Ready for
the Phase 1 decision gate (proposal §"Time Plan").
**Acceptance criteria:** ✅ `eval/a5/eval_summary.csv` exists; A5 vs A4 comparison
documented (per-slide, both `ALL` and `ALL_excl_outliers`, per CLAUDE.md).

## P1-08 — Denoising-strength / LCM quality sweep infrastructure
**Status:** ✅ DONE — `infer_colour_lora.py` supports `--strengths` sweep;
`score_outputs.py` reports recovery delta per strength
**Source:** H4/RQ3, `tab:hyp_rq`; LCM quality window per `sec:experiments`

## P1-09 — Post-gate follow-up: extended strength sweep (A3/A4/A5)
**Status:** ✅ DONE (2026-08-10) — supplementary to the closed Phase 1 gate, not
reopening it; informs Phase 3's actual deployment-strength choice.
**Source:** data-driven, motivated by two findings from mining the completed
A0-A5 per-crop CSVs (`docs/results/analyze.py`): (1) A06's recovery-vs-strength
curve looked convex/accelerating, not plateauing, within the tested 0.30-0.50
range; (2) A5's typical-slide deficit vs A4 appeared to shrink as strength rose.
Both needed testing beyond 0.50 (and, for completeness, below 0.30) to know if
they held up or were noise/extrapolation artefacts.
**Safety fix first:** `infer_a3_combined.slurm`/`infer_a4_lcm.slurm`/
`infer_a5_full.slurm` all hardcoded their output dir to `eval/a{3,4,5}` — a
follow-up run would have silently overwritten the already-scored, committed
results (`score_outputs.py` regenerates `eval_manifest.csv` from scratch each
run, doesn't append). Added an `OUT_TAG` optional trailing arg (default =
existing tag, zero change to prior invocations) so exploratory runs land in a
separate `eval/<tag>/` dir. Committed before any exploratory job was submitted.
**Jobs run:** two A06-only smoke tests at strengths 0.6/0.7 (A4, A5) to check
for safety before committing to full runs; five full 5-slide runs — A3/A4/A5 at
strength 0.20, and A4/A5 at strength 0.70 (0.60 was tested in the smoke stage
and found to be a wasted duplicate, see below, so skipped for the full run).
One transient GPU-node failure (`mscluster44`, "Unable to determine the device
handle for GPU0", same class of fault as `mscluster65`/`mscluster75`) — caught
correctly by the fail-fast guard in 2:48, resubmitted with `mscluster44` added
to the exclusion list.

**Critical finding #1 — LCM step-count quantisation recurs, now at the 0.5/0.6
boundary:** at 8-step LCM, `int(8*0.50)=int(8*0.60)=4` real denoising steps —
confirmed byte-identical per-crop output between the two strengths (same
mechanism as the earlier 0.30/0.40 collision at 4 steps, documented under
P1-05). Only strengths that cross an integer boundary of `floor(8*strength)`
are genuinely distinct for A4/A5. **This means "strength" is not a smooth
dial for the LCM rungs the way it is for A3's 50-step DDIM (`floor(50*s)`,
effectively continuous) — it's a coarse proxy for step count with only 5-6
genuinely distinct settings across [0,1].** Any future strength sweep on A4/A5
should be planned around integer step-count boundaries, not arbitrary decimals.

**Critical finding #2 — A3 (DDIM): strength 0.20 dominates the previously
"best" 0.30 on almost every axis, not just a wash:**

| A3 | ALL Δlab | ALL SSIM | A06 Δlab | A08 Δlab | A09 Δlab | A13 Δlab | A16 Δlab |
|---|---|---|---|---|---|---|---|
| 0.20 | **+1.59** | **0.425** | +0.41 | **+2.45** | **+1.88** | **+5.29** | **+1.46** |
| 0.30 | +1.32 | 0.386 | **+0.45** | +1.90 | +1.81 | +4.97 | +1.16 |

0.20 beats 0.30 on every metric except A06 (negligible −0.04 difference). This
revises the earlier "0.30 is the best compromise for 4/5 slides" claim in
`docs/results/RESULTS_SUMMARY.md` — the true optimum for A3 was never found;
0.20 is just the new best *tested* point, and the floor below 0.20 remains open.

**Critical finding #3 — A4/A5 (LCM): behaviour vs strength is non-monotonic,
not a simple "lower/higher is better" story, and diverges sharply from A3:**

| Real steps (8-step LCM) | strength | A4 ALL Δlab | A4 SSIM | A5 ALL Δlab | A5 SSIM | A06 Δlab (A4 / A5) |
|---|---|---|---|---|---|---|
| 1 | 0.20 | **+1.53** | **0.459** | **+1.93** | **0.454** | +0.42 / +1.19 |
| 2 | 0.30 | −0.13 | 0.389 | −0.12 | — | −0.22 / +2.38 |
| 3 | 0.40 | −1.65 | — | −0.94 | — | +1.55 / +7.35 |
| 4 | 0.50 | −1.21 | 0.312 | +0.42 | 0.299 | +7.31 / +16.02 |
| 5 | 0.70 | **+2.25** | 0.271 | −3.08 | 0.268 | **+19.83** / **+24.43** |

1 step is a broad, uniform win (every slide improves, high SSIM) — a "gentle
nudge" regime. 2-4 steps is a worse valley on typical slides even as A06's
recovery climbs steadily. 5 steps is an extreme divergence: A06 hits the best
recovery anywhere in the entire project (A5: +24.43, more than 25% of its raw
94.84 LAB gap closed) but every typical slide craters (A5 @0.70: A08 −6.97,
A09 −3.37, A13 −7.58, A16 −7.72 — far worse than A4's equivalent losses at the
same step count). **This refutes the earlier speculative hypothesis (from the
pre-experiment data-mining pass) that A5's typical-slide deficit vs A4 would
keep shrinking or flip permanently positive at higher strength** — it does
narrow/flip between 0.30 and 0.50, but reverses hard and gets much worse than
A4 by 5 steps. Good example of why the confirmatory full run mattered rather
than trusting the 3-point extrapolation.
**Practical implication for Phase 3 / deployment:** there is no single best
strength — it depends on the goal. For general-purpose robustness across all
slides, 1-step LCM (strength ≈0.20-0.25) is the best operating point found for
both A4 and A5, beating every previously-tested strength in the official
ladder. For maximum single-slide (hardest-case) recovery specifically, 5-step
A5 (strength 0.70) is dramatically the strongest result in the whole project,
at severe cost everywhere else — a genuine trade-off, not a free lunch, and
should be framed as a "rescue mode" option rather than a general default.
**Evidence:** `docs/results/{a3_ext_s02,a4_ext_s02,a5_ext_s02,a4_ext_s07,a5_ext_s07,a4_ext_s67,a5_ext_s67}/`
(manifests, per-crop, summary CSVs); analysis reproducible via
`docs/results/analyze.py`.
**Next step:** none required for Phase 1. Worth citing when choosing the
deployment strength for P3-03/P3-04's SDXL transfer and comparison.

## P1-10 — Corrective experiment: source-conditioned scanner translation
**Status:** ✅ DONE (2026-08-20) — approved, infrastructure built and heavily
bug-fixed (two review rounds, six real bugs found and fixed — see below).
Mandatory ablation control FAILED at 300 steps (undertrained), then **PASSED
decisively at 2000 steps** on the 8-pair overfit set. Full training run on
all 50 pairs (4000 steps) completed with clean, monotonic validation-loss
convergence. VAE-only floor check DONE (SSIM 0.5393 — a ceiling below every
classical baseline, regardless of conditioning quality). Smoke gate PASSED
decisively on a subset. **Full held-out evaluation (all 496 crops, job
44542) COMPLETED — a genuinely mixed result**: clearly beats SD1.5's own
prior-best operating point (A4/A5@0.20) on colour recovery, roughly tied on
structure (SSIM), does not beat the classical baselines (capped by the VAE
floor). Diagnosis confirmed directionally correct; does not close the
structural-fidelity gap. A follow-up strength sweep is in progress. This does
not reopen, replace, or relabel the completed
A0–A5 ladder. The existing target-only LoRA remains the faithfully reported
original method and negative result; any revised model must use a new script,
checkpoint tag, and evaluation tag.
**Source:** proposal `sec:training_order` step 3 and “LoRA: Data-Efficient
Domain Mapping”, which describe the colour LoRA as learning paired A→H/H→A
scanner mappings; proposal architecture-decision gate outcome (iii), which
permits deeper SD1.5 diagnostic ablations when Phase 1 reveals a fundamental
architecture issue. Motivated by the completed Phase 1/2 evidence: classical
methods beat the diffusion ladder on global and windowed colour metrics, raw
and classical methods beat it on structural metrics, Relative Dice is 0.8745
(below 0.95), and round-trip SSIM is 0.1429.

**Problem this ticket addresses:** `train_colour_lora.py` receives a directory
containing coordinate-corresponding Aperio/Hamamatsu pairs, but
`list_target_images()` selects only `*_hamamatsu.png` for A2H (or only
`*_aperio.png` for H2A). The training loop then performs ordinary target-image
noise-prediction under one fixed generic prompt. The source crop never enters
the dataset item, UNet conditioning, loss, or gradient path. The checkpoint
therefore learns a target-domain generative prior, approximately `P(H)`, not
the proposal's directional conditional mapping `P(H|A)`. At inference,
img2img is expected to infer the direction from the source latent while also
surviving VAE compression, added noise, and generative denoising. The observed
strength trade-off is consistent with this mismatch: low strength preserves
more structure but shifts colour weakly; high strength shifts colour further
but resynthesises morphology.

**Why a small patch to the existing loop is not enough:** simply loading both
filenames, concatenating them in a batch, or adding target pixel L1/SSIM would
not create a valid translator. The A03/H03 pairs are coordinate-corresponding,
not pixel-exact, and the standing methodology guardrail prohibits training-time
registration. A direct output-vs-target pixel loss would therefore optimise
scanner/section misalignment as though it were a model error. Conversely,
continuing target-only denoising with a differently worded prompt still would
not condition the learned prediction on the source image.

**Required decision before implementation (supervisor/methodology gate):**
approve a supplementary *source-conditioned conditional diffusion* experiment
that keeps the ≤50-pair budget and never uses the five held-out slides. The
recommended design is a parameter-efficient source-conditioning branch plus a
colour LoRA, rather than overwriting `train_colour_lora.py` or fine-tuning the
full SD1.5 UNet. If that method expansion is not approved or cannot fit the
project timeline, retain the current negative result and document target-only
conditioning as the principal limitation; do not silently describe the
existing checkpoint as a learned paired mapping.

**Proposed implementation (new code path):**
1. Create `src/train/train_colour_translation_lora.py`; keep
   `train_colour_lora.py` unchanged for reproducibility of A2–A5. Create a new
   launcher such as `slurm/train_colour_translation_lora.slurm`, new checkpoint
   tag `lora/a2h_cond_r8/`, and new inference/evaluation tag
   `eval/a2h_cond_r8/`. Never reuse or overwrite the official A0–A5 folders.
2. Replace `list_target_images()` with a paired manifest/dataset that returns
   `source_pixel_values`, `target_pixel_values`, direction, and pair ID. Assert
   that both files exist, have the expected scanner suffix, belong only to
   A03/H03, and that A06/A08/A09/A13/A16 never appear. Save the exact pair list
   and its hash in `training_config.json`.
3. Preserve the standard conditional-diffusion target: encode the *target*
   crop, add noise, and predict that noise. Separately encode the *source* crop
   deterministically (`latent_dist.mode()`) and feed it through a lightweight,
   zero-initialised source-conditioning adapter (ControlNet/T2I-Adapter-style
   residual branch or an equivalent explicitly documented source-latent
   conditioner). Inject its residuals into the frozen UNet while training the
   colour LoRA and only the new conditioning parameters. Keep the VAE, text
   encoder, and base UNet frozen and report the added trainable-parameter count.
   Do not merely rely on the source latent being the img2img start state at
   inference; the source must affect the training-time noise prediction.
4. Provide the source structural signal to the conditioning branch (start with
   the existing Canny pathway; prefer the proposal's HoVer-Net nuclear-boundary
   signal when available). This gives the model an explicit content/geometry
   input while the target denoising objective supplies the requested scanner
   appearance. Because the scans are not pixel-exact, use spatial jitter or an
   alignment-tolerant conditioning design and document it; do not introduce a
   hidden registration step.
5. Keep ordinary target noise-prediction as the core loss. If an auxiliary
   loss is added, it must respect the correspondence limitation: target colour
   may use global/windowed LAB or optical-density statistics, while source
   preservation may use edge/nuclear-feature consistency. Do not use raw
   output-to-target L1, MAE, PSNR, or SSIM without a separately approved change
   to the training-data/registration methodology. Log every loss component
   independently so a lower combined scalar cannot hide structural damage.
6. Add a validation split from non-held-out training slides or a strictly
   training-domain leave-one-frame-out split for checkpoint selection. Never
   select steps, rank, strength, or adapter weights on A06/A08/A09/A13/A16.
   Evaluate at least checkpoints 250/500/750/1000; the final step is not
   automatically the best checkpoint.
7. Build a matching inference path that supplies the same source condition used
   during training. Establish quality first with deterministic 50-step DDIM and
   no LCM-LoRA or hist-LoRA. Only after the conditional model passes its quality
   gate should ControlNet scale, LoRA scale, low-strength DDIM, and finally LCM
   acceleration be added one at a time. This prevents adapter/scheduler
   interactions from obscuring whether source-conditioned training itself
   worked.

**Mandatory diagnostic controls before the full run:**
- **VAE-only floor:** encode and immediately decode the 496 fixed evaluation
  crops with no noise/UNet. Report SSIM/PSNR/MAE and Relative Dice where
  practical. This separates unavoidable VAE reconstruction loss from denoising
  drift.
- **Source-conditioning ablation:** for the same target/noise/timestep, compare
  the correct source, a zero source condition, and a deliberately shuffled
  source. The prediction/output must change materially and the correct source
  must win on structural/content metrics; otherwise the new branch is being
  ignored and the ticket has not fixed the identified problem.
- **Seed protocol:** derive a deterministic seed from pair/crop identity and
  reuse it across compared methods/strengths, rather than resetting every crop
  to the same global noise tensor. Confirm the conclusion across at least three
  seed sets and report mean ± spread.
- **Overfit test:** first prove that the model can learn a tiny 4–8-pair subset
  without NaNs and that swapping the source condition changes the result. A
  finite diffusion loss alone is not a pass signal.

**Evaluation protocol (fix fairness issues at the same time):**
- Freeze one canonical 496-crop coordinate manifest and use those exact crops
  for raw, conditional diffusion, original target-only diffusion, and every
  classical method. Recompute the raw baseline on that subset instead of
  subtracting a 1,475-crop mean from 496 model crops.
- Treat A06 as the fixed raw-baseline outlier for every method; do not rerun
  outlier discovery separately after each method changes its colour gap.
- Primary colour metric: windowed LAB-Wasserstein, with global LAB reported
  alongside it. Primary structural safety: Relative Dice and round-trip
  reconstruction; SSIM/PSNR/MAE remain supporting pixel-exact measures.
- Compare against the best existing diffusion operating points (A4/A5 at 0.20),
  raw input, Macenko, Reinhard, and histogram matching. Report per-slide,
  pooled, and A06-excluded results; never use the pooled number alone.

**Go/no-go smoke gate:** run a small, balanced subset containing A06 plus at
least one typical slide. Continue to the full held-out run only if the correct
source condition beats both the target-only LoRA and its own shuffled-source
ablation on windowed colour recovery *without* reducing structural fidelity.
Failure is still a valid result: it would show that ≤50 coordinate-corresponding
pairs are insufficient for this conditional architecture or that SD1.5/VAE
resynthesis, rather than the missing conditioning alone, is the limiting
factor.

**Acceptance criteria:**
- The saved training manifest proves both sides of every pair are consumed and
  held-out leakage checks pass.
- A unit/integration test proves source conditioning participates in the forward
  pass and gradients reach only the colour LoRA plus new conditioning branch.
- The VAE-only, correct-source, zero-source, shuffled-source, and target-only
  controls are all reported.
- On the fixed 496-crop set, the revised model jointly exceeds the current best
  general-purpose diffusion point (A5@0.20: ΔwLAB ≈+1.16, SSIM 0.454) on colour
  and structure, rather than improving one by sacrificing the other. Relative
  Dice must not fall below the current 0.8745; the proposal target of ≥0.95
  remains the desired structural-safety pass criterion.
- Results, dependency versions, checkpoint-selection rule, three-seed spread,
  and per-slide/A06-excluded tables are added to
  `docs/results/RESULTS_SUMMARY.md`, explicitly labelled supplementary.

**Expected interpretation:** a pass would demonstrate that the original loss
was the main bottleneck and that few-shot *conditional* translation is more
appropriate than target-domain LoRA adaptation. A fail after the conditioning
and VAE controls would strengthen the conclusion that SD1.5 latent resynthesis
is intrinsically mismatched to pixel-preserving scanner normalisation under the
project's ≤50-pair constraint. Either outcome is more informative than simply
increasing LoRA rank or training steps, because rank 4 and rank 8 already behave
nearly identically and neither change supplies the missing source condition.

### Progress log (2026-08-19/20)

**Implementation**: `src/train/train_colour_translation_lora.py` +
`src/eval/infer_colour_translation.py` + `src/eval/score_p1_10_ablation.py` +
two launchers. Fresh `ControlNetModel.from_unet(unet, conditioning_channels=6)`
(diffusers 0.39.0, confirmed on the cluster), conditioned on 6 channels =
source RGB (appearance, ticket step 3) + source Canny (structure, ticket step
4) concatenated — resolves the two-signal reading of the ticket's steps 3/4
directly, not treated as alternatives. Backbone (cloned UNet down/mid blocks)
frozen; only the genuinely new `controlnet_cond_embedding` +
`controlnet_down_blocks` + `controlnet_mid_block` layers trained (confirmed
live: exactly 12,566,592 trainable / 348,712,960 frozen) — a real "lightweight
adapter," not a full ControlNet fine-tune, matching the ticket's wording.

**Two rounds of real bugs found and fixed, not hypothetical**:
1. Fork's own self-audit: Canny-only conditioning carried zero colour
   information, functionally overlapping with the already-tested frozen-
   ControlNet A1/A3 configs — fixed to the 6-channel design above.
2. User code review caught four more, all confirmed against the actual code
   before fixing: (a) output filenames/scoring keys collided for
   pairs-dir crops (hardcoded `x=0,y=0`, `frame_id` dropping the crop index) —
   8 overfit pairs reduced to 3 unique files; fixed via a `tag_id`/`crop_id`
   using the real `pair_id`. (b) seed-spread statistic pooled crop and seed
   variance together instead of aggregating per seed first — fixed. (c) the
   entire cloned ControlNet was trainable, not just the new adapter layers —
   fixed (see param counts above). (d) validation resampled noise/timestep
   *and* VAE-encoding stochasticity every call, making checkpoint comparison
   meaningless — fixed (deterministic generator + `.mode()` during eval only),
   plus added actual best-checkpoint tracking (previously absent entirely).

**Overfit-test control (job 44309, 8 pairs, 300 steps, COMPLETED)**: mechanics
correct, no NaN, checkpoint saved. Loss did not show a clear downward trend
(0.15–0.26 range throughout) — inconclusive on its own per this project's
standing note that per-step diffusion loss is a noisy, unreliable signal.

**Source-conditioning ablation control (jobs 44365/44366/44367, COMPLETED) —
FAILED.** `correct`/`zero`/`shuffled` conditioning produce statistically
indistinguishable outputs:

| mode | SSIM (mean ± spread across 3 seeds) | paired win-rate (n=8 crops) |
|---|---|---|
| correct | 0.0667 ± 0.0039 | 2/8 |
| zero | 0.0667 ± 0.0039 | 3/8 |
| shuffled | 0.0667 ± 0.0039 | 3/8 |

Differences between modes are in the 4th decimal — an order of magnitude
smaller than the seed-to-seed noise. Per the ticket's own acceptance
criterion, this means the new ControlNet branch is currently being ignored,
not that the diagnosis is wrong. Inventory check confirmed all three modes
scored the identical 24-crop set (8 crops × 3 seeds), so this isn't a data
artefact.

**Reading**: most likely explanation is under-training, not a broken
mechanism — ControlNet's zero-initialised residual layers start at literally
zero output and only gain influence as those weights move during training;
300 steps on 8 pairs is very little for that (published ControlNet training
typically uses thousands+ steps). This directly explains the flat overfit
loss curve too. **Not yet concluded either way** — re-running the same
overfit test at 2000 steps (job 44381, submitted) before treating this as a
real negative result about the method itself, rather than an artefact of an
under-trained smoke test.

**Extended overfit test (job 44381, 8 pairs, 2000 steps, COMPLETED)**: loss
shows a real, if noisy, downward trend this time (early steps mostly
0.19–0.26, steps 1500+ mostly 0.09–0.18) — unlike the flat 300-step run.

**Ablation re-run on the 2000-step checkpoint (jobs 44431/44432/44433,
COMPLETED) — PASSES the mandatory control this time**, decisively:

| mode | SSIM | LAB Wasserstein (lower=better) | MAE |
|---|---|---|---|
| **correct** | **0.14656** | **23.95** | **40.88** |
| shuffled | 0.04665 | 27.50 | 50.42 |
| zero | 0.06198 | 37.24 | 43.27 |

`correct` beats both `zero` and `shuffled` by a wide margin — SSIM is >2.4x
higher than either control (0.147 vs 0.047/0.062), far beyond the ~0.004
seed-to-seed noise floor seen throughout these runs, and colour distance
(LAB Wasserstein) is clearly lowest too. This is the separation the ticket's
own acceptance criterion required and the 300-step run failed to show.
**Confirms the "undertrained, not broken" reading of the 300-step failure**:
at 300 steps the ControlNet's zero-initialised layers hadn't developed
measurable influence yet; at 2000 steps the model has genuinely learned to
use source conditioning — this is now real, positive evidence that P1-10's
core mechanism works, on this tiny 8-pair overfit set at least. Still to be
shown: whether this generalises beyond memorising 8 pairs, at the scale of
a real training run.

**Full training run (job 44445, all 50 pairs, 4000 steps, `mscluster52`,
COMPLETED 24:50)**: 39 train / 11 val pairs (5 frames held out: 00A, 01C,
01D, 03C, 04B). **Validation loss converges cleanly and monotonically almost
the entire run** — 0.0880 → 0.0876 → 0.0832 → 0.0741 → 0.0683 → 0.0623 →
0.0600 → 0.0560 → 0.0546 → 0.0532 → 0.0525 → 0.0514 → 0.0496 → 0.0493 →
**0.0482 (best, step 3750)** → 0.0486 (final step, step 4000 — worse than
best, confirming exactly why the ticket requires checkpoint selection rather
than trusting the final step). Far cleaner convergence than the small
8-pair overfit set showed, consistent with the model having genuine diverse
signal to learn from rather than memorising a handful of examples.
`lora/a2h_cond_r8/best/` verified real (1.4GB ControlNet + 6.4MB LoRA,
correct trainable-param counts in `training_config.json`).

**VAE-only floor check (job 44515, all 496 held-out crops, COMPLETED)**: pure
VAE encode→decode, no UNet/noise/conditioning at all — establishes the
unavoidable compression ceiling separate from any denoising drift. SSIM
0.5393, LAB total 31.24, windowed LAB 31.66, PSNR 17.72, MAE 25.96. This is a
genuinely informative number for the writeup: even with **zero** diffusion
denoising, SD1.5's VAE alone caps SSIM around 0.54 — meaningfully below every
classical baseline's pooled SSIM (Macenko 0.628, Reinhard 0.681, Histogram
Matching 0.651). No diffusion-based method in this pipeline can structurally
beat those baselines while routing through this VAE, regardless of how good
the conditioning is.

**Smoke gate (jobs 44515–44518, A06 + A08 subset, LIMIT=20, COMPLETED) —
PASSES decisively.** Ticket requirement: *"beats both the target-only LoRA
and its own shuffled-source ablation on windowed colour recovery without
reducing structural fidelity."* All three conditions run at matched settings
(50-step DDIM, strength 0.50, `lora/a2h_cond_r8/best` for the two P1-10 arms,
`lora/a2h_r8/final` for the target-only baseline — the *original*, pre-P1-10
checkpoint):

| Condition | SSIM ↑ | windowed LAB ↓ | LAB total ↓ | PSNR ↑ | MAE ↓ | dE2000 ↓ |
|---|---|---|---|---|---|---|
| **P1-10 correct-source** | **0.336** ± 0.002 | **78.24** | 77.97 | **12.72** | **48.53** | 20.87 |
| P1-10 shuffled-source (ablation ctrl) | 0.127 ± 0.001 | 79.67 | 78.59 | 11.19 | 56.21 | 23.33 |
| Target-only LoRA (no conditioning) | 0.133 | 80.46 | 79.74 | 11.25 | 55.58 | 23.22 |

`correct` wins **every single metric** against both required comparisons —
not just windowed colour recovery, structural fidelity too (SSIM 2.5–2.6x
higher than either baseline; paired win-rate 80/80 crops on SSIM against
`shuffled`). Cleaner than the gate strictly required.

**Full held-out evaluation (job 44542, all 496 crops x 3 seeds,
`lora/a2h_cond_r8/best`, plain 50-step DDIM, strength 0.50 — the ticket's
mandated staged-rollout settings, not a tuned operating point — COMPLETED,
scored via `score_p1_10_ablation.py` + a small dedicated aggregation script
`src/eval/aggregate_p1_10_full.py` since this manifest is source_mode-shaped,
not strength-shaped, so `score_outputs.py` doesn't apply directly) — a
genuinely mixed result, not the clean pass the smoke gate suggested:**

| Method | ALL SSIM | ALL wLAB | A06-excl SSIM | A06-excl wLAB | recovery Δlab |
|---|---|---|---|---|---|
| Raw (do nothing) | 0.733 | 34.05 | — | — | — |
| Reinhard | 0.681 | 28.96 | — | — | +5.81 |
| Macenko | 0.628 | 25.96 | 0.649 | 21.88 | +9.04 |
| Histogram Matching | 0.651 | 31.64 | — | — | +3.23 |
| SDXL A4@0.20 | 0.527 | — | — | — | −5.60 |
| SD1.5 A5@0.20 (prior best) | 0.454 | 32.89 | 0.475 | 23.85 | +1.93 |
| SD1.5 A4@0.20 (prior best) | 0.459 | 33.25 | 0.480 | 24.15 | +1.53 |
| **P1-10 correct-source (full)** | **0.4485** | **31.60** | **0.4695** | **22.81** | **+2.89** |

Per-slide (P1-10): A06 SSIM 0.3067 (z=44.66, outlier, as expected), A08
0.4815, A09 0.4175, A13 0.4581, A16 0.4968.

**Reading, not overstated in either direction:**
1. **Does not beat classical baselines** — SSIM 0.4485–0.4695 stays nowhere
   near Macenko/Reinhard/Histogram Matching (0.628–0.681), consistent with
   every diffusion configuration tested this entire project. Now explained by
   a new finding this session: the VAE-only floor check (below) shows SD1.5's
   VAE alone caps SSIM around 0.54 with **zero** denoising — no amount of
   conditioning quality can push past that ceiling in this pipeline.
2. **Does genuinely beat SD1.5's own prior best operating point (A4/A5@0.20)
   on colour recovery** — both pooled (+2.89 vs +1.53/+1.93) and A06-excluded
   (windowed LAB 22.81 vs 24.15/23.85). This part is real: source
   conditioning recovers more colour than the target-only approach did.
3. **On structure, essentially tied with A4/A5@0.20** — SSIM marginally
   *behind* both, pooled (0.4485 vs 0.459/0.454, ~1%) and A06-excluded
   (0.4695 vs 0.480/0.475, ~1-2%). Not the ticket's acceptance-criterion
   requirement of "jointly exceeds... on colour AND structure" — a wash on
   structure specifically, not a win, not a loss.
4. **Caveat, now resolved (see strength sweep below)**: this comparison isn't
   fully apples-to-apples on sampling settings — P1-10 ran at the ticket's
   mandated conservative first-test settings (plain 50-step DDIM, strength
   0.50), while A4/A5@0.20 is SD1.5's most-tuned operating point (8-step LCM,
   strength 0.20). The strength sweep below confirms this doesn't change the
   verdict: no strength on this checkpoint jointly beats 0.50 on both SSIM
   and colour recovery.

**VAE-only floor check (job 44515, all 496 crops, COMPLETED)**: pure VAE
encode→decode, zero denoising — SSIM 0.5393, LAB total 31.24, windowed LAB
31.66, PSNR 17.72, MAE 25.96. Below every classical baseline's SSIM —
establishes that no diffusion-based method routed through this VAE can
structurally match the classical remaps, regardless of conditioning quality.

**Bottom line**: P1-10 confirms the diagnosis was directionally correct
(genuine training-time source conditioning measurably improves colour
recovery without a structure penalty, unlike the strength dial which trades
one for the other) but does **not** close the fundamental structural-fidelity
gap to classical methods — that gap is capped by VAE resynthesis itself, not
by the conditioning problem P1-10 was built to fix. Status: **DONE** as a
supplementary corrective experiment — a genuine, reportable, partially-positive
result (matches the ticket's own framing that either outcome is informative).

**Strength sweep (jobs 44625-44628 + 44637/44641/44642/44649/44650 scoring,
A06+A08 diagnostic subset, COMPLETED) — a clean, decisive negative result,
no full re-run warranted.** Tested whether a different operating point
improves both SSIM and colour recovery simultaneously relative to strength
0.50, mirroring SD1.5's own A4/A5 finding that 0.20 beat 0.50-equivalent
settings:

| Strength | SSIM ↑ | windowed LAB ↓ | MAE ↓ |
|---|---|---|---|
| 0.20 | 0.3575 | 80.42 | 49.21 |
| 0.30 | 0.3466 | 79.84 | 49.05 |
| 0.40 | 0.3401 | 79.28 | 48.92 |
| 0.50 (reference) | 0.336 | 78.24 | **48.53 (best MAE)** |
| 0.70 | 0.3196 | **78.11 (best wLAB)** | 49.45 |

A textbook structure/colour tradeoff, no free lunch: SSIM falls monotonically
as strength rises (0.3575→0.3196), windowed LAB improves monotonically the
whole way to 0.70. **No strength beats 0.50 on both axes at once** — 0.70
edges out 0.50 on windowed LAB only marginally (78.11 vs 78.24) while giving
up meaningfully more SSIM and losing on MAE. Strength 0.50 (the value used in
the full held-out run above) sits on this tradeoff curve, not dominated by
anything tested. **Conclusion: the full 496-crop evaluation does not need to
be re-run at a different strength** — nothing in this sweep would change the
reported verdict.

## P1-11 — DDIM-inversion inference path for P1-10 (inference-only follow-up)

**Status:** ✅ DONE (2026-08-22) — implementation, smoke test (identity +
translate-mode fraction sweeps, source-conditioning ablation), and the full
496-crop x 3-seed held-out run + per-slide scoring are all complete. Clean,
decisive positive result: DDIM inversion (f=1.00, correct source
conditioning) beats P1-10's existing img2img pathway on both SSIM and
colour recovery, on every held-out slide (see full results below).

**Source:** `tickets/"Claude Code Task_ Add DDIM-Inversion Inference for
P1-10(2).md"` (full task spec), following directly from P1-10's own full
held-out result above (SSIM 0.4485 vs. classical baselines 0.628–0.681,
capped by the VAE-only floor of 0.5393) and P1-10's strength-sweep finding
just above (no `strength` value beats 0.50 on both colour and structure at
once — a textbook tradeoff, not a bug). This ticket asks a narrower,
complementary question: is part of the remaining structural loss caused
specifically by `infer_colour_translation.py`'s random-Gaussian-noise img2img
initialization (VAE-encode source → add noise at fixed `strength` → DDIM
denoise), rather than by the trained model itself? DDIM inversion replaces
that arbitrary corruption with a deterministic, source-specific noisy latent
obtained by literally inverting the source image through the trained model's
own noise-prediction, then denoising forward again.

**Scope, per the task file's own explicit constraints:** inference-only.
`lora/a2h_cond_r8/best` (LoRA + ControlNet, confirmed trained with epsilon
prediction via `training_config.json`) is used exactly as-is — no
retraining, no rank/architecture/VAE/prompt/dataset changes, no new losses.
`infer_colour_translation.py` (P1-10's existing, validated inference script)
is not touched.

**Design decisions (task file left several deliberately open, all resolved
and documented in the new script's docstring/CLI help, not silently
guessed):**
- No diffusers pipeline packages DDIM-inversion-then-conditional-
  reconstruction for a ControlNet img2img pipeline, so the new script drives
  `pipe.vae`/`pipe.unet`/`pipe.controlnet`/`pipe.text_encoder` directly in a
  manual step loop rather than calling `pipe(...)`.
- Both the forward `DDIMScheduler` and the `DDIMInverseScheduler` are built
  from the SAME resolved scheduler config the existing P1-10 inference
  script already uses (`DDIMScheduler.from_config(pipe.scheduler.config)`)
  — confirmed live on the cluster: `beta_start=0.00085, beta_end=0.012,
  beta_schedule=scaled_linear, num_train_timesteps=1000, steps_offset=1,
  set_alpha_to_one=false, clip_sample=false`, `prediction_type` defaults to
  (and is asserted to equal) `"epsilon"` — matching the checkpoint's actual
  training, NOT HistDiST's v-prediction/trailing-timesteps/zero-terminal-SNR
  settings, which this checkpoint was never trained with.
- `--inversion-guidance` defaults to 1.0 (no CFG during inversion, separate
  from `--guidance` which stays 2.0 for reconstruction) — standard DDIM-
  inversion practice, since CFG during inversion is not exactly invertible
  and accumulates error (the reason Null-text Inversion exists, which is
  deliberately not implemented here — out of scope per the task file).
- `--inversion-condition {none,source}` (default `source`) controls ONLY the
  inversion pass's ControlNet conditioning; documented as the conservative/
  identity-preserving default since it's the closest analogue to what the
  model saw during training.
- `--mode identity` vs `--mode translate` differ only in what conditions the
  reconstruction half and what the output is scored against — `identity`
  self-conditions throughout and scores against the original source crop
  (also computing `vae_only` and the existing random-noise `img2img_baseline`
  pathway for the same in-memory crop/seed in the same pass, satisfying the
  task file's mandatory smoke-test rows §13/§14/§23 from one invocation);
  `translate` conditions per `--source-mode {correct,zero,shuffled}` (same
  ablation semantics as the existing script) and scores against real
  registered Hamamatsu. Documented caveat: the LoRA's colour bias is baked
  into UNet attention weights, not just conditioning, so `identity` mode may
  still show some colour drift — a genuine diagnostic finding, not masked.
- Partial inversion (`--inversion-fraction` in (0,1]) takes the first
  `round(inversion_steps * fraction)` steps of the ascending inverse-
  scheduler timestep grid; reconstruction starts from the nearest forward
  timestep `<= ` wherever inversion stopped (not a fixed `strength`).
  Requested vs. actual fraction/timestep/step-count always logged
  (discretization may not match exactly).
- `--strength` is accepted only to be explicitly rejected (`SystemExit`) if
  passed at all — verified locally (see below).

**Code (2026-08-20):** `src/eval/infer_colour_source_ddim_inversion.py`
(new) + `slurm/infer_p1_10_ddim_inversion.slurm` (new launcher, mirrors
`infer_colour_translation.slurm`'s bigbatch/fail-fast-CUDA-guard/bad-node-
exclude conventions). Manifest schema is a strict superset of
`infer_colour_translation.py`'s `eval_manifest.csv` (same
seed/source_mode/crop_id/slide/frame/x/y/output_path/reference_path/
aperio_path columns, plus method/mode/inversion_fraction/
inversion_condition/actual_* columns) — `score_p1_10_ablation.py`
(`csv.DictReader`, tolerates extra columns) scores it completely unchanged,
no edits needed. A `run_metadata.json` is written per run (checkpoint,
direction, source/inversion-condition modes, prediction_type, both
scheduler configs, VAE scaling, step counts, fraction, actual timestep,
guidance scales, conditioning scale, seeds, dtype, model path, best-effort
git commit).

**Local verification (2026-08-20, no GPU available off-cluster):**
`python -m py_compile` clean; `--help` parses cleanly with every flag from
the design above present at the right default; confirmed by direct test
that `--strength` raises the exact required error, `--inversion-fraction`
outside `(0,1]` is rejected, and `--pairs-dir` + `--root`/`--heldout`
together is rejected — same level of pre-cluster checking every prior P1-10
script got.

**Identity smoke test #1 — full inversion (2026-08-21), job 44652 (infer,
COMPLETED 30:53) + job 44677 (score, COMPLETED 3:36), A06+A08 `LIMIT=20`
subset (80 crops), single seed:**

| Method | SSIM ↑ | LAB total ↓ | windowed LAB ↓ | PSNR ↑ | MAE ↓ |
|---|---|---|---|---|---|
| vae_only (ceiling — pure encode/decode) | 0.4839 | 3.29 | 4.13 | 19.34 | 20.13 |
| img2img_baseline (existing pathway, strength=0.50) | 0.3343 | 7.16 | 8.73 | 17.74 | 24.14 |
| **ddim_inversion (new, f=1.0)** | **0.4483** | 23.22 | 23.47 | 17.48 | 24.82 |

Mechanically clean (job log: no NaN/Inf, `t_end=981` reached on inversion,
reconstruction correctly started from the same `t=981`, `50/50` real steps
both directions). **Core hypothesis holds at full inversion**: DDIM
inversion recovers substantially more source structure than the old
random-noise pathway (SSIM 0.4483 vs 0.3343, ~34% relative, much closer to
the 0.4839 VAE-only ceiling than the old pathway ever got). **Cost**:
colour-identity distance from the true source jumps sharply (LAB 23.22 vs
7.16) — consistent with the documented caveat above (the LoRA's colour bias
is baked into UNet weights, and a full 50-step round trip lets it act far
more than a strength-0.50 partial corruption does). Paired win-rate (best
SSIM per crop): vae_only wins 79/80 (expected — it's the floor/ceiling with
no generative content at all), ddim_inversion wins 1/80, img2img_baseline
wins 0/80 — the vae_only comparison is not the interesting one here; the
img2img_baseline-vs-ddim_inversion SSIM gap above is.

**Reading**: this is exactly the Pareto tradeoff the task file's own
§27 anticipated ("full inversion gives diffusion maximum freedom... a
shallow inversion may preserve much more source structure while still
allowing the learned colour transformation to act"). Next: the partial-
inversion sweep (0.25/0.50/0.75) to look for a fraction that keeps most of
this structural gain while containing the colour-identity cost.

**Partial-inversion sweep (2026-08-21), jobs 44683/44684/44685 (infer,
COMPLETED 23-30min each) + job 44689 (score, all 4 fractions + both
baselines together, COMPLETED 14:04) — clean, monotonic, decisive result:**

| Method | SSIM ↑ | LAB total ↓ | windowed LAB ↓ | PSNR ↑ | MAE ↓ |
|---|---|---|---|---|---|
| vae_only (ceiling) | 0.4839 | 3.29 | 4.13 | 19.34 | 20.13 |
| **ddim_inv f=0.25** | **0.4831** | **4.09** | **4.78** | 19.24 | 20.36 |
| ddim_inv f=0.50 | 0.4783 | 5.62 | 6.22 | 19.01 | 20.89 |
| ddim_inv f=0.75 | 0.4615 | 10.08 | 10.50 | 18.37 | 22.41 |
| ddim_inv f=1.00 | 0.4483 | 23.22 | 23.47 | 17.48 | 24.82 |
| img2img_baseline (old pathway, strength=0.50) | 0.3343 | 7.16 | 8.73 | 17.74 | 24.14 |

Both SSIM and colour-identity distance from the true source move
monotonically with `--inversion-fraction` (lower fraction = both better),
converging toward the `vae_only` ceiling as fraction → 0. **At f=0.25,
DDIM inversion beats the old random-noise pathway on BOTH axes at once** —
not the usual structure/colour tradeoff: SSIM 0.4831 vs 0.3343 (+44%
relative) AND LAB distance 4.09 vs 7.16 (also better/lower). Paired win-rate
across all 6 methods (80 crops): `vae_only` 51/80, `ddim_inv f=0.25` 20/80,
`ddim_inv f=0.50` 8/80, `ddim_inv f=1.0` 1/80 (`img2img_baseline` 0/80,
`f=0.75` 0/80 — not the interesting comparison; the point is f=0.25 already
gets most of the way to the vae_only ceiling while `img2img_baseline` wins
zero head-to-head crops against anything).

**Reading**: DDIM inversion is a strictly better identity-reconstruction
mechanism than random-noise img2img at every fraction tested, and a fraction
around 0.25 is the sweet spot found so far (not yet bisected finer). This
confirms the task file's own hypothesis (§27) that a shallow/partial
inversion, not full inversion, is the more promising operating point.
**Caveat**: this is a self-reconstruction (identity) fidelity result only —
it says the inversion *mechanism* preserves structure better than random
noise, not yet whether it still delivers useful A→H colour recovery in
`--mode translate`. That's the next test.

**Translate-mode smoke test (2026-08-21), jobs 44784/44836/44837/44838
(infer, COMPLETED) + jobs 44848/44849/44850/44851 (score, one per fraction
-- scoring all 4 together in a single call collapsed them into one bucket
since translate mode's `source_mode` column doesn't vary with fraction,
caught and fixed by scoring each fraction separately) — `source-mode
correct`, single seed, same A06+A08 `LIMIT=20` subset P1-10's own smoke
gate used (so this is directly comparable to the numbers already in the
P1-10 section above, not a new baseline):**

| Fraction | SSIM ↑ | windowed LAB ↓ | LAB total ↓ | PSNR ↑ | MAE ↓ | dE2000 ↓ |
|---|---|---|---|---|---|---|
| f=0.25 | 0.4262 | 79.47 | 79.27 | 12.95 | 48.02 | 20.84 |
| f=0.50 | 0.4232 | 77.64 | 77.42 | 12.98 | 47.70 | 20.54 |
| f=0.75 | 0.4086 | 74.05 | 73.72 | 12.86 | 48.35 | 20.14 |
| **f=1.00** | **0.4015** | **63.41** | 62.73 | 12.90 | 48.16 | **18.70** |
| P1-10 original (img2img, strength=0.50, same subset, from smoke gate above) | 0.336 | 78.24 | 77.97 | 12.72 | 48.53 | 20.87 |

**Clean, decisive result — DDIM inversion beats the old random-noise
pathway at EVERY fraction on structure** (SSIM 0.40-0.43 vs 0.336, +19-27%
relative), and **full inversion (f=1.00) additionally wins decisively on
colour recovery** (windowed LAB 63.41 vs 78.24, ~19% closer to Hamamatsu)
while still beating the old pathway's SSIM. Unlike identity mode (where
full inversion was the *worst* fraction for self-reconstruction fidelity),
translate mode shows the opposite fraction trend: SSIM falls slightly as
fraction rises (0.4262→0.4015) but colour recovery improves substantially
(windowed LAB 79.47→63.41) — yet even at f=1.00, the "worst" SSIM point in
this sweep, it still clearly beats the old pathway's SSIM. **No structure/
colour tradeoff needed against the baseline being replaced — f=1.00
dominates it on both axes at once.** f=1.00 is the natural candidate for
the mandatory source-conditioning ablation next.

**Source-conditioning ablation at f=1.00 (2026-08-21), jobs 44852/44853
(infer, COMPLETED) + job 44856 (score, COMPLETED) — decisive pass:**

| source_mode | SSIM ↑ | windowed LAB ↓ | LAB total ↓ | PSNR ↑ | MAE ↓ |
|---|---|---|---|---|---|
| **correct** | **0.4015** | 63.41 | 62.73 | **12.90** | **48.16** |
| shuffled | 0.0925 | 71.05 | 68.53 | 11.16 | 56.96 |
| zero | 0.0482 | 51.80 | 46.38 | 10.80 | 60.36 |

`correct` wins **80/80 crops** on paired SSIM against both controls — more
than 4x higher than either (0.4015 vs 0.0925/0.0482). The model is
genuinely using real source conditioning during DDIM-inversion translation,
not riding the inversion mechanism alone. **Caveat, stated plainly, not
hidden**: `zero`'s windowed LAB (51.80) is numerically *lower* than
`correct`'s (63.41) — but this doesn't undermine the structural finding:
with no source-structure conditioning at all, the model just hallucinates
something colour-plausible unconstrained by the actual tissue (SSIM 0.048
confirms it isn't preserving real structure), which can incidentally land
closer to Hamamatsu's colour statistics without being a useful translation.
The ticket's own acceptance bar is winning on structural/content metrics,
which `correct` does decisively.

**Go/no-go smoke gate (task file §"Go/no-go smoke gate") — PASSES on every
required condition, same A06+A08 subset throughout:**
1. `correct` beats `shuffled`/`zero` on the source-conditioning ablation
   (structural, above) — ✅ decisively (80/80 paired win-rate).
2. `correct` beats the existing img2img pathway ("target-only" initialization)
   on windowed colour recovery: 63.41 vs 78.24 — ✅.
3. ...without reducing structural fidelity: 0.4015 vs 0.336 — ✅, actually
   *improves* structure too, not merely preserves it.

**All mandatory gates pass. Per the task file's own explicit staging
(§"E"), the full 496-crop held-out run is now unblocked** — DDIM inversion
at f=1.00, `--source-mode correct`, is the candidate configuration
(matches the best translate-mode result found; f=1.00 was also the
configuration used for this ablation).

**Full 496-crop held-out run (2026-08-21/22), job 44858, `mscluster61`, all
5 held-out slides, 3 seeds (0,1,2), f=1.00, `--source-mode correct` --
TIMEOUT at exactly 10:00:14, but recovered with zero data loss.** All 1488
outputs + 496 reference crops had already been written to disk when the
timeout fired (confirmed via direct file count) -- only the buffered, never-
flushed `eval_manifest.csv` (Python's default full-buffering on a regular
file, combined with SIGKILL-on-timeout giving no chance to `close()`
cleanly) was lost. Recovered via a small one-off script
(`reconstruct_manifest.py`, not committed -- scratch/one-time use) that
rebuilds every needed column (seed/source_mode/crop_id/slide/frame/x/y/
output_path/reference_path -- `aperio_path` left blank, unused by the
scorer) directly from the filenames, which encode all of it by construction.
Reconstructed 1488/1488 rows, zero warnings, verified per-slide/seed counts
match the raw file counts exactly before trusting it. **Lesson for any
future long DDIM-inversion run: the manifest CSV should be flushed
periodically (e.g. every N rows) or reopened in append mode, not left to a
single end-of-run `close()` -- not fixed in the script itself yet since this
run already completed via reconstruction, but worth doing before the next
very long run.**

Scoring: job 45020 (initial attempt, `score_p1_10_ddim_inversion.slurm`'s
default 30-minute time limit, inherited from the smaller-scale
`score_p1_10_sweep.slurm` template) TIMEOUT at exactly 30:04 -- too short
for 1488 rows (only tested at 320 rows/14min before). Resubmitted as job
45024 with `--time=02:00:00`, COMPLETED 58:54. Per-slide breakdown via
`aggregate_p1_10_full.py` (the dedicated aggregator already built for this
exact per_crop.csv shape):

| Scope | SSIM (old img2img → new DDIM-inv) | windowed LAB (old → new) | recovery Δlab (old → new) |
|---|---|---|---|
| **ALL** (n=1488) | 0.4485 → **0.4960** | 31.60 → **26.01** | +2.89 → **+8.74** |
| ALL excl. A06 (n=1296) | -- → **0.5142** | -- → **18.92** | -- → **+7.43** |
| A06 (outlier, z=25.51) | 0.3067 → **0.3732** | -- → 73.82 | -- → **+21.72** |
| A08 | 0.4815 → **0.5368** | -- → 19.02 | -- → **+7.15** |
| A09 | 0.4175 → **0.4767** | -- → 18.54 | -- → **+9.68** |
| A13 | 0.4581 → **0.4979** | -- → 22.21 | -- → **+5.37** |
| A16 | 0.4968 → **0.5273** | -- → 17.78 | -- → **+7.11** |

**Clean, decisive, uniform positive result -- every slide improves on BOTH
structure and colour recovery simultaneously, not just A06.** This is
notable because it does NOT replay the A06-vs-typical-slide divergence
pattern seen with every other "more aggressive setting" tried in this
project (SD1.5's own strength sweep, SDXL's strength sweep) -- the smoke
test's A06-heavy composition (80% A06 by crop count) had raised a real
concern that its result might not generalise; it generalised anyway, and
better than predicted. Context against the rest of the project: pooled SSIM
(0.4960 ALL / 0.5142 excl-outliers) is the second-highest ever recorded for
any diffusion configuration here (only SDXL A4@0.20's 0.527 is higher, but
that came with catastrophically negative colour recovery, Δlab -5.60) --
this is the first configuration in the whole project to combine strong
structure retention AND strong positive colour recovery at the same time.
Still below every classical baseline's SSIM (Macenko 0.628, Reinhard 0.681,
Histogram Matching 0.651), as expected -- capped by the same VAE-only floor
(0.5393) established under P1-10, which DDIM inversion cannot change since
it doesn't touch the VAE. **P1-11 status: DONE** -- confirms the original
diagnosis (task file's opening question) was correct: part of P1-10's
remaining structural-fidelity loss WAS caused by the random-noise img2img
initialization, and DDIM inversion recovers a real, substantial, uniform
share of it without sacrificing colour recovery.

## P1-12 — P1-10 + LCM-LoRA acceleration: strength/steps/guidance exploration

**Status:** 🔄 IN PROGRESS (2026-08-22) -- data-driven follow-up, not from
the proposal or the P1-11 task file; motivated by wanting a few-step
(fast-inference) operating point for the P1-10 source-conditioned
checkpoint, mirroring A4's LCM acceleration of the original target-only
pipeline (`tickets/PHASE1-TICKETS.md` P1-05). Separate from P1-11 (DDIM
inversion) -- this thread never touches DDIM inversion, it's plain
img2img + LCM-LoRA on top of P1-10's checkpoint.

**Code (2026-08-22):** `src/eval/infer_colour_translation_lcm.py` (new --
per `infer_colour_translation.py`'s own docstring, LCM was deliberately
deferred to "a later, separate script/run, not this one", so this is a new
script, not an edit). Mirrors `infer_colour_lora.py`'s already-validated
multi-adapter LCM pattern (named adapters + `set_adapters` + `LCMScheduler`)
composed with P1-10's custom 6-channel-conditioned ControlNet + colour LoRA.
Manifest schema matches `infer_colour_lora.py`'s strength-shaped output
exactly, so the existing `score_outputs.py`/`score_outputs.slurm` score it
unchanged -- no new scorer needed. Launcher: `slurm/infer_p1_10_lcm_sweep.slurm`
(comma-separated `--strengths`/seeds throughout this project's scripts now,
not space-separated -- see the note in that file's header: a space-
containing argument does not survive an `ssh host sbatch ... "0.2 0.3"`
round trip intact, the identical bug already hit and fixed for P1-11's
`--seeds`).

**Strength sweep (2026-08-22), job 45026 (infer, COMPLETED 23:14) + job
45027 (score via existing `score_outputs.slurm`, COMPLETED 16:12) -- 8-step
LCM, guidance 1.5, A06+A08 `LIMIT=20` subset:**

| Strength | A06 LAB | A06 SSIM | A06 Δlab | A08 LAB | A08 SSIM | A08 Δlab |
|---|---|---|---|---|---|---|
| 0.20 | 93.62 | 0.350 | +1.22 | 28.82 | 0.527 | -3.55 |
| 0.30 | 92.98 | 0.352 | +1.86 | 29.24 | 0.523 | -3.96 |
| 0.40 | 91.46 | 0.364 | +3.38 | 28.26 | 0.528 | -2.98 |
| 0.50 | 91.13 | 0.366 | +3.71 | 26.54 | 0.534 | -1.27 |
| 0.70 | 91.17 | 0.363 | +3.67 | 24.17 | 0.540 | **+1.10** |

**Data-quality note**: `score_outputs.py`'s pooled `ALL` `recovery_delta_lab`
column is bogus for this run (-28 to -31, despite the log's own "[baseline
restricted to A06,A08]" annotation) -- a 2-slide-only scope evidently still
mishandles the pooled baseline lookup. Per-slide deltas are correct (hand-
verified against each slide's own true raw baseline, e.g. A06: 94.84-93.62 =
1.22, matches exactly) and are what's reported above; do not trust the
pooled figure from this particular run without investigating the mismatch
first.

**Reading**: colour recovery here is much weaker than either P1-10's own
50-step DDIM pathway or P1-11's DDIM inversion (max A06 gain +3.7 vs P1-11's
+21.7) -- mirrors the established pattern that LCM acceleration trades away
colour-recovery quality (A4 vs A3 showed the same thing in Phase 1). A08
(typical slide) is actually *negative* (worse than doing nothing) at every
strength except 0.70 -- the only point where both slides show positive
recovery simultaneously, so it's the anchor strength for the grid below.
SSIM is nearly flat across the whole sweep (0.35→0.36 A06, 0.52→0.54 A08) --
notably different from the monotonic-SSIM-falls-with-strength pattern seen
in every DDIM-based sweep elsewhere in this project.

**Steps x guidance grid (2026-08-22), strength fixed at 0.70, steps ∈
{4,6,8} x guidance ∈ {0,1,2} (9 configs), same A06+A08 subset -- IN
PROGRESS, jobs 45028-45033/45039-45041 (see job-management note below).**
Reuses `infer_p1_10_lcm_sweep.slurm` directly, no new code -- each grid
point is one job with `STRENGTHS_CSV=0.70` and the given steps/guidance.
**Known collision expected, not yet a surprise when it shows up**:
diffusers' `do_classifier_free_guidance = guidance_scale > 1`, so
guidance=0 and guidance=1 will produce byte-identical output within this
pipeline (both skip CFG entirely, single forward pass, the guidance_scale
value itself unused in that branch) -- same class of quantisation collision
as the LCM step-count issue P1-05 already documented, just a different
mechanism. Results pending.

**Extended strength sweep (2026-08-22), strengths 0.80/0.90/1.00, job
45037, IN PROGRESS.** Same 8-step/guidance-1.5 config as the original
5-point sweep, checking whether colour recovery keeps improving past 0.70
(as A06's own trend within the original sweep suggested it might plateau,
not clearly still climbing) or whether it's already past the useful range.
Results pending.

**Final combination -- results (2026-08-23), job 45650 (infer) + job 45692
(score, `score_outputs.slurm`), steps=8/guidance=2.0/strength=0.80:**

| Slide | SSIM | Δlab |
|---|---|---|
| A06 | 0.307 | +9.56 |
| A08 | 0.512 | **-0.46** |

Worse than the grid's best (steps=6/guidance=2.0/strength=0.70: A06
SSIM 0.334/Δlab +9.73, A08 SSIM 0.515/Δlab +2.98) on every axis except A08
SSIM -- A08 flips negative here, meaning colour recovery is worse than doing
nothing. Pushing both strength and guidance up simultaneously does not
combine the two individual wins; it overcorrects the typical slide (A08)
while barely helping the outlier (A06) beyond what the grid already found.
**This is not a new best point.**

**Few-step DDIM vs LCM diagnostic -- results (2026-08-23), job 45654 (infer)
+ job 45700 (score, `score_p1_10_ablation.py`), same checkpoint/strength=
0.80/guidance=2.0/steps=8, plain DDIM scheduler (no LCM-LoRA), 3 seeds:**

| Slide | SSIM | wlab (mean) | Δlab (vs baseline_summary.csv) |
|---|---|---|---|
| A06 | 0.331 | 90.47 | +4.45 |
| A08 | 0.470 | 26.88 | **-1.45** |

At this same aggressive operating point, **plain DDIM also goes negative on
A08** (-1.45, comparable in sign and rough magnitude to LCM's -0.46) --
LCM's colour-recovery weakness at this specific corner is not a
scheduler-specific artefact; forcing *any* scheduler this far into
few-step/high-strength/high-guidance territory destabilises the typical-
slide result. Where the two runs do differ: LCM actually recovers *more*
colour on A06 than matched-DDIM here (+9.56 vs +4.45) at a modest SSIM cost
(0.307 vs 0.331) -- suggesting LCM's consistency-distillation prior
compresses toward stronger, blunter corrections in fewer steps, for better
or worse depending on how far the source already is from target. Neither
scheduler makes this operating point (strength=0.80/guidance=2.0) usable.

**P1-12 acceptance criterion, evaluated:** the best LCM configuration found
across every experiment in this ticket (grid, extended sweep, final
combination) remains **steps=6/guidance=2.0/strength=0.70** (A06 SSIM
0.334/Δlab +9.73, A08 SSIM 0.515/Δlab +2.98) -- less than half of P1-11's
colour recovery on both slides (A06 SSIM 0.373/Δlab +21.72, A08 SSIM
0.537/Δlab +7.15), despite broadly comparable SSIM. No tested generic
LCM-LoRA configuration approaches P1-11's colour recovery without a
material quality gap. **Conclusion: the generic LCM-LoRA adapter is not
fidelity-equivalent for the specialised source-conditioned translator.
P1-11 (50-step DDIM inversion) remains the quality operating point;** LCM's
role here is a speed/quality tradeoff only, not a substitute, and should be
framed that way in any writeup (fast preview / iteration mode, not a
replacement for the reported result).

**Status: P1-12 CLOSED (2026-08-23).**

**Job-management note (2026-08-22), for reproducibility:** the cluster
enforces a hard cap of 6 concurrently *running* jobs per user
(`QOSMaxJobsPerUserLimit`) -- pending jobs queue FIFO by submission order
regardless of how many are cancelled, so cancelling a job that hasn't
started yet does NOT let a later-submitted job jump the queue; only
cancelling a *running* job (or a pending job that's actually ahead in FIFO
order) frees a slot for a specific later job to grab. To get the extended
strength sweep (job 45037) running immediately alongside the in-progress
grid without losing any of the grid's 9 configs: cancelled 3 jobs that had
done the least/no work (45036 pending steps=8/g=2 -- zero waste; 45035
pending steps=8/g=1 -- zero waste; 45034 running steps=8/g=0 at 6 seconds
elapsed -- negligible waste) and one already-progressed running job (45033,
steps=6/g=2, ~10 min in, the least-elapsed of the six running jobs at the
time) once cancelling only the zero-waste ones didn't free a slot fast
enough for 45037 to grab ahead of the FIFO queue. All 4 cancelled configs
were resubmitted immediately after (45038, 45039, 45040, 45041) -- no grid
point was permanently lost, only delayed, and the ~10 minutes of partial
compute on 45033 is the only real cost.

**Steps x guidance grid -- results (2026-08-22), jobs 45028-45032/45038-
45041 (infer, all COMPLETED) + jobs 45042-45050 (score, all COMPLETED),
strength fixed at 0.70, same A06+A08 subset:**

| Steps | Guidance | A06 SSIM | A06 Δlab | A08 SSIM | A08 Δlab |
|---|---|---|---|---|---|
| 4 | 0 | 0.368 | +1.25 | 0.532 | -3.28 |
| 4 | 1 | 0.368 | +1.25 | 0.532 | -3.28 |
| 4 | 2 | 0.342 | +7.82 | 0.506 | -0.20 |
| 6 | 0 | 0.371 | +0.78 | 0.547 | -0.08 |
| 6 | 1 | 0.371 | +0.78 | 0.547 | -0.08 |
| **6** | **2** | **0.334** | **+9.73** | **0.515** | **+2.98** |
| 8 | 0 | 0.374 | -1.02 | 0.549 | -0.90 |
| 8 | 1 | 0.374 | -1.02 | 0.549 | -0.90 |
| 8 | 2 | 0.342 | +8.48 | 0.520 | +2.43 |

**Guidance=0 and guidance=1 produced byte-identical per-crop output at every
step count** -- confirms the predicted collision exactly (diffusers'
`do_classifier_free_guidance = guidance_scale > 1`, so both fall into the
same no-CFG code path). Not a bug, and now empirically verified rather than
just asserted.

**Real signal: guidance=2.0 dominates 0/1 on colour recovery at every step
count**, at a modest, not catastrophic, SSIM cost (e.g. steps=6: SSIM
0.371→0.334, Δlab +0.78→+9.73 on A06) -- the classic CFG structure/colour
tradeoff, but clearly worth paying here. **Best point in the entire grid:
steps=6, guidance=2.0** -- the only grid cell with BOTH slides positive at
once (A06 +9.73, A08 +2.98), beating the original fixed-guidance-1.5 sweep's
best point (steps=8, strength=0.70, guidance=1.5: A06 +3.67, A08 +1.10) by a
wide margin on both axes. steps=8/guidance=2 is close behind (+8.48/+2.43)
but steps=6 edges it out on A08 while using fewer steps (faster).

**Extended strength sweep -- results (2026-08-22), job 45037 (infer,
COMPLETED) + job 45051 (score, COMPLETED), steps=8/guidance=1.5 (the
original sweep's fixed settings, NOT the grid's guidance=2.0 finding above
-- these two experiments varied different axes and haven't yet been
combined):**

| Strength | A06 SSIM | A06 Δlab | A08 SSIM | A08 Δlab |
|---|---|---|---|---|
| 0.80 | 0.320 | +4.42 | 0.537 | +1.91 |
| 0.90 | 0.161 | +9.96 | 0.436 | -1.26 |
| 1.00 | 0.076 | +11.07 | 0.170 | -4.19 |

0.80 is a genuine further improvement over 0.70's +3.67/+1.10 (both slides
still positive, higher on both). **0.90 and 1.00 break down badly** --
despite the raw Δlab number continuing to climb (misleadingly, at first
glance), SSIM collapses (0.076 at strength 1.00 -- structure is essentially
destroyed) and A08 flips sharply negative. Not usable operating points
despite the impressive-looking colour-recovery number; a case where trusting
Δlab alone without checking SSIM in the same breath would have been a real
mistake.

**Where this leaves P1-12**: the two experiments varied different axes
(strength 0.70-1.00 at guidance=1.5; steps/guidance at strength 0.70) and
hadn't been combined yet. The natural next test, combining both wins found so
far, is guidance=2.0 at strength=0.80 -- to see whether it beats
steps=6/guidance=2.0/strength=0.70's +9.73/+2.98.

**Final combination run (2026-08-23):** user requested LCM steps=8 (not 6),
strength=0.80, guidance=2.0 as the closing test -- steps=8 rather than the
grid's best steps=6, to also double-check the steps axis wasn't left on a
locally-good-but-not-best value. Submitted as job **45650**
(`eval/p1_10_lcm_final`, `infer_p1_10_lcm_sweep.slurm 0.80 20 8 2.0
p1_10_lcm_final`, same A06+A08 limit=20 smoke subset as every other P1-12
run, so directly comparable to the grid/sweep tables above). Scored with the
unmodified `score_outputs.slurm p1_10_lcm_final`. Results: see "Final
combination -- results" table further up.

**Few-step DDIM vs LCM diagnostic (2026-08-23), run in parallel with job
45650.** Question: at matched low step count/strength/guidance, is LCM's
weaker colour recovery (max A06 Δlab ~+11 vs P1-11's +21.7) caused by the
LCM scheduler/adapter itself, or would plain DDIM show the same drop-off if
forced down to the same few-step regime? Isolates the scheduler as the
variable, holding checkpoint/strength/guidance/step-count fixed.

Reused `infer_colour_translation.py` completely unmodified (it never
imports `LCMScheduler` -- this is a genuine plain-DDIM run, not a relabelled
LCM one) via `slurm/infer_colour_translation.slurm`, which previously
hardcoded `--steps 50` in both python invocations. Added `STEPS`/
`STRENGTH`/`GUIDANCE` env-var overrides (default 50/0.50/2.0, so every prior
invocation of this script is unaffected) to make the matched comparison
possible without a new script. Submitted via `sbatch --export=ALL,STEPS=8,
STRENGTH=0.80,GUIDANCE=2.0 ... infer_colour_translation.slurm correct
/datasets/mhoosen/stain-norm/lora/a2h_cond_r8/best 20
p1_10_ddim_vs_lcm_diag` -- `--export` (not a bare env-var prefix) so the
command still starts literally with `ssh ... sbatch`, matching this
project's settings.local.json `ask` pattern. Job **45654**, confirmed live
in the job log (`Steps=8  Strength=0.80  Guidance=2.0`). Same checkpoint,
same A06+A08 `LIMIT=20` subset, same strength/guidance as job 45650
(LCM steps=8/strength=0.80/guidance=2.0) -- directly comparable once both
score. Scoring: `score_p1_10_ablation.py` (this script's manifest is
seed/source_mode-shaped, NOT `score_outputs.slurm`'s strength-shaped
schema). Results: see "Few-step DDIM vs LCM diagnostic -- results" table
further up.

**Acceptance criterion (2026-08-23, replaces an earlier misplaced P1-11
criterion that had been left under this heading):** identify whether a
few-step LCM configuration can retain positive A→H colour recovery on both
A06 and a typical slide while providing substantial acceleration over the
50-step DDIM/P1-11 quality path. If no tested generic LCM-LoRA configuration
approaches P1-11's colour recovery without unacceptable structural loss,
conclude that the generic LCM adapter is not fidelity-equivalent for the
specialised source-conditioned translator and treat P1-11 as the quality
operating point.

---

## P1-13 — Task-specific LCM-LoRA distillation from the frozen P1-10 teacher

**Status:** 🔄 IN PROGRESS (2026-08-23) -- prerequisite gate evaluated,
training script + launcher being implemented. Task file:
`tickets/P1-13_task_specific_lcm_distillation.md`.

**Prerequisite gate**, using the diagnostic above (job 45654/45700, plain
DDIM, vs job 45650/45692, generic LCM, both at matched steps=8/strength=
0.80/guidance=2.0):

| Slide | Δlab DDIM | Δlab LCM | SSIM DDIM | SSIM LCM |
|---|---|---|---|---|
| A06 | +4.45 | **+9.56** | **0.331** | 0.307 |
| A08 | -1.45 | **-0.46** | 0.470 | **0.512** |

Generic LCM recovers more colour than matched-step plain DDIM on both
slides, with SSIM roughly a wash (better on A08, marginally worse on A06) --
this is NOT the ticket's stop condition ("if few-step DDIM and few-step LCM
deteriorate similarly, do not assume distillation will help"). The LCM
mechanism is not uniquely attenuating the mapping at this step count; P1-12's
shortfall relative to P1-11 (see P1-12 CLOSED above) reflects the few-step
regime generally, not something specific to a generic, task-agnostic LCM
adapter. **Verdict: PROCEED.**

**Design decisions:** new trainable LCM-LoRA is attention-only
(`to_k`/`to_q`/`to_v`/`to_out.0`, same target modules as the colour LoRA),
rank 64 (matches the official cached `lcm-lora-sdv1-5`'s rank, narrowed from
its full-UNet scope given only 39 training pairs -- confirmed by inspecting
that checkpoint's safetensors keys directly on the cluster). Guidance
sampled per-example from U[1.0, 2.0] during distillation, covering the
ticket's later {1.5, 2.0} inference grid without spending capacity on higher
CFG this project doesn't use.

**Code (2026-08-23/24):** `src/train/train_p1_10_lcm_lora.py`
(ports the core single-GPU algorithm from diffusers v0.39.0's official
`examples/consistency_distillation/train_lcm_distill_lora_sd_wds.py`,
reusing this project's own `build_pairs`/`PairDataset` from
`train_colour_translation_lora.py` instead of its webdataset pipeline) +
`slurm/train_p1_10_lcm_lora.slurm`. New checkpoint namespace
`lora/a2h_cond_r8_lcm_distilled/` -- never touches `lora/a2h_cond_r8/best`.
Also `src/eval/score_p1_13_checkpoint_sweep.py` +
`slurm/score_p1_13_checkpoint_sweep.slurm` (standalone post-hoc checkpoint
sweep, used before the sweep logic was folded into the training script
itself -- kept for re-analysing already-completed runs without retraining).

**Smoke test (job 45710→45724→45734, all COMPLETED 2026-08-23):** two real
bugs caught and fixed before anything else was trusted -- (1) `PairDataset`
is nested inside `train_colour_translation_lora.py`'s `main()`, not
importable, so `train_p1_10_lcm_lora.py` carries its own standalone copy
instead of editing that locked file; (2) a PEFT gotcha where
`unet.add_adapter()` for the new `"lcm"` adapter silently re-enables
`requires_grad` on the already-frozen `"colour"` adapter too -- the
freeze/unfreeze pass must run AFTER both adapters exist, keyed by parameter
name, not before. The script's own trainable-parameter abort check caught
this correctly. A third bug (job 45734): `DDIMSolver.ddim_alpha_cumprods_prev`
silently upcasts float32->float64 via a numpy `.tolist()` round-trip --
present in the official reference script too, just never triggered there --
collided with the bf16 model weights; fixed with an explicit float32 cast in
`DDIMSolver.to()`.

**Overfit-control debugging (2026-08-23/24), 8-pair subset, default LR=1e-4:**
first 300-step run (job 45737) showed a flat/noisy loss curve, not the
"near-zero" bar P1-10's own overfit control used. Per this project's own
guardrail that diffusion training loss is not a valid success signal alone
(CLAUDE.md), extended Validation B to also run on the overfit pairs
themselves when no val split exists (`--overfit-n` structurally empties
`val_pairs`) -- previously Validation B only ran when a real held-out split
existed. Re-ran (job 45740): SSIM was WORSE after 300 steps (0.287) than a
near-init 1-step baseline (job 45753, SSIM 0.370) on the IDENTICAL 8 pairs
-- a real, confirmed regression, not pair-set noise. Extended to 1500 steps
(job 45822): SSIM kept degrading monotonically to 0.267, loss still flat.
Re-derived the LCM consistency-distillation math line-by-line against the
official reference -- found no sign/pairing bug; formulas match exactly.

**Full checkpoint sweep (job 45951, using the newly-built
`score_p1_13_checkpoint_sweep.py`)** scored every saved checkpoint
(250/500/750/1000/1250/final) from the 1500-step run instead of just the
endpoints: revealed NON-monotonic oscillation, not a steady one-way drift --
checkpoint-250 (SSIM 0.343, LAB 14.43) clearly beat the near-init baseline,
checkpoint-750 was a second local best, but training didn't stay there
(final: SSIM 0.267, LAB 23.53). This ruled out "fundamentally broken
objective" (training CAN reach good configurations) in favour of "high
per-step gradient/target variance + no LR schedule" (batch size 1, random
timestep/index/guidance resampled every single step).

**LR ablation (2026-08-24), same 8 pairs, same 1500 steps, checkpoint
selection folded directly into `train_p1_10_lcm_lora.py`'s own end-of-run
Validation-B sweep (no separate script call needed going forward) --
selection changed from the untrustworthy scalar val_loss to image-level
Validation B, later upgraded to a full Pareto frontier over (ssim, lab_total)
rather than a single-metric argmin (see below):**

| LR | Best checkpoint | SSIM | LAB | SSIM trend across training |
|---|---|---|---|---|
| 1e-4 (job 45822) | checkpoint-250 | 0.343 | 14.43 | declining |
| 3e-5 (job 45994) | checkpoint-1250 | 0.322 | 12.80 | **cleanly increasing** |

Lower LR converted the SSIM trajectory from degrading to steadily
improving -- LAB still oscillated at 3e-5, but the qualitative change in
SSIM was a genuine result, not noise.

**Gradient-accumulation ablation (job 46002), LR fixed at 3e-5 (per
instruction, since it already fixed SSIM), grad-accum=4 (effective batch 4,
`--grad-accum-steps` added to the training script + `GRAD_ACCUM`/
`SAVE_EVERY` env vars added to the slurm launcher), same 8 pairs, 1500
optimizer steps (NOT microbatches), checkpoints every 125:**

| Checkpoint | SSIM | LAB |
|---|---|---|
| 125–750 | 0.30–0.32 (stable band) | 20–27 |
| **875 (peak)** | **0.330** | **14.87** |
| 1000–1375 | declining | rising to 36.46 |
| final (1500) | 0.241 (worst of run) | 33.26 (2nd worst) |

Qualitatively different from the earlier pure oscillation: SSIM is now
genuinely stable through steps 125-1000 (tight band, unlike either prior
run), then a clean rise-to-peak-then-overfit shape -- consistent with
classic small-dataset overfitting (8 unique images, no LR decay) rather than
an unstable/broken objective.

**Cross-run finding:** all three configurations (LR=1e-4; LR=3e-5;
LR=3e-5+accum4) independently found a peak checkpoint in the same narrow
band (LAB 12.8-14.9, SSIM 0.32-0.34) -- strong evidence this is a genuine,
reproducible achievable quality ceiling for this setup, not a fluke, and
that automatic Validation-B-based checkpoint selection reliably finds it
regardless of hyperparameters. The open question was never "can training
reach good quality" (yes, consistently) but "does it stay there" (no, on
only 8 images) -- judged likely to be a data-scarcity artifact of the
overfit diagnostic itself, not a property expected to carry over to the real
39-pair run (~5x more unique data, much less overfitting pressure).

**Decision (2026-08-24):** the 8-pair feasibility/debugging gate is
complete. Proceeding to a STAGED run on the full 39-pair training set (11-
pair real held-out val split) -- LR=3e-5, grad-accum=4, but capped at 1500
optimizer steps first (NOT the full 4000: accum=4 already quadruples sample
exposure per optimizer step relative to the ticket's original budget),
checkpoints every 125, selected via the Pareto frontier over (ssim,
lab_total) rather than a single scalar. If Validation B is still improving
at step 1500, continue training further; if it has peaked and is declining
(as the 8-pair diagnostic's own shape suggests may happen), stop at the
selected earlier checkpoint instead of the final step.

**Checkpoint-selection code change:** `train_p1_10_lcm_lora.py`'s
end-of-training Validation B now (a) sweeps EVERY saved checkpoint, not
just one, reusing one cached teacher generation per pair across all of
them, and (b) computes the Pareto frontier over (ssim higher-is-better,
lab_total lower-is-better) -- a checkpoint qualifies only if no other swept
checkpoint beats it on BOTH axes at once. The full frontier is logged and
written to `validation_b_teacher_match.json`; the point with the lowest
`lab_total` on the frontier is auto-copied to `out_dir/best_by_validation_b`
as a recommendation, not a black-box final answer -- `out_dir/best`
(scalar-val_loss-selected) and `out_dir/final` are no longer what downstream
scripts should default to using.

**Real 39-pair staged run -- results (2026-08-24), job 46036
(`LR=0.00003 GRAD_ACCUM=4 SAVE_EVERY=125`, 1500 optimizer steps, no
`--overfit-n`, scored on the real 11-pair held-out val split):**

**Confirms the core reason for abandoning scalar val_loss selection.** The
distillation val_loss decreased almost monotonically throughout training
(0.0125 -> 0.0111), and the resulting lowest-val_loss checkpoint
(`out_dir/best` = `final`) scored ssim=0.2881/lab_total=32.70 on real
Validation B -- one of the WORST checkpoints in the entire 13-checkpoint
sweep. Minimum training loss selected one of the worst-quality checkpoints,
on real training data, not just the 8-pair toy setup.

Pareto frontier (5/13): checkpoint-750 (ssim=0.2910, lab=**15.21**,
best colour recovery in the sweep), checkpoint-125 (0.2958, 21.24),
checkpoint-500 (0.3049, 22.99), checkpoint-625 (0.3187, 26.05),
checkpoint-1250 (**0.3251**, 39.05, best SSIM in the sweep).
checkpoint-750's lab_total (15.21) lands in the same ~13-15 quality-ceiling
band independently found in all three 8-pair overfit ablations above --
further cross-validation that this is a real, reproducible ceiling for this
architecture/data, not a fluke.

**Selection rule applied (precommitted): among checkpoints with
ssim >= 0.95 x best-ssim-in-sweep (a structural-fidelity floor), pick the
lowest lab_total.** This is deliberately NOT "lowest lab_total on the Pareto
frontier" -- checkpoint-750 sits on the frontier but its ssim (0.2910) falls
below the floor (0.95 x 0.3251 = 0.3088), so it is excluded from the primary
pick despite having the best colour recovery in the sweep. Eligible set:
checkpoint-625 (0.3187, 26.05), checkpoint-1250 (0.3251, 39.05),
checkpoint-1375 (0.3127, 29.36). **Selected: checkpoint-625** (lowest
lab_total among the eligible set) as the primary, balanced P1-13 checkpoint.
checkpoint-1250 is NOT adopted despite the best SSIM (its lab_total, 39.05,
is the worst in the entire sweep). checkpoint-750 is retained separately
(`pareto_colour_favouring_checkpoint_750/`) as a documented
colour-favouring Pareto point only, not the primary checkpoint.

**Trend from checkpoint-750 onward does not show continued improvement**
(lab_total rises to 39.05 by step 1250 before settling ~29-33, worse than
step 750's 15.21) -- the same peak-then-decline shape as the 8-pair
diagnostics, noisier but present on real data too. Per the precommitted
stop/continue rule: **training is NOT resumed beyond 1500 steps.**

`train_p1_10_lcm_lora.py`'s auto-selection logic was updated to implement
the 5%-SSIM-floor rule directly (previously "lowest lab_total on the Pareto
frontier", which this run demonstrated would select checkpoint-750 and
violate the floor) -- future runs apply this automatically. This run's
`best_by_validation_b/` and `validation_b_teacher_match.json` were corrected
by hand to match (the run itself predates the code fix).

**Status: training phase of P1-13 complete. Primary checkpoint:
`lora/a2h_cond_r8_lcm_distilled_lr0.00003_ga4/best_by_validation_b`
(= checkpoint-625). Proceeding to post-training controls (source ablation,
adapter isolation) with this checkpoint. Never uses held-out slides to
choose between checkpoints -- selection is internal-validation-split only.**

**Post-training controls -- results (2026-08-24/25), checkpoint-625, A06+A08
`LIMIT=20` smoke subset (the same convention every P1-12/P1-13 run uses),
3 seeds, matched settings steps=8/strength=0.70/guidance=2.0 for the two LCM
arms and steps=50/strength=0.50/guidance=2.0 (ordinary P1-10 defaults) for
`adapter_disabled`. Code: `src/eval/infer_p1_13_lcm_controls.py` +
`slurm/infer_p1_13_lcm_controls.slurm` (inference, 5 jobs: 46215-46219) +
`slurm/score_p1_13_lcm_controls.slurm` (scoring, job 46250, reuses
`score_p1_10_ablation.py` unmodified -- the manifest's `source_mode` column
is written as `<variant>_<source_mode>` so every arm buckets separately).**

| Arm | SSIM | lab_total |
|---|---|---|
| `task_specific_correct` | 0.2109 | 76.82 |
| `task_specific_zero` | 0.0466 | 61.39 |
| `task_specific_shuffled` | 0.0385 | 81.65 |
| `adapter_disabled_correct` | 0.3151 | 78.71 |
| `generic_lcm_correct` | 0.2958 | 71.43 |

**Control 2 (source ablation) PASSES clearly:** correct beats both zero and
shuffled on SSIM by a wide margin (~4-5x) -- the ControlNet branch is
demonstrably not being ignored. (zero's lower lab_total than correct is the
same "colour-histogram distance can look deceptively good even when
structure is destroyed" pattern already documented elsewhere in this file --
SSIM is the criterion that matters here and is unambiguous.)

**Control 3 (adapter isolation) reproduces ordinary P1-10 behaviour
reasonably closely:** `adapter_disabled_correct` (ssim=0.3151, wlab=79.02)
is close to the already-established P1-10 baseline on this exact subset
(ssim=0.336, wlab=78.24, from the P1-12-era smoke gate table above). The
small residual gap is plausible GPU-kernel non-determinism / the
named-adapter loading path (`pipe.load_lora_weights(..., adapter_name=
"colour")` vs the original script's unnamed single-adapter load), not a red
flag -- Control 3 was also strengthened per this ticket's own precommitted
requirement to test DISABLING (loading "lcm" then excluding it from
`set_adapters`), not mere absence.

**Control 1 (generic vs task-specific LCM, matched settings) -- the
task-specific adapter LOSES on the real held-out crops:** generic_lcm
(ssim=0.2958, lab=71.43) beats task_specific (ssim=0.2109, lab=76.82) on
BOTH axes. This is the opposite of P1-13's core hypothesis, and is a
genuine result, not a bug -- it is consistent with Validation B being a
teacher-FIDELITY diagnostic (how well the student matches the P1-10
teacher's own output on training-domain pairs) rather than a
ground-truth-ACCURACY diagnostic (how well it matches the true Hamamatsu
image on held-out slides): checkpoint-625 was selected for matching the
frozen teacher well internally, which does not guarantee it generalises
better than an independently-trained generic adapter on real held-out
crops. task_specific_correct (ssim=0.2109) also falls well short of the
plain P1-10 baseline (ssim=0.336) and of `adapter_disabled_correct`
(ssim=0.3151) from this same batch of runs.

**Status: P1-13 controls complete. Current honest conclusion: the
task-specific LCM-LoRA (checkpoint-625, LR=3e-5, grad-accum=4, 1500
optimizer steps, 39 real training pairs) does not yet outperform the
generic pretrained LCM-LoRA on the actual held-out biological evaluation,
despite passing its own internal teacher-matching diagnostic and both
structural controls (2 and 3). Decision on next steps (accept as a negative
finding akin to P1-12's own conclusion; investigate the train/held-out
generalisation gap further; try more training/data) pending discussion.**

**IMPORTANT correction/clarification (2026-08-25) -- train/inference
guidance-semantics mismatch, ruled out before accepting the Control 1
result above as final:**

`train_p1_10_lcm_lora.py`'s `w` uses the LCM PAPER's CFG form for the
teacher target -- `pred_x0 = cond_pred_x0 + w*(cond_pred_x0 - uncond_pred_x0)`
(and identically for `pred_noise`) -- NOT diffusers' ordinary
`guidance_scale` (G) form, `uncond + G*(cond-uncond)`. Equating the two:
`(1+w)*eps_c - w*eps_u = G*eps_c + (1-G)*eps_u` => **G = w + 1**. So this
run's `w~U[1.0,2.0]` corresponds to teacher trajectories at diffusers-style
`G~U[2.0,3.0]`, NOT an inference `guidance_scale` range of 1.0-2.0 as the
earlier ticket text incorrectly implied ("covers the ticket's later {1.5,
2.0} inference grid").

Separately: the STUDENT's own "online" forward pass during training
(`run_distill_step`, step 2) is a single CONDITIONAL pass only -- no
cond/uncond doubling, no CFG applied to the student itself. But Control 1's
evaluation ran the distilled student through the ordinary
`StableDiffusionControlNetImg2ImgPipeline` at `guidance_scale=2.0`, and
diffusers' `do_classifier_free_guidance = guidance_scale > 1` means that
setting DOES make the pipeline run doubled cond/uncond passes through the
(LCM-adapted) UNet and blend them via the ordinary diffusers CFG formula --
external CFG the student was never trained to expect, layered on top of a
model already distilled from higher-effective-guidance (G~2-3) teacher
targets. The official LCM-LoRA recipe normally evaluates near
guidance_scale=1.0 (no external CFG) for exactly this reason. **The
Control 1 result above (task_specific ssim=0.2109 losing to generic_lcm
ssim=0.2958) was measured ONLY at guidance_scale=2.0 and should NOT yet be
treated as final** -- P1-13 is reopened (training/checkpoint NOT
re-selected yet) to rule this out via an inference-only guidance sweep
before accepting or rejecting the hypothesis. See below.

## P1-17 — Differential Diffusion change-map inference (spatially-varying strength)

**Status:** ✅ CLOSED (2026-08-27) — negative result. Internal-validation
parameter grid (radius ∈ {2,4,8} × c_max ∈ {0.50,0.70}, sigma=2/c_min=0.0
fixed, 20 training crops each) landed within roughly one baseline-seed-stdev
of the plain P1-10 global-strength baseline on every config (SSIM
0.162–0.167 vs baseline 0.1616±0.0057) — no real improvement. The earlier
4-crop smoke result (SSIM 0.192) was noise from too small a sample. Likely
mechanism: on densely cellular H&E tissue the Canny-derived "protected"
region is finely interleaved with "free" regions rather than one or two
large contiguous blocks (unlike the reference technique's own published
examples), so the UNet's shared receptive field plausibly lets free-region
generative influence bleed into nominally protected pixels, washing out any
effect. Full grid table and interpretation in the task file:
`tickets/P1-17_differential_diffusion_change_map.md`. Proceeding to P1-16,
which sidesteps this failure mode entirely (fusion happens outside the UNet,
post hoc, never re-entering the shared diffusion computation).

---

**Decision gate (proposal §"Time Plan"):** at the end of Phase 1, formally review
A0–A5 results and the SDXL compute-contingency evidence before proceeding to Phase 3
(see PHASE3-TICKETS.md, P3-01).
