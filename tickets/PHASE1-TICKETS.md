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
**Status:** 🔄 IN PROGRESS (2026-08-20) — approved, infrastructure built and
heavily bug-fixed (two review rounds, six real bugs found and fixed — see
below), overfit-test control run at 300 steps, mandatory ablation control run
and FAILED (see "Progress log" below) — re-running at more steps before
concluding anything. This does not reopen, replace, or relabel the completed
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

**Not yet done**: re-run the ablation on the 2000-step checkpoint; VAE-only
floor check; full training run; smoke gate; full held-out evaluation.

---

**Decision gate (proposal §"Time Plan"):** at the end of Phase 1, formally review
A0–A5 results and the SDXL compute-contingency evidence before proceeding to Phase 3
(see PHASE3-TICKETS.md, P3-01).
