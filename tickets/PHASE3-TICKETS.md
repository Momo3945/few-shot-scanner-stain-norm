# Phase 3 — Architecture Transfer to SDXL

Source of truth: `docs/proposal.tex`, §"Phase 3: Architecture Transfer to SDXL"
(`sec:phase3_sdxl`, `sec:sdxl_a5`).

**Unblocked (2026-08-10)** — Phase 1 (A0–A5) is complete and the decision gate
(P3-01) has been reviewed. Outcome (i): proceed as planned.

---

## P3-01 — Architecture decision gate review
**Status:** ✅ DONE (2026-08-10) — outcome (i): results are clean, SDXL is
feasible (compute confirmed, see P3-02). Phase 3 proceeds as planned.
**Source:** proposal §"Time Plan", "Architecture decision gate" paragraph
**Description:** Formally review SD1.5 A0–A5 results plus SDXL compute-contingency
evidence. Three possible outcomes per the proposal:
  (i) results clean + SDXL feasible → proceed as planned (P3-03);
  (ii) SDXL expected slow → descope to smaller rank / fewer eval slides;
  (iii) Phase 1 reveals a fundamental architecture issue → run deeper SD1.5
       diagnostics instead of transferring; SD1.5 ablation ladder remains the
       primary contribution regardless.
**Review (2026-08-10):**
- **A0–A5 results are clean** — every rung produced valid, verified, non-crashing
  full held-out results (`docs/results/RESULTS_SUMMARY.md`); no fundamental
  architecture issue found, so outcome (iii) does not apply.
- **SDXL compute is confirmed feasible** — SDXL base and SD3.5-large weights are
  both fully cached and verified with real byte sizes (SDXL base 6.94GB,
  SD3.5-large 16.46GB; see CLAUDE.md, corrected from a stale "KNOWN BROKEN" note).
  No compute blocker found, so outcome (ii)'s descope path is not currently needed
  (revisit if actual SDXL training on `bigbatch` turns out slow in practice).
- **Base config recommendation for P3-03: A4** (ControlNet + colour LoRA + LCM-LoRA),
  not A3. The proposal's own abstract frames the contribution as a unified pipeline
  combining all three components with fast inference as a co-equal requirement
  (not optional acceleration on top of a "real" A3 result) — A4 meets the proposal's
  own bar for a clean result (SSIM flat-to-better than the 50-step DDIM reference at
  every scope; the pre-committed 20-step-DDIM fallback exists for structural/artefact
  failure, which did not occur).
- **P3-05 (A5 warm-start transfer) is triggered, not just contingent-possible** —
  `sec:sdxl_a5`'s literal test is measurable A5 vs A4 improvement "beyond the
  inter-run noise floor." P2-05 established that floor at ~0.1–0.3 LAB units
  (rank-change noise). A5 beats A4 on A06 by +2.6 / +5.8 / +8.7 LAB units across
  strengths 0.30/0.40/0.50 — one to two orders of magnitude past the floor. See
  P3-05 below, now unblocked.
- ~~**New Phase 3 blocker, not previously on this list:** `latent-consistency/
  lcm-lora-sdxl` and an SDXL Canny ControlNet checkpoint are not in the cache at
  all~~ **RESOLVED (2026-08-10)** — job 40399 fetched both, verified with real
  byte sizes not exit code: `lcm-lora-sdxl` 393.9MB, `diffusers/controlnet-
  canny-sdxl-1.0` 5.0GB. See P3-02 below. P3-03 is now fully unblocked.

## P3-02 — Fix SDXL model weight download
**Status:** ✅ DONE (2026-08-08) — job 37073 COMPLETED 16:49; verified with real file
checks, not exit code: `models--stabilityai--stable-diffusion-xl-base-1.0/` = 33G,
13 `.safetensors` files present.
**Source:** infrastructure prerequisite for all of Phase 3
**Root cause (confirmed via `--dry-run`, not guessed):** `hf download`'s `--include`
is a single-value option (Click-based CLI: `hf download [OPTIONS] REPO_ID
[FILENAMES]...`). The old `--include "*.safetensors" "*.json" "*.txt" "*.model"`
bound only `*.safetensors` to `--include`; the other three patterns became
positional `FILENAMES`, which switches `hf download` to a code path that ignores
`--include` entirely. Reproduced the exact broken result via dry-run: 17
config/tokenizer files, 3.2M, zero `.safetensors` — matches the actual cache exactly.
**Fix applied:** `slurm/fetch_models.slurm` now uses repeated `--include` flags (the
CLI's own documented syntax), verified via dry-run to actually select the
`.safetensors` files this time.
**Update (2026-08-10):** `latent-consistency/lcm-lora-sdxl` and an SDXL Canny
ControlNet (`diffusers/controlnet-canny-sdxl-1.0`) added to `fetch_models.slurm`
and fetched — job 40399 COMPLETED 4:20, verified with real byte sizes not exit
code: `lcm-lora-sdxl` 393.9MB, `controlnet-canny-sdxl-1.0` 5.0GB (fp16 + fp32
variants both present). All Phase 3 model weights are now cached and verified.

## P3-03 — Transfer best Phase 1 configuration to SDXL
**Status:** ✅ DONE (2026-08-19) — A2H colour LoRA trained on SDXL (job 43969),
inference pipeline built and debugged (offline snapshot-completeness bug found
+ fixed), full inference run (job 44206) and scoring (job 44282) complete, all
496 held-out crops. See P3-04 for the comparison result — a genuinely
interesting one, not a clean transfer.
**Source:** `sec:phase3_sdxl`
**Description:** Transfer ONLY the best-performing SD1.5 configuration (from A0–A5).
The proposal is explicit: **do not** repeat the full ablation ladder on SDXL — that
would multiply compute cost without proportional scientific value, since Phase 1
already isolates each component's contribution.
**Tooling:** same diffusers training ecosystem; `lcm-lora-sdxl` for 2–8 step inference.
**Plan (2026-08-10):** fork `train_colour_lora_sdxl.py`/`infer_colour_lora_sdxl.py`
from the SD1.5 scripts (dual text encoders + pooled embeds + micro-conditioning
make a flag-based extension impractical; the multi-adapter `set_adapters`
composition logic ports over almost unchanged). Force the VAE to fp32 while the
rest of the pipeline stays fp16 (SDXL's official VAE is documented to NaN under
fp16 — must not be "fixed" back to fp16 by a future edit). New Slurm launchers
follow the existing OUT_TAG/fail-fast conventions, output to
`lora/a2h_r8_sdxl/` and `eval/a4_sdxl/` (namespaced so nothing collides with
the committed SD1.5 results). `score_outputs.py` needs zero changes (confirmed
fully generic over resolution, same as the P2-11 baselines).
**Resolution decision (2026-08-10):** train at the **existing 512×512 crops**
in `pairs/train`, told honestly to SDXL via its micro-conditioning
(`original_size=(512,512)`) rather than spoofed as 1024. Zero new data
extraction, keeps `pairs/train` and its leak-checks completely untouched,
fastest path to a first real result. If this underperforms or looks
structurally degraded, see the native-1024 fallback in **P3-03b** below —
do not preemptively re-extract data before finding out whether 512 was
actually the bottleneck.
**Implementation (2026-08-17/18):** `src/train/train_colour_lora_sdxl.py` +
`src/eval/infer_colour_lora_sdxl.py` + `slurm/train_colour_lora_sdxl.slurm` +
`slurm/infer_colour_lora_sdxl.slurm`, forked from the SD1.5 scripts per the plan
above. Confirmed `diffusers 0.39.0` in the `stainnorm` env exposes
`StableDiffusionXLControlNetImg2ImgPipeline`/`StableDiffusionXLImg2ImgPipeline`/
`StableDiffusionXLPipeline.save_lora_weights` — no new env needed. Training
script builds `encoder_hidden_states` by concatenating both text encoders'
penultimate hidden states and `added_cond_kwargs["text_embeds"]` from
`text_encoder_2`'s pooled output, with `add_time_ids` set to the honest
`(512,512)/(0,0)/(512,512)` micro-conditioning values (not spoofed 1024, per the
resolution decision above). VAE forced fp32 and kept outside the autocast
region in both scripts, exactly as planned. `unet.enable_gradient_checkpointing()`
added (new vs. the SD1.5 script — SDXL's ~2.6B-param UNet needed it to fit
LoRA fine-tuning on a 24GB RTX 3090 alongside bf16 autocast + fp32 VAE).

**Training run (job 43969, `mscluster60`, COMPLETED 17:24):** A2H direction
only (P3-03's scope — H2A-on-SDXL is out of scope, not needed for the P3-04
comparison), rank 8, 1000 steps. Real finite loss throughout (0.06–0.36 range,
no NaN), final checkpoint `lora/a2h_r8_sdxl/final/pytorch_lora_weights.safetensors`
verified as a real 44.4MB file (matches 11.6M trainable params, not a stub).
A 5-step smoke test (job 43763) ran first and passed cleanly before committing
to the full run — same discipline as every other training job this project runs.

**Bug found and fixed (2026-08-18): offline pipeline loading failed despite
component-level loading succeeding.** The training script loads SDXL components
individually (`AutoencoderKL.from_pretrained(..., subfolder="vae")` etc.) and
worked immediately. The inference script's pipeline-level
`StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(...)` call failed
under `HF_HUB_OFFLINE=1` with `OSError: model is not cached locally`, even
though `model_index.json` and every subfolder were present. Root cause,
confirmed by direct testing with `local_files_only=True`: diffusers' full-
pipeline load validates the **entire repo's file manifest** (not just the
subfolders it actually needs) before proceeding — `fetch_models.slurm`'s
original download used `--include "*.safetensors" --include "*.json" --include
"*.txt" --include "*.model"` to save bandwidth, which left 27 non-essential
files missing (`.gitattributes`, example PNGs, `LICENSE.md` — no weights among
them), and that's enough to fail the pipeline-level completeness check even
though component-level loads never look at those files. **Fix:** a plain
`hf download stabilityai/stable-diffusion-xl-base-1.0` (no `--include` filter)
completed the snapshot to 57 files (job 44124, 7:26, verified via a direct
`local_files_only=True` reload afterward — the earlier "✓ Downloaded"-only
check would NOT have caught this, per CLAUDE.md's standing warning about
trusting exit codes/log lines alone). **Worth remembering for any future SDXL/
SD3.5 pipeline-level load**: `--include` filters that work fine for
component-level loading can silently break pipeline-level loading later.

**Inference (job 44125, smoke test, `mscluster49`, COMPLETED 10:01; job 44206,
full run, COMPLETED 59:58):** 496 output crops — matches the SD1.5 runs' crop
count exactly, same held-out set. Scored (job 44282, `score_outputs.slurm`,
COMPLETED 20:23) — see P3-04 below for the result.
**Status: DONE** as far as this ticket's own scope (transfer + inference +
scoring). P3-05 (A5-on-SDXL) remains a separate, still-unstarted ticket.

## P3-03b — Supplementary: SDXL training at native 1024×1024 resolution (contingent)
**Status:** TODO — contingent, not started. Only pursue if P3-03's 512×512
result underperforms or looks structurally degraded; do not start this
pre-emptively.
**Source:** derived from P3-03's planning pass (2026-08-10) — SDXL was
predominantly trained at ≥1024px, so this is the fallback path if training at
512 (P3-03's chosen default) turns out to be the bottleneck rather than the
backbone itself.
**Description:** Re-extract MITOS training pairs at 1024×1024 into a **new**
sibling folder `pairs/train_1024` (via `extract_pairs.py --crop 1024`) — never
overwrite `pairs/train`, since the SD1.5 pipeline still depends on it. Retrain
the SDXL colour LoRA (`train_colour_lora_sdxl.py`, once it exists per P3-03) on
this new data at native resolution, then repeat P3-03's inference+scoring
sequence for a direct 512-vs-1024 comparison.
**Known constraint (unresolved, check before starting):** the raw MITOS-ATYPIA
training frames are only ~1539×1376 (Aperio) / ~1663×1485 (Hamamatsu) per
`extract_pairs.py`'s own docstring — only ~1.5× a 1024 crop, vs ~3× at 512.
Re-extracting at 1024 will sharply shrink the number of usable per-frame crop
positions `grid_offsets()` can place under the same tissue-threshold filter, so
the resulting training set will be meaningfully smaller than today's 512-crop
`pairs/train`. Before starting: confirm the actual usable crop count with a
read-only cluster check (raw frame dimensions × `grid_offsets(H,1024)` ×
`grid_offsets(W,1024)` per frame × number of raw training frames) and decide
whether that's enough data for a rank-8 LoRA before committing training compute.
**Trigger condition:** pursue only if P3-04's SDXL-vs-SD1.5 comparison shows
either (a) visibly degraded/structurally off SDXL outputs at 512, or (b) SDXL
underperforming SD1.5's A4 in a way plausibly attributable to under-resolution
training rather than the backbone itself.

## P3-04 — SDXL vs SD1.5 comparison
**Status:** ✅ DONE (2026-08-20) — mechanism identified (frozen SDXL base
drifts colour at this operating point, not the LoRA), and the follow-up A08
sweep confirms no untested strength rescues the pooled comparison: SDXL's
best measured pooled SSIM (0.527) stays below every classical baseline.
Supplementary track only (P1-10 is the more load-bearing open hypothesis for
the structural-fidelity gap). P3-05 (A5 warm-start) remains open separately.
**Source:** `sec:phase3_sdxl`
**Description:** Compare the transferred SDXL configuration against its SD1.5
counterpart using the same colour, structure, and speed metrics from Phase 2's
harness (reuse `score_outputs.py`).

**Result — A4 config, strength 0.20, both backbones, same 496-crop held-out
set:**

| Backbone | ALL recovery Δlab | ALL SSIM | A06 Δlab | A06 SSIM |
|---|---|---|---|---|
| SD1.5 | **+1.53** | 0.459 | +0.42 | — |
| SDXL | **−5.60** | **0.527** | −3.58 | 0.386 |

SDXL's colour-fidelity result is not just weaker than SD1.5's at the same
operating point — it's the **opposite sign**. The pipeline moves colour
*away* from the target scanner on the larger backbone, on every slide
(A06 −3.58, A08 −5.92, A09 −5.19, A13 −4.75, A16 −5.10 — negative across the
board, not just pooled). At the same time, **structural fidelity is
genuinely better on SDXL** (SSIM 0.527 vs SD1.5's 0.459 — the highest SSIM
recorded anywhere in this project for this config).

**Reading this, not yet conclusive:** two things worth testing before treating
this as "SDXL is worse at this task":
1. **The most likely candidate — the operating point may not actually
   transfer.** "Strength 0.20 + 8-step LCM" was tuned as the best point *for
   SD1.5's specific LCM-LoRA distillation*. `latent-consistency/lcm-lora-sdxl`
   is a separately-trained distillation on a different base model — there's
   no guarantee it maps the same nominal strength/step values onto the same
   effective noise level or number of real denoising steps. Higher SSIM +
   negative colour shift together is exactly the signature you'd expect if
   SDXL's LCM path is *effectively more conservative* than SD1.5's at the
   same nominal settings (preserves more of the input generally) — which
   would explain the better structure score, but also means the colour LoRA
   gets even less relative influence to work with, and whatever colour drift
   the frozen SDXL base's own prior contributes (conditioned on the fixed
   "H&E stained histopathology tissue" prompt) isn't necessarily aligned with
   Hamamatsu's specific palette the way it may incidentally have been for
   SD1.5's base. A proper SDXL-side strength sweep (the same P1-09-style
   investigation already done for SD1.5) hasn't been run yet — this is one
   untuned point, not SDXL's best possible point.
2. **Possible under-training relative to backbone size.** SDXL's UNet is
   ~3x SD1.5's parameter count; the same 1000 training steps at rank 8 that
   was sufficient for SD1.5 may just not be enough signal for a
   proportionally larger network to learn as strong a colour-shifting
   adapter. Not yet tested (would mean a longer SDXL training run, not a
   re-scoring).

### Both follow-ups run (2026-08-20) — mechanism identified, question narrowed but not fully closed

**SDXL base-model-only ablation (job 44370 full inference, job 44425 scoring,
COMPLETED)** — same A4 config, colour LoRA simply omitted:

| | ALL Δlab | ALL SSIM |
|---|---|---|
| With colour LoRA (job 44282) | −5.60 | 0.527 |
| **Base model only, no LoRA** (job 44425) | **−4.35** | **0.527** |

SSIM is identical to 3 decimal places, and the negative colour drift is
already fully present with **zero** LoRA involvement. This settles the
mechanism question: the negative recovery is not the trained adapter doing
something wrong — the frozen SDXL base (+ ControlNet + LCM) already drifts
away from Hamamatsu's colour at this operating point before the LoRA ever
touches it. The LoRA's actual measured effect is small (−4.35 → −5.60, ~1.25
units) and, if anything, pushes slightly further in the wrong direction
rather than correcting it — consistent with a lightly-trained (1000 steps,
≤50 pairs) adapter being outweighed by a much larger frozen base's own
generation tendency at this setting.

**SDXL strength sweep (job 44369 inference, job 44382 scoring, COMPLETED,
A06 only — see caveat below):**

| Strength | Δlab (A06 only) | SSIM |
|---|---|---|
| 0.20 | −2.45 | 0.399 |
| 0.30 | −1.33 | 0.374 |
| 0.40 | **+1.56** | 0.348 |
| 0.50 | +5.64 | 0.323 |
| 0.70 | +9.64 | 0.299 |

Colour recovery flips from negative to positive between strength 0.30 and
0.40 on A06, climbing steadily after that — with SSIM falling the whole time
(the same structure/colour tradeoff seen throughout this project). **Caveat
that keeps this from being conclusive**: this diagnostic pass only covered
A06 (the `LIMIT=8` diagnostic run happened to grab only A06's frames) — and
A06 is already independently documented (SD1.5 ladder) to recover *more* at
higher strength while typical slides recover *less*. So this result is
consistent with either (a) SDXL's whole colour-recovery curve is genuinely
shifted and needs a higher strength across the board, or (b) A06 is simply
following its own already-known strength-sensitivity pattern, and this says
nothing new about SDXL's calibration on typical slides. The original negative
finding (−5.60) was on the **pooled 496-crop, all-5-slide** set at strength
0.20 — this sweep hasn't touched that comparison at any strength but 0.20 yet.

**Net read**: the mechanism is now identified (frozen-base behaviour at this
nominal operating point, not an adapter bug), and there's a real,
strength-dependent path to positive A06 recovery on SDXL. What's still open:
whether a higher strength also helps SDXL's *typical*-slide/pooled numbers,
or whether (mirroring SD1.5's own pattern) higher strength on SDXL trades A06
gains for worse typical-slide performance, in which case there may be no
single SDXL strength that beats SD1.5's pooled A4@0.20 result. **Not yet
done**: repeat the strength sweep on a typical slide (e.g. A08) before
concluding either way.

### A08 (typical slide) strength sweep (job 44558 inference, job 44563
scoring, COMPLETED 2026-08-20) — closes the open question, negative

| Strength | Δlab (A08) | SSIM (A08) | Δlab (A06, known) | SSIM (A06, known) |
|---|---|---|---|---|
| 0.20 | −5.92 | 0.562 | −2.45 | 0.399 |
| 0.30 | −5.60 | 0.534 | −1.33 | 0.374 |
| 0.40 | −4.66 (least bad) | 0.507 | +1.56 | 0.348 |
| 0.50 | −5.14 | 0.477 | +5.64 | 0.323 |
| 0.70 | −5.71 | 0.432 | +9.64 | 0.299 |

A06's "colour recovery flips positive at higher strength" trend does **not**
generalise. On A08, colour recovery stays negative at every strength tested,
never crossing zero, with no clean monotonic trend (least-bad point is 0.40,
not either extreme). SSIM falls monotonically with strength exactly as on
A06 (0.562 → 0.432). **Conclusion: there is no single SDXL strength that
both fixes colour recovery and preserves structure on a typical slide** — A06
and A08 pull in genuinely opposite directions as strength rises, mirroring
the same A06-vs-typical-slide asymmetry already documented for SD1.5.
Strength-tuning cannot rescue the pooled A4@0.20 comparison; SDXL's pooled
SSIM (0.527, job 44282/44206) remains below every classical baseline
(Macenko 0.628, Reinhard 0.681, Histogram Matching 0.651) and no untested
strength is expected to change that. This closes P3-04's open question — no
further strength sweeps planned. Whatever fixes the structural-fidelity gap,
if anything does, is more likely to come from P1-10 (source-conditioned
training on SD1.5, see `tickets/PHASE1-TICKETS.md`) than from further SDXL
tuning.

## P3-05 — A5 warm-start variant on SDXL (contingent)
**Status:** ✅ DONE (2026-08-21) — small, real, consistent colour-recovery
improvement over A4-SDXL, but a fundamentally different story than SD1.5's
A5: no A06-specific win, no sign flip, doesn't change the P3-04 verdict.
**Source:** `sec:sdxl_a5`
**Description:** Only transfer the histopathology warm-start variant to SDXL if A5
first shows measurable benefit over A4 on SD1.5 (beyond the inter-run noise floor).
**Trigger condition MET (2026-08-10):** A5 beats A4 on A06 recovery delta by
+2.6/+5.8/+8.7 LAB units at strengths 0.30/0.40/0.50 (`docs/results/RESULTS_SUMMARY.md`,
`tickets/PHASE1-TICKETS.md` P1-07) — the established noise floor for adapter-level
changes is ~0.1–0.3 LAB units (P2-05, rank 4 vs 8), so this is 10-30x past it, not a
borderline call. Note A5's benefit is slide-dependent (worse than A4 on typical
slides A08/A16 at low strength, though that gap narrows or flips positive at higher
strength per the Phase 1 follow-up experiments) — this nuance should carry over into
how the SDXL transfer is scoped and reported, not just "A5 wins."

**Build (2026-08-20/21):** `src/train/train_hist_lora_sdxl.py` (new — merges
`train_hist_lora.py`'s unpaired-pool data loading, incl. its independent
held-out-slide leak check, with `train_colour_lora_sdxl.py`'s SDXL machinery:
dual text encoders, fp32 VAE, gradient checkpointing). `infer_colour_lora_sdxl.py`
extended with `--hist-lora`/`--hist-scale`, generalising the existing 2-way
(colour+lcm) multi-adapter `set_adapters` composition to 3-way
(colour+hist+lcm) — direct port of `infer_colour_lora.py`'s proven P1-07
pattern. New launchers `train_hist_lora_sdxl.slurm` and `infer_a5_sdxl.slurm`
(a separate file from the A4-SDXL launcher, mirroring how `infer_a5_full.slurm`
is separate from `infer_a4_lcm.slurm` on SD1.5, so the already-validated A4
launcher is never touched).

**Training (job 44665, rank 32, 3000 steps, COMPLETED 43:03):** 5-step smoke
test passed first (finite loss, checkpoint saved). Full run: loss stays in
the expected noisy 0.10–0.21 range throughout (no clean downward trend —
consistent with this project's standing note that per-step diffusion loss
isn't a success signal). Checkpoint `lora/hist_r32_sdxl/final/
pytorch_lora_weights.safetensors` verified real (186MB, proportionally larger
than SD1.5's 25.5MB `hist_r32` given ~46.4M trainable params, 1.777% of
SDXL's UNet, vs SD1.5's smaller LoRA).

**Inference smoke test (job 44676, 2 frames, COMPLETED):** log confirms
`"LoRA + ControlNet(...) + Hist-LoRA + LCM-LoRA ..."` — the 3-way adapter
composition loads and runs successfully, proving the `--hist-lora` extension
works end-to-end without regressing the existing 2-way A4-SDXL path.

**Full held-out evaluation (job 44682 inference + job 44699 scoring, all 496
crops, strength 0.20, 8-step LCM — same operating point as `a4_sdxl` for a
direct comparison, COMPLETED):**

| Scope | A4-SDXL SSIM | A5-SDXL SSIM | A4-SDXL Δlab | A5-SDXL Δlab | Δ (A5 − A4) |
|---|---|---|---|---|---|
| ALL (pooled) | 0.5272 | 0.5263 | −5.60 | −5.30 | +0.30 |
| A06 | 0.3856 | 0.3850 | −3.58 | −3.55 | +0.03 |
| A08 | 0.5622 | 0.5604 | −5.92 | −5.54 | +0.39 |
| A09 | 0.5136 | 0.5127 | −5.19 | −4.92 | +0.28 |
| A13 | 0.5096 | 0.5093 | −4.75 | −4.19 | +0.56 |
| A16 | 0.5746 | 0.5739 | −5.10 | −4.86 | +0.24 |

**Reading**: adding the histopathology warm-start LoRA gives a small,
*consistent* colour-recovery improvement on every slide (+0.24 to +0.56 LAB
units) — real, not noise, since it moves the same direction everywhere. But
it's a fundamentally different result from SD1.5's A5:
- SD1.5's A5 win was concentrated on A06 specifically (+2.6 to +8.7 LAB
  units, its single best result project-wide). On SDXL, **A06 barely moves
  at all** (+0.03) — the exact slide where the hist prior helped most on
  SD1.5 is where it helps least here.
- **No sign flip anywhere** — colour recovery stays negative on every slide,
  just slightly less negative. This doesn't rescue P3-04's finding: SDXL's
  frozen-base colour drift (−3.5 to −5.9) is a much larger effect than what a
  histopathology prior can nudge (~+0.3).
- **Structure (SSIM) is essentially unchanged** — differences are all in the
  4th decimal, within noise.
- **Bottom line**: a genuine, measurable positive result on colour recovery,
  but it does not change SDXL's overall verdict from P3-04 — still below the
  classical baselines on structure (0.526 vs 0.628–0.681), still
  colour-negative on every slide.

## P3-06 — Transfer P1-10 (source-conditioned colour LoRA + fresh ControlNet) to SDXL
**Status:** 🔄 IN PROGRESS (2026-08-23).
**Source:** `sec:phase3_sdxl`'s "transfer the best-performing SD1.5 configuration"
principle — P3-03/P3-05 transferred A4/A5 because those were the best SD1.5
configs *when Phase 3 started*. Since then, `tickets/PHASE1-TICKETS.md` P1-10
(colour LoRA trained jointly with a fresh, trainable 6-channel
source-conditioning ControlNet, so the source image genuinely participates in
training, not just the img2img starting latent) + P1-11 (DDIM-inversion
inference on top of P1-10) have become the actual best SD1.5 result in the
project (SSIM 0.4960/0.4485, positive colour recovery on every held-out
slide) — clearly ahead of A4-SD1.5 and of A4/A5-SDXL's negative-colour-drift
finding (P3-04). This ticket transfers that architecture, not A4/A5 again.

**Scope: P1-10 only** (colour LoRA + fresh source-conditioned ControlNet,
plain 50-step DDIM, A2H direction only — H2A stays out of scope, matching
P3-03/P3-05's own descope). P1-11's DDIM-inversion-on-SDXL is a separate,
later ticket (P3-07, not started), once this base transfer is validated.

**Design (forks and merges this project's own already-validated SDXL-port and
source-conditioning patterns — no novel SDXL plumbing):**
- SDXL adaptation recipe from `src/train/train_colour_lora_sdxl.py` (P3-03):
  dual tokenizers/text encoders concatenated for `encoder_hidden_states`,
  `text_encoder_2` pooled output + `add_time_ids` for `added_cond_kwargs`,
  VAE forced fp32 kept outside the autocast region (SDXL's official VAE NaNs
  under fp16 — never "fix" this back), `unet.enable_gradient_checkpointing()`,
  `StableDiffusionXLPipeline.save_lora_weights` for checkpointing.
- Source-conditioning recipe from `src/train/train_colour_translation_lora.py`
  (P1-10, SD1.5): `ControlNetModel.from_unet(unet, conditioning_channels=6)`
  (fresh, zero-initialised conditioning encoder + zero-conv output layers;
  only these + the LoRA are trainable, the cloned backbone stays frozen),
  6-channel source-RGB+Canny conditioning tensor (`canny.
  extract_canny_control_image`), conditioning-only spatial jitter, leak-checked
  `build_pairs()` (A03/H03 only), leave-one-frame-out val split with
  best-val-loss checkpoint selection.
- SDXL inference-side recipe from `src/eval/infer_colour_lora_sdxl.py`
  (P3-03): fp32 `AutoencoderKL` passed into
  `StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(...,
  vae=vae_fp32, torch_dtype=torch.float16)` — the pipeline call handles
  dual-encoder prompt embedding internally at inference (unlike training).
- Ablation controls from `src/eval/infer_colour_translation.py` (P1-10,
  SD1.5): `--source-mode {correct,zero,shuffled}`, `--vae-only` floor
  control, the exact 6-channel control-tensor construction (must match
  training bit-for-bit), hash-derived per-crop seed protocol.

**New files:**
- `src/train/train_colour_translation_lora_sdxl.py` -- merge point requiring
  genuinely new wiring (not copy-paste from either parent): `run_step()`'s
  `controlnet(...)` AND `unet(...)` calls both need
  `added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids}` (P1-10's
  SD1.5 script passes neither; P3-03's SDXL script passes it to `unet()` only,
  no ControlNet to also thread it through). Checkpoints ->
  `lora/a2h_cond_r8_sdxl/` (new, isolated namespace).
- `src/eval/infer_colour_translation_sdxl.py` -- manifest schema unchanged
  (`seed, source_mode, crop_id, slide, frame, x, y, output_path,
  reference_path, aperio_path`), so `score_p1_10_ablation.py` scores it with
  zero changes. Output -> `eval/a2h_cond_r8_sdxl/`.
- `slurm/train_colour_translation_lora_sdxl.slurm`,
  `slurm/infer_colour_translation_sdxl.slurm` -- mirror the existing SD1.5
  launchers' conventions (bigbatch, fail-fast CUDA guard, bad-node
  `--exclude` reminder, mandatory smoke -> overfit-8 -> ablation staging
  before any real run).

**Not touched:** `train_colour_translation_lora.py`, `infer_colour_translation.py`,
`train_colour_lora_sdxl.py`, `infer_colour_lora_sdxl.py` (all four stay exactly
as validated), `score_outputs.py`/`score_p1_10_ablation.py` (already confirmed
resolution/backbone-agnostic), every existing checkpoint/eval directory.

**Implementation (2026-08-23):** `src/train/train_colour_translation_lora_sdxl.py`,
`src/eval/infer_colour_translation_sdxl.py`,
`slurm/train_colour_translation_lora_sdxl.slurm`,
`slurm/infer_colour_translation_sdxl.slurm` written per the design above.
Local verification passed: `python -m py_compile` clean on both new scripts,
`--help` on both confirms every intended flag/default present, `bash -n` clean
on both new launchers. Synced to the cluster (2026-08-23/24).

**Smoke test (2026-08-24), job 45773, `mscluster50`, COMPLETED 01:34.**
Confirms the one genuinely new wiring point: SDXL's `ControlNetModel.forward`
accepts `added_cond_kwargs={"text_embeds", "time_ids"}` on this diffusers
version (0.39.0) exactly like `UNet2DConditionModel.forward` does -- no
error, 5 steps ran cleanly, finite loss throughout (0.0699/0.3450/0.4711/
0.0061/0.2327, no NaNs), val_loss 0.0882, checkpoint (LoRA + ControlNet)
saved to `lora/a2h_cond_r8_sdxl/final` and `/best`. Params: LoRA (rank 8)
11,612,160 trainable; ControlNet 7,647,552 trainable (conditioning encoder +
zero-conv output layers) / 1,243,367,040 frozen (cloned SDXL UNet backbone) --
proportionally similar split to P1-10's SD1.5 ControlNet, scaled to SDXL's
larger UNet. Only stderr output: a harmless `torch.cuda.amp.GradScaler`
deprecation warning (scaler is disabled anyway under the bf16 default).
5-step smoke ran in ~14s -- SDXL+ControlNet step time is not yet
representative of the real training-run pace; will re-check from the
overfit-8 control's timing before setting the full run's `--time`.

**Overfit-8 control, 300 steps (2026-08-24), job 45845, `mscluster82`,
COMPLETED 5:18.** Mechanics correct, no NaN, checkpoint saved. Loss did not
show a clear downward trend (0.15-0.22 range throughout) -- **this exact
pattern already has precedent**: P1-10's own SD1.5 overfit-8 control at 300
steps (job 44309) showed the identical flat 0.15-0.26 range and was
diagnosed as under-training, not a broken mechanism (ControlNet's
zero-initialised layers start at literally zero output and need more than
300 steps on 8 pairs to develop measurable influence -- see
`tickets/PHASE1-TICKETS.md` P1-10). Extending to 2000 steps (SD1.5's job
44381) resolved it there: a real downward trend appeared and the
source-conditioning ablation then passed decisively. Applying the same fix
here before running the ablation control at 300 steps, which would likely
fail for the same under-training reason rather than a genuine SDXL-specific
problem.

**Extended overfit-8 control, 2000 steps (2026-08-24), job 45952,
`mscluster53`, COMPLETED 36:24.** No NaNs, checkpoints saved at 250-step
intervals plus final. Loss trend (first-20% mean 0.1792 vs last-20% mean
0.1507, ~16% reduction) is real but noticeably **milder** than SD1.5's own
extended overfit test (job 44381: roughly halved, 0.19-0.26 -> 0.09-0.18) --
plausibly slower ControlNet learning nested inside SDXL's much larger
(~3x) UNet at the same LR/step count, or just noisier on 8 pairs. Per this
project's own standing guardrail, per-step diffusion loss is not a success
signal either way -- the actual mandatory control is the source-conditioning
ablation (`correct` must beat `zero`/`shuffled` on structural/content
metrics), same as SD1.5's resolution. Proceeding to that ablation on this
2000-step checkpoint next, rather than reading the milder loss trend as a
verdict on its own.

**Source-conditioning ablation -- PASSES (2026-08-24), jobs 46004/46005/46006
(infer, `correct`/`zero`/`shuffled`, all COMPLETED, 24 crops each) + job
46018 (score, `score_p3_06_ablation.slurm` -- new one-off scorer, output
dirs sit directly under `eval/`, not the `eval/p1_10_ddim_inversion/<tag>`
convention `score_p1_10_ddim_inversion.slurm` assumes; first attempt, job
46015, reused that script and correctly found 0 rows against the wrong
path):**

| mode | SSIM (mean ± spread across 3 seeds) | LAB total |
|---|---|---|
| **correct** | **0.1554 ± 0.0039** | **25.04** |
| zero | 0.0658 ± 0.0010 | 38.22 |
| shuffled | 0.0543 ± 0.0017 | 33.93 |

Paired win-rate: `correct` wins 6/8 crops vs `zero`. `correct` beats both
controls by >2x on SSIM, far past the ~0.004 seed-to-seed noise floor, and
LAB is clearly lowest too -- decisive separation, the ControlNet branch is
genuinely being used, not ignored. **Remarkably close to SD1.5's own P1-10
2000-step ablation result** (`tickets/PHASE1-TICKETS.md` P1-10: correct SSIM
0.1466/LAB 23.95, zero SSIM 0.0620/LAB 37.24, shuffled SSIM 0.0467/LAB
27.50) -- SDXL's `correct` SSIM (0.1554) is even marginally higher. The
milder loss-curve trend noted above did not translate into a weaker
mechanism; confirms the "loss curve is not a success signal, ablation
separation is" reading. **P3-06's core mechanism is validated on the
overfit-8 set.**

**Full training run submitted (2026-08-24), job 46025.** All 50 A03/H03
pairs (39 train / 11 val, matching P1-10's SD1.5 split exactly), rank 8,
4000 steps -- same step count as P1-10's real run (job 44445) for direct
comparability. `--time=03:00:00`, extrapolated from the extended overfit
run's measured ~1.07s/step (job 45952: 2000 steps in 2170.8s) plus periodic
checkpoint-save overhead -- ~85min central estimate, doubled for margin
after job 44858's earlier TIMEOUT lesson (see `tickets/PHASE1-TICKETS.md`
P1-11) that estimates can run optimistic.

**COMPLETED (2026-08-24), `mscluster43`, ~80:20 wall time.** Val loss
converges cleanly and near-monotonically: 0.0853 -> 0.0852 -> 0.0814 ->
0.0802 -> 0.0757 -> 0.0736 -> 0.0756 -> 0.0713 -> 0.0705 -> 0.0701 -> 0.0694
-> 0.0696 -> 0.0689 -> **0.0676 -> 0.0676 -> 0.0671 (best, final step)** --
same clean-convergence pattern P1-10's SD1.5 full run showed (vs. the
noisier overfit-8 plateau), consistent with the model having genuine
diverse signal to learn from rather than memorising 8 pairs. Plateaus
somewhat higher than SD1.5's equivalent (0.0671 vs 0.0482) -- not
comparable in absolute terms across backbones/resolutions, noted for the
record only. `lora/a2h_cond_r8_sdxl/best` saved (== `final` here, best
val_loss landed on the last step).

**Held-out inference smoke test (2026-08-24), job 46190 (infer, `mscluster`,
COMPLETED, 96 crops = 32 locations x 3 seeds) + job 46214 (score, new
generic `score_p3_06.slurm`, COMPLETED)**, on `lora/a2h_cond_r8_sdxl/best`,
strength 0.50, 50-step DDIM (script defaults, matching P1-10's own mandated
staged-rollout operating point). `LIMIT=8` grabbed only A06 frames (first in
`heldout_frames.csv`) -- **A06-only, the confirmed colour-gap outlier, small
sample** -- read with that caveat, not a real multi-slide result:

| | SD1.5 P1-10, A06 (full 179-crop, job 44542) | SDXL P3-06, A06 (smoke, 96-crop) |
|---|---|---|
| SSIM | 0.3067 | **0.2661** (lower) |
| LAB recovery Δlab | not isolated per-slide in the SD1.5 write-up | wlab_mean 91.44 vs baseline_summary.csv's A06 raw 94.92 -> **+3.48** (positive) |

SDXL's structural fidelity on A06 trails SD1.5's at the same operating
point; colour recovery looks genuinely positive but isn't directly
comparable (no equivalent SD1.5 A06-only figure on record). Single small
sample on the one known outlier slide -- not yet a verdict.

**Full held-out inference submitted (2026-08-24), job 46226.** All 5 slides,
496 crops x 3 seeds = 1488 outputs, `correct` source-mode, strength 0.50,
50-step DDIM, on `lora/a2h_cond_r8_sdxl/best`. User opted to proceed
straight to the full run rather than an intermediate A08-only smoke check
(A06 vs typical-slide divergence risk noted and accepted, not investigated
further first). `--time=06:00:00`, extrapolated from job 46190's measured
~9.4s/crop (00:17:58 for 96 crops, minus ~2-3min load/registration overhead)
-> ~4hr central estimate for 1488 crops, extra margin given job 44858's
earlier SD1.5 TIMEOUT lesson. **COMPLETED (2026-08-24/25), `mscluster`, 02:33:51.** Scored via job 46360
(`score_p3_06.slurm full_heldout p3_06_full_heldout`, `score_p1_10_ablation.py`)
+ job 46370 (`aggregate_p3_06_full.slurm`, the existing unmodified
`aggregate_p1_10_full.py` -- confirmed fully generic over backbone,
zero changes needed, same as it already was for SD1.5).

**Final SDXL-vs-SD1.5 comparison, same operating point (strength 0.50,
50-step DDIM, `correct` source-mode, 496 crops x 3 seeds both sides):**

| | SD1.5 P1-10 (job 44542) | SDXL P3-06 (job 46226) |
|---|---|---|
| ALL SSIM | **0.4485** | 0.3920 |
| ALL wLAB | 31.60 | 33.04 |
| ALL_excl_outliers SSIM | **0.4695** | 0.4120 |
| ALL_excl_outliers wLAB | 22.81 | 24.24 |
| Recovery Δlab (pooled) | **+2.89** | +1.22 |
| Recovery Δlab (excl outliers) | -- | +1.61 |
| A06 SSIM (outlier) | 0.3067 | 0.2567 |
| A08 SSIM | 0.4815 | 0.4101 |
| A09 SSIM | 0.4175 | 0.3718 |
| A13 SSIM | 0.4581 | 0.4041 |
| A16 SSIM | 0.4968 | 0.4406 |

**Verdict: SDXL underperforms SD1.5 on this exact same P1-10 architecture.**
Every single slide's SSIM is lower on SDXL, not just the pooled figure --
this is a real, consistent structural-fidelity gap, not an artefact of one
outlier slide. Colour recovery is positive on every slide on SDXL too
(A06 +2.70, A08 +1.74, A09 +1.03, A13 +2.31, A16 +1.58 -- no sign flips),
confirming the source-conditioning mechanism genuinely transferred and
works, consistent with the ablation pass above -- but the magnitude is
meaningfully weaker than SD1.5's (+1.22 pooled vs +2.89, roughly 60% lower).

**Reading, in context of this project's other SDXL findings:** this is a
different, more nuanced result than A4-SDXL's (P3-04: colour recovery
negative on every slide, a sign flip). P1-10's fresh, jointly-trained
source-conditioning branch is more robust on SDXL than A4's pretrained-
ControlNet-plus-LoRA approach was -- it does not break in the same way. But
it is still not an improvement over SD1.5 at this operating point: bigger
backbone does not mean better result here, consistent with SDXL's general
underperformance pattern throughout this project (P3-04's A4 finding, and
now this). **Open question, not yet checked**: SD1.5's own SSIM ceiling was
shown to be capped by its VAE-only floor (0.5393, no denoising at all) --
SDXL's equivalent VAE-only floor has not been measured for this ticket, so
it's unknown whether SDXL's own VAE imposes a lower structural ceiling
(which would partly explain the SSIM gap architecturally) or whether the
gap is purely about how well the conditioning is learned. Not run without
checking in first.

**Status: P3-06 core transfer + comparison complete.** No untested
strength/step sweep run yet (P1-10's own strength=0.50 was the mandated
staged-rollout point, not a tuned optimum -- unclear whether SDXL has more
headroom at another operating point, mirroring the open question left after
P3-04's A4-SDXL strength sweep). P1-11-on-SDXL (DDIM inversion) remains
out of scope for this ticket, per the original scope decision above.

**Downstream-classifier extension (2026-08-28, P2-09's atypia_r18
checkpoint, job 47418):** recovery delta (accuracy vs. raw_hamamatsu,
excl. A06) = **+0.0072** -- weak, essentially tied with P1-10/P1-11, well
below every classical baseline, despite P3-06's positive (if smaller than
SD1.5's) colour recovery. Full comparison table and caveats: `tickets/
PHASE2-TICKETS.md` P2-09's Extension section.
**STALE (2026-08-28): A->H, not H->A -- see `tickets/
P2-12_atypia_classifier_evaluation_hardening.md` §0. Not valid clinical-
utility evidence until P2-12 lands.**

## P3-07 — P3-06 at native 1024×1024 resolution (retargets P3-03b onto P1-10)
**Status:** 🔄 IN PROGRESS (2026-08-25).
**Source:** `sec:phase3_sdxl`; retargets the already-drafted-but-never-started
**P3-03b** contingency (native-1024 SDXL training) onto P1-10/P3-06's
architecture instead of A4 -- P3-03b was written when A4 was the transfer
target; P1-10/P3-06 has since become the relevant SD1.5 baseline. Also
motivated by P3-06's own open question: whether SDXL's SSIM/colour-recovery
gap vs SD1.5 (ALL SSIM 0.3920 vs 0.4485, Δlab +1.22 vs +2.89) is partly
because P3-06 trained at 512×512 with *honest* micro-conditioning (not
spoofed 1024) on a backbone predominantly pretrained at ≥1024px -- i.e. an
under-resolution-training artefact, not a ceiling on the architecture itself.

**Hard scope constraint (methodology guardrail, corrected 2026-08-25 after
an initial mis-scoped draft -- see below):** H1/RQ1 in `docs/proposal_draft(6).tex`
is a literal, capped hypothesis, not just an informal "small dataset"
description: *"≤50 coordinate-corresponding A03/H03 crop pairs are
sufficient to train the scanner-colour LoRA... trained on ≤50 examples from
**one slide pair** and applied to five different slide pairs without
retraining."* The contingency table's only sanctioned fallback if ≤50 proves
insufficient is *"expand overlapping crops from the same A03/H03 pair while
reporting the minimum viable pair count"* -- still one slide pair, not new
slides. **This ticket stays within that scope: A03/H03 only, ≤50 pairs,**
matching P1-10/P3-06's existing training budget exactly so resolution is the
only variable that changes.

**Correction note:** an earlier draft of this plan proposed ~90-96 pairs
(all non-overlapping 1024 candidates available from A03 alone) and floated
an 11-slide multi-slide expansion (up to 1195 candidates, using the other
10 training slides already reserved in the proposal's own data table for
"A5 hist LoRA pool only" -- a different, unpaired purpose) as a "Phase B."
Both would have exceeded H1's literal ≤50/one-slide-pair cap. Caught before
any extraction ran. If a ≤50-pair 1024 result underperforms, the next step
is the proposal's own sanctioned fallback (`--overlap` > 0 on A03/H03 only,
reporting the minimum viable pair count) -- not multi-slide expansion, which
would need to be explicitly flagged as a deliberate departure from H1, not
a default escalation.

**Candidate crop counts, measured (2026-08-25, job 46384 on `stampede`,
read-only, via `extract_pairs.py`'s own `candidate_boxes()`/
`tissue_fraction()`, no files written/modified):** A03 alone has 96
tissue-passing non-overlapping 1024×1024 candidates (24 x20 frames x ~4
avg) -- comfortably above the 50-pair budget, so no overlap or extra data
is needed to hit ≤50 at 1024. (Raw MITOS-ATYPIA frames are only
~1539x1376 (Aperio) -- ~1.5x a 1024 crop, vs ~3x at 512, hence the lower
per-frame yield: ~4 candidates/frame at 1024 vs ~9 at 512, matching
P3-03b's original note.) A process lesson from this measurement: the first
attempt was run directly over `ssh` instead of `sbatch`, landing on the
**login node** -- caught mid-run by a direct question, killed, and
resubmitted correctly on `stampede` (`slurm/measure_1024_candidates.slurm`).

**Plan:**
1. `extract_pairs.py --root data/mitos --crop 1024 --max-pairs 50 --out pairs/train_1024`
   -- run locally (matches how `pairs/train` at 512 was originally produced;
   the paired training-side raw data, both Aperio and Hamamatsu, exists only
   locally in `data/mitos/`, not on the cluster -- the cluster's
   `mitos_atypia_train_aperio/` is Aperio-only, fetched for P2-09). New
   sibling folder, `pairs/train` never touched.
2. Upload `pairs/train_1024` to the cluster (`/datasets/mhoosen/stain-norm/pairs/train_1024/`).
3. Reuse `train_colour_translation_lora_sdxl.py`/`infer_colour_translation_sdxl.py`
   completely unchanged (already generic over `--resolution`/`--crop`) --
   only the launcher's `DATA_DIR`/`--resolution 1024` need overriding, no
   new Python code. Main open risk: GPU memory at 1024 (4x the latent
   tokens vs 512) on a 24GB 3090 -- `ControlNetModel.enable_gradient_checkpointing()`
   is confirmed available in the installed diffusers if the plain smoke
   test OOMs with only `unet.enable_gradient_checkpointing()` (P3-06's
   existing setup).
4. Same staged discipline as P3-06: smoke test -> overfit-8 control
   (escalate 300->2000 steps if it plateaus flat, per precedent) ->
   source-conditioning ablation -> full training run -> held-out inference
   at `--crop 1024` -> score -> compare against P3-06's 512 result (SSIM
   0.3920 pooled, Δlab +1.22) and SD1.5's P1-10 (SSIM 0.4485, Δlab +2.89).

**Extraction complete (2026-08-25).** `extract_pairs.py --root data/mitos
--crop 1024 --max-pairs 50 --seed 0 --skip-heldout` run locally (Python
3.10, Pillow 12.3.0, numpy 2.2.6 -- the paired A03/H03 raw data exists only
locally, matching the extraction of the original 512 set). Matched exactly
the read-only measurement: 96 candidates found, 50 selected (capped),
written to `pairs/train_1024/` (100 files = 50 pairs x 2 sides) +
`pairs/train_1024_manifest.csv` + `pairs/train_1024_extract_config.json` --
new siblings, `pairs/train`/`pairs/train_manifest.csv` untouched.
Spot-verified: `A03_00A_c000_{aperio,hamamatsu}.png` both exactly
1024x1024 RGB. Uploaded to
`/datasets/mhoosen/stain-norm/pairs/train_1024/` (verified: 100 files
present on the cluster). `--skip-heldout` used since `heldout_frames.csv`
is crop-size-independent (raw frame paths, not crops) -- the existing one
in `pairs/` is reused unchanged.

**Smoke test (job 46409, COMPLETED).** 5 steps, `RESOLUTION=1024
PAIRS_DIR=pairs/train_1024`, A2H, rank 8 -- no OOM on the 24GB 3090 (the
flagged main risk did not materialise; `unet.enable_gradient_checkpointing()`
alone was sufficient, `ControlNetModel.enable_gradient_checkpointing()`
fallback not needed), no NaNs, val_loss 0.0855 computed, checkpoint saved.

**Overfit-8 control, 300 steps (job 46471, COMPLETED) then extended to 2000
steps (job 46512, COMPLETED).** Same flat-loss-plateau pattern already seen
in both P1-10 (SD1.5) and P3-06 (SDXL@512) at 300 steps -- loss oscillates
0.11-0.21 with no visible downward trend even at 2000 steps. Per this
project's standing guardrail, per-step MSE is not a success signal; proceeded
to the mandatory ablation rather than reading anything into the flat curve.

**Source-conditioning ablation (jobs 46529/46530/46531 correct/zero/shuffled,
COMPLETED; scored job 46543) -- passed decisively:**

| Mode | LAB total | ΔE2000 | SSIM | Win-rate |
|---|---|---|---|---|
| **correct** | **24.32** | **15.93** | **0.129** | **7/8 crops** |
| shuffled | 34.39 | 18.57 | 0.046 | 1/8 |
| zero | 62.13 | 19.98 | 0.062 | 0/8 |

`correct` beats both controls by a wide margin on every metric, confirming
the ControlNet branch is genuinely used at 1024 resolution, not ignored --
same conclusion as P1-10 and P3-06's own equivalent ablations.

**Bug found and fixed during this ablation:** `zero` mode initially crashed
(`RuntimeError: size of tensor a (128) must match size of tensor b (64) at
non-singleton dimension 3` inside `ControlNetModel.forward`). Root cause:
`infer_colour_translation_sdxl.py`'s zero-conditioning branch built
`torch.zeros(1, 6, args.crop, args.crop, ...)` using the `--crop` CLI default
(512) instead of the actual target crop's shape -- silently correct at every
prior resolution only because 512 (the default) always matched the real data
before P3-07. Fixed to derive the zero tensor's shape from
`target_src_rgb.shape[:2]` directly; re-synced and the `zero` ablation job
(46534) then completed cleanly. Generic fix, not resolution-specific --
applies to `zero` mode at any resolution mismatch, not just 1024.

**Full training run, 4000 steps (job 46552, COMPLETED).** Same step count as
P3-06 for direct comparability. val_loss trended down over the run (0.0654 ->
0.0663 -> 0.0638), no NaNs/crashes. Checkpoint: `lora/a2h_cond_r8_sdxl_1024/final`.

**Held-out smoke (job 46635, COMPLETED, A06-only -- `LIMIT=8` grabs A06's
frames first in `heldout_frames.csv`, same caveat as P3-06's own smoke) --
scored job 47011:**

| | P1-10 (SD1.5) A06, full | P3-06 (SDXL@512) A06, smoke | P3-07 (SDXL@1024) A06, smoke |
|---|---|---|---|
| SSIM | 0.3067 | 0.2661 | **0.3077** |
| Recovery Δlab | -- | +3.48 | **-2.50** |

SSIM already close to SD1.5's own A06 number; colour recovery flipped
slightly negative on this small (93-crop) sample -- opposite sign from
P3-06's A06-only smoke. Per this project's standing caveat that A06-only
smoke samples are unreliable predictors of the full-slide result, went
straight to the full run rather than reading too much into 8 frames.

**Full 496-crop x 3-seed held-out result (job 47029 inference, COMPLETED;
scored job 47236 after the first scoring attempt (47226) timed out at its
30-minute default -- 1024x1024 crops take ~4x the metric-computation time of
512x512, resubmitted with `--time=02:00:00`; aggregated job 47260):**

| | SD1.5 P1-10 | SDXL P3-06 (512) | **SDXL P3-07 (1024)** |
|---|---|---|---|
| ALL SSIM | 0.4485 | 0.3920 | **0.4313** |
| ALL_excl_outliers SSIM | 0.4695 | 0.4120 | **0.4485** |
| A06 SSIM (outlier) | 0.3067 | 0.2567 | **0.3128** |
| A08 SSIM | 0.4815 | 0.4101 | 0.4572 |
| A09 SSIM | 0.4175 | 0.3718 | 0.3964 |
| A13 SSIM | 0.4581 | 0.4041 | 0.4311 |
| A16 SSIM | 0.4968 | 0.4406 | 0.4807 |
| ALL recovery Δlab | **+2.89** | **+1.22** | **-5.60** |
| A06 / A08 / A09 / A13 / A16 Δlab | +2.70/+1.74/+1.03/+2.31/+1.58 | (all positive, P3-06 ticket) | **-3.48/-5.54/-6.64/-4.81/-4.89** |

**Verdict -- a genuine trade-off, not a wash.** Native 1024 resolution
closes most of SDXL's structural-fidelity gap versus SD1.5 (every slide's
SSIM improves meaningfully over P3-06's 512 result; A06 SSIM now *exceeds*
SD1.5's own P1-10 number, 0.3128 vs 0.3067) -- confirming this ticket's
motivating hypothesis that part of P3-06's SSIM deficit was an
under-resolution-training artefact. But colour recovery, positive on every
slide for both P1-10 and P3-06, **reverses to negative on every single
slide** at 1024 -- not a pooled-outlier artefact, consistent in direction and
magnitude across all five slides (-3.48 to -6.64). This is the same failure
signature as A4-SDXL (model actively moves colour away from the target
scanner, worse than doing nothing), even though the ablation confirms the
source-conditioning mechanism itself works correctly at this resolution.

**Working hypothesis (untested, open question):** at the same nominal
strength=0.50 operating point, a higher-resolution encoding may correspond to
a smaller *effective* per-pixel change for the same nominal strength --
structurally conservative in a way that parallels why A4-SDXL's low-strength
point was colour-negative -- leaving less room for the colour LoRA's effect
to manifest even though it is genuinely being applied. Not yet investigated:
whether a strength sweep at 1024 (mirroring A4/A4-SDXL's own strength
investigations) finds an operating point that keeps 1024's structural gain
while restoring positive colour recovery.

**Status: P3-07 held-out evaluation COMPLETE (2026-08-27).** Structural
hypothesis confirmed; colour-recovery regression is a new, decisive, and
unresolved finding. Next step (not yet started, needs explicit go-ahead) is
either a strength sweep at 1024 to search for a colour-recovery-positive
operating point, or accepting the trade-off and closing this ticket as-is.

**Reopened (2026-08-27): -5.60 recovery is too systematic to close on.** Not
treated as a broken implementation -- the source-conditioning ablation above
already confirms the ControlNet branch is genuinely used at 1024 (correct
24.32 / shuffled 34.39 / zero 62.13 LAB). The open question is *why* structure
improves on every slide while colour recovery flips negative on every slide
at the same time. Revised: the "smaller effective per-pixel change at
strength=0.50 at higher resolution" idea in the working hypothesis above is
NOT established -- the scheduler still uses the same nominal fraction of the
denoising trajectory regardless of resolution -- and should not be assumed
before D2/D3 below test it mechanistically.

Diagnostic plan, in strict order, before any retraining, strength sweep,
rank change, or pair-count change:

- **D1 -- baseline/metric-artefact check.** `aggregate_p3_07_full.slurm` and
  `aggregate_p3_06_full.slurm` both diff their model output against the SAME
  `pairs/baseline_metrics/baseline_summary.csv`, which `score_baseline.slurm`
  generates with a hardcoded `--crop 512` (metrics.py `run_baseline()`). So
  P3-07's reported recovery is raw-Aperio-@512-vs-real-Hamamatsu-@512 MINUS
  model-output-@1024-vs-real-Hamamatsu-@1024 -- a resolution-mismatched
  baseline (this ticket's H5), not yet ruled out before the -5.60 figure was
  first reported. New dedicated diagnostic script `src/eval/d1_verify_p3_07.py`
  (+ `slurm/d1_verify_p3_07.slurm`, stampede, read-only) re-registers every
  unique held-out frame P3-07's own `eval_manifest.csv` used and recomputes
  the raw baseline at native 1024 crop size (same `register_h_to_a` +
  `score_aligned_pair` calls `run_baseline()` uses, just crop=1024), then
  reports recovery recomputed against that 1024-native baseline vs the
  originally-reported 512-baseline recovery, per-slide and ALL, plus an
  independent spot check on 10 random crops. **Submitted 2026-08-27, job
  47277 (stampede, no positional args) -- COMPLETED.**

  **D1 RESULT: H5 ruled out.** The 1024-native raw baseline is essentially
  identical to the old 512 baseline, per-slide and pooled (ALL 34.18 @1024
  vs 33.84 @512; A06 94.40 vs 94.84; A08 25.34 vs 25.28; A09 27.36 vs 27.49;
  A13 26.70 vs 26.64; A16 23.75 vs 23.78 -- resolution has no material effect
  on the raw-Aperio-vs-real-Hamamatsu LAB distance). Recovery recomputed
  against the resolution-matched 1024 baseline is **-5.25 ALL** (vs -5.60
  originally reported against the 512 baseline) -- same sign, same order of
  magnitude, every slide still negative (A06 -3.91, A08 -5.47, A09 -6.76,
  A13 -4.74, A16 -4.92). Reproducibility check passed exactly (495/495
  recomputed registered-Hamamatsu crops matched the saved `reference/` PNGs,
  MAE ≤ 1.0), so the independent recomputation is trustworthy. The 10-crop
  manual spot check independently confirms the same sign on every single
  sampled crop (raw always beats model, by 2.6 to 8.0 LAB). **Verdict: the
  negative colour recovery is a real model effect, not a baseline/metric
  artefact -- proceed to D2.**
- **D2 -- colour-LoRA-disabled comparison.** infer_colour_translation_sdxl.py
  gained a `--no-lora` flag (skips `pipe.load_lora_weights()`, keeps the
  trained 1024 ControlNet + source conditioning + frozen SDXL base
  identical). Run on a balanced A06+A08 held-out subset (not A06 alone) at
  the same operating point as the full run (strength=0.50, steps=50,
  guidance=2.0, crop=1024). Inference: job 47298 (bigbatch, TIMEOUT at
  1h30m, 303/513 outputs) -> resubmit job 47330 (`--time=03:30:00`, TIMEOUT
  again at the SAME 303 outputs -- turned out to be `mscluster61` running
  2.4x slower than the first attempt's node, not a hang; see Cluster facts
  in CLAUDE.md) -> resubmit job 47402 (`--time=10:00:00`,
  `--exclude=...,mscluster61`, **COMPLETED**, 2h11m, all 525/525 outputs).
  Scored + compared via `slurm/score_d2_p3_07.slurm` (job 47451, COMPLETED).

  **D2 RESULT: Case A confirmed.** No-LoRA recovery is *more negative* than
  WITH-LoRA on both slides -- A06 -6.92 (no-LoRA) vs -3.91 (with-LoRA); A08
  -8.94 vs -5.47; pooled A06+A08 -8.21 vs -5.25. The colour LoRA is not the
  cause and is not inert either -- it measurably pulls recovery back toward
  positive (~+3 LAB on both slides) but is too weak to overcome a stronger
  negative drift coming from the frozen SDXL base + trained ControlNet +
  source conditioning path itself. Case B (LoRA learned the wrong mapping)
  is ruled out. **Proceed to D3** to test whether source-RGB conditioning
  specifically (vs Canny-only) is what's preserving Aperio scanner
  appearance alongside morphology at native resolution.
- **D3 -- RGB vs Canny conditioning split.** infer_colour_translation_sdxl.py
  gained `--condition-mode {rgb_canny,rgb_only,canny_only}` (zeroes one half
  of the 6-channel condition post-construction, trained weights untouched).
  Run on the same A06+A08 subset, same operating point, trained colour LoRA
  ENABLED (D2 already isolated the LoRA question; D3 asks which
  conditioning channel the deployed model actually uses). Three parallel
  jobs (47505 rgb_canny, 47506 rgb_only, 47507 canny_only), all bigbatch,
  all COMPLETED (525/525 outputs each, ~2.1-2.5h). Scored + compared via
  `slurm/score_d3_p3_07.slurm` (job 47544, COMPLETED).

  **D3 RESULT: H1 not supported as stated -- a more specific finding.**
  `rgb_only` is nearly IDENTICAL to `rgb_canny` (unmodified) on both SSIM
  (0.4060 vs 0.4052 pooled) and recovery (-4.93 vs -4.91 pooled) -- the Canny
  half of the condition contributes almost nothing to this trained model's
  behaviour; it appears to have learned to rely on the RGB channels almost
  exclusively. `canny_only` (RGB zeroed) does NOT recover colour as H1
  predicted -- it collapses on BOTH axes: SSIM 0.1594 (vs 0.405, more than
  halved) and recovery -17.86 (vs -4.91, over 3.6x worse), with A08
  particularly severe (raw 25.34 -> model 51.92, recovery -26.58, vs A06's
  milder -2.37 since A06's raw gap is already ~94 LAB). So RGB conditioning
  is not a separable "preserves colour at zero structural cost" channel --
  it is carrying essentially all the useful signal (structural AND colour),
  and losing it doesn't free up colour recovery, it breaks the model.
  H1 is revised: it is not RGB-vs-Canny that explains the colour-recovery
  regression; the Canny half is largely inert at this checkpoint. **Points
  toward H2** (the RGB-driven base+ControlNet path needs that channel for
  structure, but the same channel also couples in source-appearance bias
  that the colour LoRA is too weak to overcome) as the leading remaining
  explanation. **Proceed to D4** (controlnet_conditioning_scale sweep) to
  test whether reducing overall ControlNet/RGB-conditioning influence (not
  swapping which half is used) restores positive recovery while retaining
  some of 1024's SSIM gain.
- **D4 -- small conditioning-balance sweep.** infer_colour_translation_sdxl.py
  gained `--val-frames-json` (filters `--pairs-dir` to exactly the frame_ids
  recorded as `val_frames` in the trained checkpoint's own
  `pair_manifest.json` -- the 5-frame/11-pair internal validation split from
  training, never the A06/A08/A09/A13/A16 held-out test set, per this
  ticket's "do not tune on the full held-out set" guardrail).
  `controlnet_conditioning_scale` in {0.25, 0.50, 0.75, 1.00} at fixed
  strength=0.50/steps=50/guidance=2.0, trained colour LoRA enabled,
  condition-mode=rgb_canny (trained default). Sweep job 47599 (bigbatch,
  COMPLETED, 29min, all 4 scales x 33 outputs). Scored + compared via
  `slurm/score_d4_p3_07.slurm` (job 47605, COMPLETED) -- `d4_verify_p3_07.py`
  computes the raw baseline directly from the 11 val pairs (unregistered,
  coordinate-corresponding training-domain crops -- lab_total/de2000 are
  valid per metrics.py's distributional definition, SSIM here is for
  relative cross-scale comparison only, not comparable to the registered
  held-out SSIM numbers elsewhere in this document).

  **D4 RESULT: no scale restores non-negative recovery.** raw baseline lab
  29.18; model lab_total ranges 32.73-33.56 across all 4 scales (recovery
  -3.55 to -4.38) -- no meaningful improvement even at the lowest
  conditioning scale tested (0.25: recovery -4.21, *worse* than 0.50's
  -3.55). SSIM moves the "wrong" direction for a pure tradeoff story too --
  it INCREASES monotonically with more ControlNet influence (0.1486 at 0.25
  -> 0.1852 at 1.00), not the expected structure-for-colour tradeoff.
  Reducing source-conditioning strength alone does not fix this at 1024.

Working hypotheses (in priority order, none yet confirmed):
H1 source-RGB conditioning too dominant at 1024, preserving Aperio colour as
well as morphology. H2 colour LoRA too weak relative to SDXL+ControlNet at
1024, exposing SDXL's already-observed negative colour bias (cf. A4-SDXL).
H3 the 1024 crops are a harder/more heterogeneous colour-learning problem
under the same ≤50-pair budget. H4 inference conditioning balance doesn't
transfer from 512 to 1024. H5 recovery aggregation uses a resolution-
mismatched (512) baseline -- see D1 above.

Explicitly NOT started: a new 4000-step training run, rank change, expanding
beyond 50 pairs, or adding new slides.

**D1-D4 diagnostic arc summary (2026-08-28): every mechanistic fix tested so
far fails to restore positive colour recovery.** D1 ruled out a
baseline/metric artefact -- the effect is real. D2 ruled out "the colour
LoRA learned the wrong mapping" -- the LoRA measurably helps (~+3 LAB) but
is fighting a stronger negative drift from the base+ControlNet+conditioning
path itself. D3 ruled out "RGB vs Canny channel selection" as the
explanation in the form H1 predicted -- RGB conditioning carries essentially
all the useful signal (structural AND colour together); Canny is nearly
inert at this checkpoint, and removing RGB breaks both axes rather than
trading one for the other. D4 ruled out "just reduce ControlNet conditioning
scale" -- no scale in {0.25, 0.50, 0.75, 1.00} restores non-negative
recovery on internal validation, and SSIM moves the wrong direction for a
clean tradeoff story (more conditioning = better SSIM, not worse).

**D5 -- colour-LoRA-scale sweep** (final bounded inference-only test before
considering training-data expansion). infer_colour_translation_sdxl.py
gained `--lora-scale`, applied via `pipe.set_adapters(["colour"],
adapter_weights=[scale])` -- the same mechanism already established in
`infer_colour_lora_sdxl.py` for `--lcm-scale`/`--hist-scale`, here given a
custom weight for the single colour-LoRA adapter instead of the implicit
1.0 default. Swept `lora_scale` in {1.0, 1.25, 1.5, 2.0} at fixed
strength=0.50, controlnet_conditioning_scale=1.0, guidance=2.0, 50-step
DDIM, condition-mode=rgb_canny, on the SAME 11-pair internal validation
split D4 used. Sweep job 47646 (bigbatch, COMPLETED, 36min, all 4 scales x
33 outputs). Scored + compared via `slurm/score_d5_p3_07.slurm` (job 47710,
COMPLETED) -- `d5_verify_p3_07.py` reports SSIM, LAB total, windowed LAB,
ΔE2000, and recovery Δlab per scale, plus applies the selection rule
(require recovery Δlab > 0 first; among positive configs pick highest SSIM).

**D5 RESULT: increasing LoRA scale makes recovery WORSE, monotonically, not
better.** raw baseline lab 29.18. Recovery by scale: 1.0 -> -4.38, 1.25 ->
-5.65, 1.5 -> -7.64, 2.0 -> -8.00 -- a clean monotonic degradation, not
noise. SSIM degrades in lockstep (0.1852 -> 0.1832 -> 0.1700 -> 0.1126). D2
showed the LoRA at its trained weight (1.0) genuinely pulls colour in the
correct direction relative to no-LoRA; D5 shows that amplifying that same
LoRA beyond its trained/calibrated weight does not extend the correction
further -- it pushes the output out of the regime the adapter was actually
trained for, degrading colour AND structure together. No configuration in
the grid achieves positive recovery, and the trend is not consistently
toward zero/positive (it moves away from zero) -- both conditions for the
second (strength x lora_scale) grid are unmet. **Per the diagnostic
protocol, stop inference-only tuning here.**

**D1-D5 diagnostic arc summary (2026-08-29): every mechanistic and
inference-only fix tested fails to restore positive colour recovery.**
Every inference-only knob available at this checkpoint has now been tried:
baseline artefact (D1), LoRA mis-mapping (D2, LoRA actually helps but is
too weak), RGB-vs-Canny channel selection (D3, RGB carries all the signal),
ControlNet conditioning scale (D4, no scale helps), and colour-LoRA
influence (D5, more influence makes it worse, not better).

**Proposed next step: P3-07b, a bounded >50-pair training follow-up
(NOT YET STARTED -- needs explicit go-ahead).** Per the remaining hypothesis
(H3: the native-1024 crops may be a harder/more heterogeneous colour-
learning problem than 512, and <=50 paired A03/H03 crops may simply be
insufficient to learn it), retrain the same architecture (rank-8 colour
LoRA + fresh 6-channel source-conditioned ControlNet, same SDXL base, same
4000 steps, same A03/H03-only scope) using all 96 available non-overlapping
A03/H03 1024 candidate pairs (measured in the original extraction job
46384) instead of the capped 50, with everything else held fixed --
architecture, rank, step count, slide scope, loss, no overlapping crops, no
new slides. **P3-07b must be explicitly labelled a supplementary >50-pair
follow-up, not evidence for or against the proposal's formal <=50-pair H1
claim** ("<=50 coordinate-corresponding A03/H03 crop pairs are sufficient...
trained on <=50 examples from one slide pair") -- P3-07 itself, at <=50
pairs, remains the H1-relevant result regardless of what P3-07b finds. Do
not add slides, sweep rank, add new losses, or use overlapping crops in
P3-07b.

**P3-07b IN PROGRESS (2026-08-29).** Extracted all 96 non-overlapping
tissue-passing A03/H03 1024 candidates locally (`extract_pairs.py --root
data/mitos --out pairs_1024_full96_staging --crop 1024 --max-pairs 96
--seed 0 --skip-heldout --tissue-thresh 0.3`), written to
`pairs/train_1024_full96/` (192 files) + `pairs/train_1024_full96_manifest.csv`.
**Verified byte-for-byte superset of P3-07's 50 pairs** -- same seed's
per-frame round-robin selection deterministically reproduces the original
50 as a prefix, plus 46 more (spot-checked all 50 aperio/hamamatsu
filenames match exactly). Uploaded to
`/datasets/mhoosen/stain-norm/pairs/train_1024_full96/` (192 files
verified). New dedicated training script `src/train/train_p3_07b_lora_sdxl.py`
(+ `slurm/train_p3_07b_lora_sdxl.slurm`) -- a new file per this ticket's own
instruction, not a parameterised reuse of `train_colour_translation_lora_sdxl.py`,
but line-for-line identical architecture/hyperparameters (rank 8, 4000
steps, resolution 1024, A2H, same leak check, same leave-one-frame-out val
split) -- only `--pairs-dir` differs. New checkpoint tag
`lora/a2h_cond_r8_sdxl_1024_full96/`, never touches `lora/a2h_cond_r8_sdxl_1024/`
(P3-07).

Smoke test (job 47734, COMPLETED, 3m15s): 96 pairs -> 24 frames -> 5 held
out for val (76 train / 20 val), ControlNet/LoRA trainable-param counts
identical to P3-07 (7,647,552 / 11,612,160), no NaNs, checkpoint saved
cleanly. **Full 4000-step run COMPLETED (job 47743, bigbatch, 1h52m58s,
exit 0).** val_loss trended down over the run to a final/best 0.0982
(no NaNs/crashes). Checkpoint: `lora/a2h_cond_r8_sdxl_1024_full96/{final,best}`.

Per this project's own standing discipline (every new checkpoint gets the
mandatory source-conditioning ablation before its held-out numbers are
trusted -- same requirement P3-06 and P3-07 each satisfied on their own
checkpoints even though the architecture is shared): **ablation PASSED**
(jobs 47807/47808/47809 correct/zero/shuffled, scored job 47820) --
`correct` LAB 25.25 decisively beats `shuffled` 35.94 and `zero` 57.49 on
every pooled metric, wins the SSIM win-rate 5/8 vs `zero`'s 3/8 (`shuffled`
0/8). ControlNet is genuinely used on the new checkpoint.

**Full 5-slide/3-seed held-out inference COMPLETED** (job 47846, bigbatch,
6h24m, 1485/1485 outputs -- first attempt job 47832 was cancelled and
resubmitted with a longer `--time` after measuring its throughput would
exceed the original 8h budget; mscluster57 both times, no bad-node issue).
Scored (job 47975, COMPLETED, 1h06m) and aggregated against the SAME
`pairs/baseline_metrics/baseline_summary.csv` P3-07 used (job 48390,
COMPLETED) -- D1 already confirmed this baseline is resolution-robust, so
the comparison to P3-07's own table is direct and valid.

**P3-07b RESULT -- colour recovery flips positive on every single slide.**

| | P3-07 (<=50 pairs) | **P3-07b (96 pairs, supplementary)** |
|---|---|---|
| ALL SSIM | 0.4313 | **0.4416** |
| ALL_excl_outliers SSIM | 0.4485 | **0.4591** |
| ALL recovery Δlab | **-5.60** | **+1.74** |
| ALL_excl_outliers recovery Δlab | -- | **+1.62** |
| A06 (outlier) Δlab | -3.48 | **+2.55** |
| A08 Δlab | -5.54 | **+1.82** |
| A09 Δlab | -6.64 | **+1.31** |
| A13 Δlab | -4.81 | **+0.81** |
| A16 Δlab | -4.89 | **+2.00** |

Same architecture, same 4000 steps, same resolution, same everything --
the ONLY variable that changed is training-pair count (50 -> 96, a strict
superset). Every slide's recovery flips sign, and SSIM also improves
slightly rather than trading off against it. This is strong support for
H3: the negative colour recovery at native 1024 was a genuine
insufficient-data problem under the <=50-pair budget, not an architectural
ceiling -- D1-D5 correctly ruled out every mechanistic/inference-only fix
because the actual cause was upstream of all of them.

**Reiterating the scope note (do not lose this in the result's excitement):
P3-07b is a supplementary >50-pair finding, not evidence for or against the
proposal's formal <=50-pair H1 claim.** P3-07 (<=50 pairs, recovery -5.60)
remains the number that answers H1 as literally stated in
`docs/proposal_draft(6).tex`. P3-07b shows WHY P3-07 regressed (insufficient
data at native resolution) and demonstrates the regression is fixable with
more data -- a genuinely informative finding for the thesis discussion
section -- but it does not retroactively make P3-07's <=50-pair colour
recovery positive, and must always be reported alongside P3-07, never in
place of it.

**Status: P3-07/P3-07b both COMPLETE.** Next step (not started, awaiting
direction): write up the full P3-07/P3-07b comparison for the thesis, and
decide whether this closes Phase 3's SDXL-transfer work or motivates a
further descoped follow-up (e.g. checking whether P3-06's 512 checkpoint
shows a similar pair-count sensitivity, out of scope unless explicitly
requested).

**Downstream-classifier extension (2026-08-28, P2-09's atypia_r18
checkpoint, job 47418):** recovery delta (accuracy vs. raw_hamamatsu,
excl. A06) = **+0.0769** -- the single best result across every method
tested this session (P1-10/P1-11/P1-16/P3-06/P3-07), beating even the
original P2-09 headline diffusion config (A3@0.50, +0.0625) and every
classical baseline. **Read this together with the colour-recovery
regression above, not instead of it** -- P3-07's colour recovery is
`-5.60` (negative, confirmed real by D1, not a metric artefact), so this
is not a scanner-normalisation win. Two things line up to explain it: (1)
P3-07 has the best structural SSIM of any SDXL config (0.4313, closing
most of the gap to SD1.5's 0.4485, exceeding SD1.5 outright on A06), and
atypia scoring is a morphology task; (2) the negative colour recovery
means P3-07's output stayed closer to Aperio's own colour statistics
rather than migrating to Hamamatsu's -- and the classifier's own
raw_aperio sanity check already scores higher (0.4279) than real
raw_hamamatsu (0.4135), i.e. the classifier is more comfortable with
Aperio-flavoured colour to begin with. So P3-07 likely wins by combining
the best available structure preservation with colour that (by failing to
normalise) happens to stay in the classifier's more legible domain --
the same "structure over colour" pattern P2-09 already flagged for A1.
**Not evidence P3-07 is the best normalisation method; evidence the
classifier rewards structure + Aperio-like colour, which P3-07 currently
does best of the five, partly because its colour normalisation is
broken.** Full comparison table: `tickets/PHASE2-TICKETS.md` P2-09's
Extension section.

**STALE, and the mechanism above is now confirmed, not just hypothesised
(2026-08-28):** `tickets/P2-12_atypia_classifier_evaluation_hardening.md`
§0 empirically confirms (job 47468) every method scored here, including
P3-07, is A→H (Aperio input, Hamamatsu-styled output) — the opposite of
the H→A direction the clinical-utility claim needs, since the classifier
is Aperio-trained. That is exactly why "structure + Aperio-like colour"
wins regardless of real normalisation quality: the underlying content
being classified is literally still Aperio's. This number is not valid
clinical-utility evidence until P2-12 lands and a genuine H→A rerun
exists (nontrivial for P3-07 specifically — needs fresh SDXL training,
per P2-12 §11, since H→A was never trained for this architecture).

---

**Compute note:** proposal states SDXL is compute-contingent — if training time or
memory is excessive on `bigbatch`, descope to smaller rank, fewer eval slides, or
partial transfer, while SD1.5 A0–A5 remains the core contribution regardless of how
Phase 3 goes.
