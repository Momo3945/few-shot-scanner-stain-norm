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
**Status:** 🔄 IN PROGRESS (2026-08-19) — first real comparison point in hand,
genuinely surprising result, worth a closer look before calling this closed.
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

## P3-05 — A5 warm-start variant on SDXL (contingent)
**Status:** TODO — UNBLOCKED (2026-08-10). P1-07 resolved: A5 shows measurable
benefit over A4 on SD1.5, well beyond the inter-run noise floor (see P3-01's
review). Still blocked on the same SDXL model-weight prerequisites as P3-03.
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

---

**Compute note:** proposal states SDXL is compute-contingent — if training time or
memory is excessive on `bigbatch`, descope to smaller rank, fewer eval slides, or
partial transfer, while SD1.5 A0–A5 remains the core contribution regardless of how
Phase 3 goes.
