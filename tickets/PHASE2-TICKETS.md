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
**Status:** ✅ DONE (2026-08-08) — full 5-slide run. Inference job 37039 COMPLETED
1:27:11 (1488-row manifest, all 5 slides confirmed), rescored by job 37185
COMPLETED 24:41 after the `score_outputs.py` baseline-scope fix landed.
**Source:** `tab:eval` (Colour accuracy — MITOS row); H2/RQ4 in `tab:hyp_rq`
**Acceptance criteria:** `eval/a2h_r8/eval_summary.csv` with per-strength, per-slide
LAB Wasserstein + recovery delta vs raw baseline, across all 5 held-out slides. **Met.**
**Result (strength 0.30 / 0.40 / 0.50, recovery Δlab vs raw baseline):**
| Slide | Baseline LAB | Δlab @0.30 | Δlab @0.40 | Δlab @0.50 |
|---|---|---|---|---|
| A06 *(outlier, z=33.7)* | 94.84 | +0.92 | +1.52 | +2.65 |
| A08 | 25.27 | +2.49 | +2.10 | +1.62 |
| A09 | 27.48 | +1.90 | +1.76 | +1.42 |
| A13 | 26.63 | +5.27 | +4.98 | +4.37 |
| A16 | 23.77 | +1.62 | +1.47 | +1.31 |
| **ALL** | 33.83 | +1.72 | +1.60 | +1.44 |
| **ALL excl. A06** | 25.41 | +2.45 (hand-computed — `score_outputs.py` doesn't auto-fill this column yet, see below) | — | — |

Every slide recovers at every strength tested — real, positive signal, not an
artifact. But the strength trend **diverges by scope**: A06 improves *with* higher
strength (+0.92→+2.65), while A08/A09/A13/A16 all recover *less* at higher strength
— 0.30 is the better strength for 4 of 5 slides. SSIM falls monotonically with
strength throughout (pooled: 0.289→0.229→0.179) — the expected colour/structure
tradeoff. Per CLAUDE.md guardrails: report per-slide + outlier-excluded aggregate,
never let pooled ALL stand alone — table above does both.
**Known minor gap, not blocking:** `score_outputs.py`'s `ALL_excl_outliers` row
never had its `recovery_delta_lab` column wired up (always blank) — only the `ALL`
row's delta got fixed earlier. Worth a follow-up fix so this doesn't need
hand-computing each time.
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
- **Fourth blocker, also found while fixing:** `heldout_frames.csv` (Do NOT touch —
  fixed in code instead) stores Windows backslash-separated relative paths;
  `Path()` on Linux doesn't treat `\` as a separator, so every join with
  `MITOS_ROOT` would have failed. Fixed in `infer_colour_lora.py`: paths are
  normalised to `/` right after the CSV is read. Verified directly against real
  files on the cluster before re-submitting.
- **Re-submitted with all 4 fixes → job 36713, FAILED again, 3:29 elapsed.** Got
  much further this time: SD1.5 pipeline loaded from cache fine (42s). New error:
  `ValueError: When using the offline mode, you must specify a weight_name.` —
  `pipe.load_lora_weights(args.lora)` normally auto-detects the weight filename via
  a Hub API call, which is blocked by `HF_HUB_OFFLINE=1` (set deliberately in the
  script). **Fifth blocker, fixed:** pass `weight_name="pytorch_lora_weights.safetensors"`
  explicitly — confirmed this is the actual filename diffusers saved for all 3
  existing runs (a2h_r8, a2h_r4, h2a_r8).
**First successful end-to-end run (2026-08-08): job 36735 (infer) COMPLETED 9:09,
job 36768 (score) COMPLETED 2:15.** Pipeline validated — all 5 fixes held. But this
is a partial result, not yet the ticket's acceptance criteria:
- Default `LIMIT=8` pulled only the *first* 8 rows of `heldout_frames.csv`, which are
  all **A06** — the confirmed colour-gap outlier slide (baseline LAB≈95 vs ≈25–27 for
  the other four). The other 4 held-out slides (108 of 124 total frames) were not run.
- **Real A06-scoped result:** baseline LAB 94.84 → LoRA output 92.41/91.89/90.63 at
  strength 0.30/0.40/0.50 → recovery Δlab **+2.42/+2.94/+4.21** (modest, real, grows
  with strength; SSIM falls 0.193→0.153→0.126 over the same range — the expected
  colour/structure tradeoff).
- **`eval_summary.csv`'s `scope=ALL` row is misleading, not a real regression:** it
  shows recovery_delta_lab ≈ −58, because this run's data (A06 only) gets compared
  against `baseline_summary.csv`'s true 5-slide pooled `ALL` (33.83) under the same
  `ALL` label — an apples-to-oranges scope mismatch, not the LoRA making things worse.
  `score_outputs.py` should ideally only emit an `ALL` row when the run's slide
  coverage actually matches the baseline's, or label it more precisely (e.g.
  `ALL_present_slides`) — not fixed yet, just flagged.
**Next step:** re-submit with `LIMIT=0` (all 124 frames, all 5 slides) to get the
actual proposal-compliant result — `tab:split` requires "All ×20 frames" for
A06/A08/A09/A13/A16. At ~9min/8 frames the full run is ~2–2.5h, still under the
`.slurm` file's 3h time limit, but worth confirming before submitting a run that long.

## P2-05 — Evaluation run: A2H rank 4
**Status:** ✅ DONE (2026-08-08) — full 5-slide run. Inference job 37071 COMPLETED
1:28:58, rescored by job 37218 COMPLETED 24:23.
**Source:** rank experiment, H1/RQ1 in `tab:hyp_rq`
**Purpose:** decide the winning rank empirically (held-out recovery delta), not from
training loss — see CLAUDE.md methodology guardrails.
**Result — rank 4 vs rank 8, recovery Δlab (rank 4 / rank 8):**
| Slide | @0.30 | @0.40 | @0.50 |
|---|---|---|---|
| A06 *(outlier)* | +1.16 / +0.92 | +1.92 / +1.52 | +3.12 / +2.65 |
| A08 | +2.18 / +2.49 | +1.85 / +2.10 | +1.64 / +1.62 |
| A09 | +1.75 / +1.90 | +1.64 / +1.76 | +1.37 / +1.42 |
| A13 | +5.08 / +5.27 | +4.67 / +4.98 | +4.06 / +4.37 |
| A16 | +1.47 / +1.62 | +1.40 / +1.47 | +1.33 / +1.31 |
| **ALL** | +1.58 / +1.72 | +1.50 / +1.60 | +1.46 / +1.44 |

**Rank barely matters** — the two ranks track each other within ~0.1–0.3 LAB units
on every slide at every strength, well inside likely noise. Rank 8 edges out rank 4
on 3 of 5 slides at strength 0.30 (A08, A13, A16) but the margin is small. No strong
empirical case for rank 8 over rank 4, or vice versa — supports the proposal's H1
rank hypothesis (`r∈{4,8}` should be "sufficient in theory because scanner
normalisation is expected to be a low-complexity colour shift"). **Recommendation
for P1-04:** rank 8 as the marginal edge-case winner, but this is a weak signal, not
a strong one — worth noting in the writeup rather than treating as decisive.

## P2-06 — Cycle consistency: ground-truth direct comparison
**Status:** ✅ DONE (2026-08-10) — no new compute required.
**Source:** `sec:experiments` "Ground Truth Direct Comparison"; equation for
A06_00A → Â H06_00A ↔ Real H06_00A
**Description:** Normalised Aperio output compared directly to registered real
Hamamatsu on held-out test slides. Metrics: grayscale SSIM, PSNR, MAE (post-registration).
**Why no new compute was needed:** this exact metric triple, computed against the
registered real Hamamatsu reference, is already produced as a byproduct of every
`score_outputs.py` run via `src/eval/metrics.py`'s `score_aligned_pair()` (calls
`grayscale_ssim`, `psnr`, `mae` alongside `lab_wasserstein`), and is already written
into every completed run's `eval_summary.csv` (`ssim`/`psnr`/`mae` columns) for
baseline, A0–A5, the P1-09 extended-strength sweep, and the P2-11 classical
baselines. P2-06 was therefore a compilation task, not an experiment: pulled the
`ssim`/`psnr`/`mae` columns already present across all 16 already-scored
configs into one dedicated ground-truth-direct-comparison view.
**Results:** `docs/results/RESULTS_SUMMARY.md` ("P2-06: ground-truth direct
comparison (SSIM / PSNR / MAE)"). Headline: this metric lens tells a starkly
**different** story than the LAB-Wasserstein recovery-delta tables elsewhere in
this file — the raw do-nothing baseline (SSIM 0.733 pooled) and the classical
colour-remap baselines (SSIM 0.63–0.68) both beat every diffusion rung (SSIM
0.27–0.46) on SSIM/PSNR/MAE, including on A06. Diffusion img2img regeneration
trades pixel-exact structural/luminance fidelity for improved colour-distribution
matching, whereas raw/classical methods leave pixel structure untouched. See the
results file for the full table and interpretation — this is a real limitation to
flag in the write-up, not a scoring artefact.
**Update (2026-08-10):** CIEDE2000 (`de2000_mean`) was added to `score_aligned_pair()`
and tested as a candidate independent perceptual-colour check. It isn't one —
computed pixel-wise on the same registered pair, it belongs to this same
pixel-exact metric family and tracks SSIM rank-for-rank (every diffusion config
scores worse than the raw baseline, same pattern as SSIM/PSNR/MAE above). The
metric that did provide genuinely independent evidence was windowed
LAB-Wasserstein, added under P2-11 — see that ticket and
`docs/results/RESULTS_SUMMARY.md` for the full analysis.

## P2-07 — Cycle consistency: round-trip reconstruction
**Status:** ✅ DONE (2026-08-17/18)
**Source:** `sec:experiments` "Round Trip Reconstruction"
**Description:** A06 → (LoRA A→H) → Ĥ06 → (LoRA H→A) → Â06 ↔ Original A06.
**IMPORTANT (proposal is explicit):** this test uses a **deterministic 50-step DDIM
sampler**, NOT LCM — LCM's stochastic drift would contaminate the structural deviation
measurement. LCM is reserved strictly for the unidirectional deployment pipeline.

**Implementation:** new `src/eval/roundtrip_a06.py` + `slurm/roundtrip_a06.slurm` —
deliberately not a reuse of `infer_colour_lora.py`, which always ECC-registers
against a real Hamamatsu image and scores against that real reference; this test
never touches a real Hamamatsu image (Ĥ06 is synthetic) and compares Â06 back to
the exact same original A06 crop it started from (trivially pixel-aligned by
construction, no registration needed). Two sequential single-adapter DDIM passes
(A→H LoRA, then H→A LoRA on the result), strength 0.20 (P1-09's best
general-purpose point), no ControlNet (the proposal's round-trip equation names
only the trained LoRA weights). Scored with `metrics.py`'s existing
`score_aligned_pair` (SSIM/PSNR/MAE + bonus LAB/windowed-LAB/CIEDE2000), no new
metric code.

Smoke test (job 43667, 2 frames, 20-step DDIM, n=8 crops): mechanically clean,
finite non-degenerate numbers. **Full run (job 43971, all 16 A06 frames, 50-step
DDIM, n=64 crops, `per_crop.csv` confirmed 64 rows):**

| metric | value |
|---|---|
| SSIM | 0.1429 |
| PSNR | 13.11 |
| MAE | 41.54 |
| LAB total | 28.17 |

**Reading this result:** SSIM 0.1429 is very low structural preservation —
notably lower even than the one-way normalisation-vs-real-Hamamatsu SSIM numbers
already reported under P2-06 (0.27–0.46 for the best diffusion rungs at their
best strength). Since this round trip never touches a real Hamamatsu image at
all, this isolates the pipeline's own structural drift (both LoRA passes
combined) with the colour-matching task removed entirely — the round trip
should, in principle, return close to the identity if the LoRAs were only
learning colour. It doesn't. This is consistent with, and reinforces, the
P2-06/P2-08 pattern (diffusion resynthesis does not preserve pixel-exact
structure well) rather than complicating it — a third independent metric family
(after SSIM/PSNR/MAE on real pairs, and Relative Dice) now points the same
direction. No proposal-stated pass/fail threshold exists for this experiment
(unlike H3's 0.95); reported descriptively per this project's convention.

## P2-08 — Structural safety: HoVer-Net vs Lizard (Relative Dice)
**Status:** DONE — gate FAILED (2026-08-17). Dice(A,G) validation gate DONE on
the full 130-image set (mean Dice 0.6982, jobs 43364/43365). Normalisation (B)
side complete (jobs 43465/43537) and Relative Dice computed (job 43661):
**0.8745, below the 0.95 threshold — H3/RQ2 does not hold** at the P1-09 best
general-purpose operating point. See full results below.
Lizard
`dpath_*`/`glas_*` subset (130 images + `.mat` labels, 695MB) uploaded to
`/datasets/mhoosen/stain-norm/lizard_heldout/{images,labels}/`, verified (file
counts match the local staged copy exactly). `tiatoolbox==1.6.0` installed and
verified working (`tiatoolbox 1.6.0`, `torch 2.5.1+cu121`, `cuda available: True`,
`NucleusInstanceSegmentor` imports cleanly) in a **new, separate conda env
`stainnorm-hovernet`** — NOT the working `stainnorm` env, because a dry-run showed
tiatoolbox's dependency stack would downgrade `scipy` 1.15.3→1.14.1, `scikit-image`
0.25.2→0.24.0, and `opencv-python-headless` 5.0.0.93→4.11.0.86, which are exactly
the packages every already-verified LAB-Wasserstein/CIEDE2000/SSIM number in this
project depends on. Isolating into a second env avoided that risk entirely.
**Install notes for reference:** needed `pip install openslide-bin` afterward (missing
native `libopenslide.so`, not a Python-level issue — tiatoolbox imports openslide at
module load time even though this eval doesn't need whole-slide-image support).
The cluster login node's SSH connection dropped mid-install twice during this
work — unrelated to tiatoolbox itself; fixed by re-running with `nohup ... &
disown` so the remote process survives a dropped session, then polling the log
for a completion marker instead of holding one long-lived SSH connection open.
**Not yet done:** `src/eval/hovernet_wrapper.py` and `src/eval/lizard_dice.py`
(steps 3-4 in the pipeline below) have not been written.
**Source:** `sec:experiments` "Structural Safety"; success threshold Relative Dice ≥ 0.95
**Description:** Dice(B,G)/Dice(A,G) where G=Lizard ground truth, A=HoVer-Net on
original patch, B=HoVer-Net on normalised patch. PanNuke excluded from this eval
(it appears in A5 training). HoVer-Net must first be validated against Lizard human
masks (Dice(A,G)) before being trusted as a measurement instrument.

**Scoping research (2026-08-10, web-verified, sources in the research agent's report):**

- **HoVer-Net integration — use `tiatoolbox==1.6.0`, not raw `vqdang/hover_net`.**
  The official HoVer-Net repo hard-pins torch 1.6.0 + CUDA 10.2 — incompatible with
  this project's working `stainnorm` env (torch 2.5.1+cu121) and would need a whole
  separate, six-years-obsolete environment. `tiatoolbox` 1.6.0 specifically requires
  `torch>=2.1.0,<=2.5.1` — the existing env sits exactly at that upper bound, so it
  installs directly with no torch churn (verify with a dry-run/`--no-deps` check
  before actually installing on the cluster; never let this silently touch the
  pinned torch build per CLAUDE.md). Newer tiatoolbox (≥2.1.0) requires Python
  ≥3.11, incompatible with this env's Python 3.10 — 1.6.0 is the version to pin, not
  latest. Ships pretrained weights that auto-download by name; `NucleusInstanceSegmentor`
  is the inference engine in this version (deprecated in favour of
  `MultiTaskSegmentor` in newer releases, but 1.6.0's API is what we're pinning to).
  License BSD-3 (toolkit) / CC BY-NC (weights) — fine for this non-commercial project.
- **Checkpoint choice: `hovernet_fast-pannuke`.** "Fast" mode's 256×256-in/164×164-out
  shape is the tiling target for inference (see below). Being trained on PanNuke does
  not conflict with excluding PanNuke-*sourced Lizard images* from the eval set —
  those are separate concerns (this is a third-party off-the-shelf instrument being
  validated via Dice(A,G), not a component under test for data leakage).
- **Local Lizard data inspected directly (2026-08-10) — both open items from the
  first scoping pass are now resolved, not just hypothesized:**
  - **Packaging:** this is the **original variable-size region release** (NOT the
    CoNIC 256×256 repackaging guessed at earlier). 238 images total across
    `data/lizard/lizard_images{1,2}/Lizard_Images{1,2}/`, real sizes verified
    directly (`dpath_1.png` 1190×928, `glas_1.png` 775×522, `crag_1.png` 1509×1516)
    — will need `grid_offsets()`-style tiling into HoVer-Net's 256×256 input, the
    same tiling approach already used for MITOS crops elsewhere in this codebase.
    No new tiling logic needed, just reuse of the existing pattern.
  - **Source-origin encoding: a plain filename prefix**, verified by listing both
    image folders — `consep_*` (16), `crag_*` (64), `dpath_*` (69), `glas_*` (61),
    `pannuke_*` (28) = 238. No `tcga_*` present — Lizard's own README explains the
    authors are holding the TCGA portion back for a future challenge, so it was
    never part of any download, not something missing from this copy.
    **Usable subset for this eval: `dpath_*` + `glas_*` = 130 images, ~237MB.**
  - **Ground-truth format: verified directly via `scipy.io.loadmat`**, not just
    trusted from the README — `data/lizard/lizard_labels/Lizard_Labels/Labels/
    dpath_1.mat` has exactly the documented keys: `inst_map` (int32, shape == image
    H×W exactly, 0=background), `id` (N,1), `class` (N,1, confirmed values 1–6),
    `bbox` (N,4), `centroid` (N,2). One `.mat` per image, flat directory, 238 files
    matching 238 images 1:1. `Lizard_Labels/info.csv` also ships a
    Filename/Source/Split column (the dataset's own paper split) — not needed for
    this eval but available if a train/val/test provenance check is ever useful.
- **Dice metric: binary, pixel-pooled foreground-vs-background Dice** (nucleus
  pixels vs. not, instance identity collapsed) — this is HoVer-Net's own
  `get_dice_1()`/"standard DICE", and also what the Lizard paper's own authors report
  as their headline "Binary Dice" when validating segmentation quality. No
  stain-normalisation paper was found running this exact before/after-normalisation
  protocol — it's original to this proposal, not a reused published recipe. Secondary
  metrics (IoU, object-level F1, nuclear count consistency) are all computable from
  the same instance-map output tiatoolbox/hover_net already produce, at no extra
  inference cost.
- **Compute:** GPU on `bigbatch`; published HoVer-Net benchmarks (~5s/1000×1000 image)
  suggest 130 images tiled into 256×256 patches is a low-tens-of-minutes job, not a
  long run.

**Planned pipeline:**
1. ✅ Upload the `dpath_*`/`glas_*` subset (130 images + matching `.mat` labels,
   695MB) to `/datasets/mhoosen/stain-norm/lizard_heldout/{images,labels}/`,
   verified. (Actual size 695MB, not the ~237MB images-only estimate — `.mat`
   label files carry full-resolution instance maps and are the bulk of it.)
2. ✅ `tiatoolbox==1.6.0` installed and verified in the new `stainnorm-hovernet`
   env (see above).
3. ✅ `src/eval/hovernet_wrapper.py` written and committed (`2167c5c`) — wraps
   `NucleusInstanceSegmentor(pretrained_model="hovernet_fast-pannuke", mode="tile")`
   run directly over whole Lizard region images (design changed from the original
   plan: tiatoolbox's own internal patch extraction + stitching already tiles and
   reassembles at native resolution, so manually pre-tiling via `grid_offsets()`
   would only have introduced boundary artefacts — clipped/split nuclei at tile
   edges — that tiatoolbox is specifically built to avoid). Rasterises each
   predicted nucleus contour into a binary mask via `cv2.fillPoly`, re-serialises
   tiatoolbox's joblib output as stdlib `pickle` so downstream scoring has zero
   tiatoolbox/joblib dependency. Runs in `stainnorm-hovernet`.
4. ✅ `src/eval/lizard_dice.py` written and committed (`2167c5c`) — binary
   pixel-pooled Dice (primary), IoU, greedy centroid-matched object-F1 (documented
   as a simplified convention, not Hungarian-optimal — the primary pass/fail
   metric is Dice, not this), and nuclear count ratio, all against the `.mat`
   ground-truth `inst_map`/`centroid` fields. `--against <summary.csv>` computes
   Relative Dice = Dice(this)/Dice(against) with the proposal's ≥0.95 verdict
   printed directly. Runs in the plain `stainnorm` env (no tiatoolbox needed).
   **Verified before deploying:** Dice/IoU/F1 math against synthetic masks
   (known-answer checks, e.g. two offset 4×4 squares → Dice 0.5625 exactly as
   hand-computed), and `load_gt()` against the real `dpath_1.mat` — mask shape
   (928,1190) and centroid count (2411) both matched the values already
   confirmed during scoping. Both scripts syntax-check in their real cluster
   envs (`stainnorm` / `stainnorm-hovernet` respectively).
5. Run the best Phase 1 config (per P3-01: A4, or the P1-09 strength-0.20 operating
   point) as the "normalisation" step over the Lizard images to produce B — likely
   needs `infer_colour_lora.py` adapted or pointed at a Lizard-shaped manifest
   rather than the MITOS `heldout_frames.csv` schema it currently expects; not
   yet designed.
6. Compute Dice(A,G) first as the validation gate (proposal: HoVer-Net must be
   trusted as an instrument before Relative Dice means anything) — only proceed to
   Relative Dice = Dice(B,G)/Dice(A,G) once that gate is sane. `lizard_dice.py`
   already supports this directly (run once without `--against` for the gate,
   again with `--against` for the real comparison).

**Slurm launchers:** `slurm/infer_hovernet.slurm` (GPU/`bigbatch`, activates
`stainnorm-hovernet`, takes `TAG [IMAGES_DIR] [PRETRAINED_MODEL] [LIMIT]` so the
same script serves both the original-Lizard and future normalised runs) and
`slurm/score_lizard.slurm` (CPU/`stampede`, plain `stainnorm` env, takes
`PRED_TAG [AGAINST_TAG]` — omit `AGAINST_TAG` for the Dice(A,G) validation gate,
supply it for Relative Dice). Both syntax-checked locally and on the cluster.

**Smoke test (2026-08-12):** first attempt (job 42420) landed on a bad GPU node
and failed fast (1:10, fail-fast guard caught it). Resubmit (job 42904)
COMPLETED 3:13 — `sbatch slurm/infer_hovernet.slurm lizard_original "" "" 5`,
5/5 `dpath_*` images processed, `instances/`+`masks/`+`manifest.csv` written to
`eval/lizard_original/` (nucleus counts 1624-7224/image, mask dims match image
dims exactly — sane output, no design changes needed from the earlier
`hovernet_wrapper.py` write-up).
**Dice(A,G) validation gate (2026-08-12), job 43055 COMPLETED 27s:**
`sbatch slurm/score_lizard.slurm lizard_original` — 5-image smoke sample:

| image | dice | iou | object_f1 | count_ratio |
|---|---|---|---|---|
| dpath_1 | 0.722 | 0.565 | 0.832 | 0.784 |
| dpath_10 | 0.776 | 0.634 | 0.864 | 0.855 |
| dpath_11 | 0.700 | 0.539 | 0.796 | 0.749 |
| dpath_12 | 0.671 | 0.505 | 0.836 | 0.854 |
| dpath_13 | 0.765 | 0.619 | 0.858 | 0.824 |
| **mean** | **0.727** | **0.573** | **0.837** | **0.813** |

Consistent across all 5 (0.67-0.78 Dice, no outliers/failures) — HoVer-Net
behaves as a sane instrument on this data (under-detects nuclei somewhat,
~75-85% of ground-truth count, typical for a PanNuke-pretrained model applied
out-of-domain, not a pipeline bug). This is a 5-image smoke sample, not yet the
full validation gate — n=5 is too small to trust as the official Dice(A,G)
baseline that Relative Dice will be divided by.
**Full 130-image Dice(A,G) validation gate (2026-08-15):** job 43364
(`sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44
slurm/infer_hovernet.slurm lizard_original "" ""`) COMPLETED 8:36 on
`mscluster79` — "Processed 130/130 images", `instances/`+`masks/`+
`manifest.csv` written to `eval/lizard_original/` (first attempt at this full
run, job 43114, had failed on `mscluster83` with a healthy GPU due to a stale
`_tiatoolbox_raw/` dir left over from the 5-image smoke test occupying the
same output path — fixed by deleting the stale `_tiatoolbox_raw/`/`instances/`
/`masks/`/`manifest.csv` before resubmitting, not a hardware fault). Rescored:
job 43365 (`sbatch slurm/score_lizard.slurm lizard_original`) COMPLETED 1:06,
`per_image.csv` has 130 rows.

| metric | value | n |
|---|---|---|
| Dice | 0.6982 | 130 |
| IoU | 0.5394 | 130 |
| object-F1 | 0.7332 | 130 |
| count_ratio | 0.6678 | 130 |

This is the official Dice(A,G) baseline Relative Dice will be divided by —
lower than the 5-image smoke mean (0.727), as expected once the full
distribution (incl. `glas_*`) is included; still consistent with the smoke
test's read that HoVer-Net is a sane (if somewhat under-detecting)
out-of-domain instrument, not a broken one.
**Step 5 — normalisation (B side) + Relative Dice (2026-08-16/17):** new
standalone script `src/eval/normalize_lizard.py` + launcher
`slurm/normalize_lizard.slurm`, deliberately not a reuse of
`infer_colour_lora.py` (Lizard has no Aperio/Hamamatsu pair or registration
step, and `lizard_dice.py:154` requires the normalised output to match its
source image's `(H, W)` exactly or that image is silently skipped). Approach:
pad each image to a multiple of `--crop` (512) with reflect padding, tile into
non-overlapping blocks (deliberately not `grid_offsets()`'s overlap-sampling —
overlap would double-process pixels and seam on reassembly), run the P1-09 best
general-purpose config (colour LoRA `a2h_r8` + ControlNet-Canny + LCM-LoRA, A2H,
strength 0.20, 8 steps, guidance 1.5, seed 0 — `infer_a4_lcm.slurm`'s A4 config,
not A5@0.70's outlier-rescue mode, since Lizard is a general held-out set) per
tile, stitch, crop back to source size.

Smoke test (job 43375, 5 images): all 5 shapes matched their originals exactly,
pixel stats non-degenerate. Full run (job 43465, `mscluster51`, COMPLETED 7:26):
**130/130 images normalised**, no shape-mismatch errors, output count verified
(`ls | wc -l` = 130).

HoVer-Net on the normalised images (job 43537, `mscluster84`, COMPLETED 9:02):
**"Processed 130/130 images"**, 130 masks + 130 instance files verified directly
(not just exit code).

Final scoring (job 43661, `sbatch slurm/score_lizard.slurm lizard_normalised
lizard_original`, `mscluster41`, COMPLETED 1:21), `per_image.csv` has 131 lines
(header + 130 rows, no skips):

| metric | value | n |
|---|---|---|
| Dice (B,G) | 0.6105 | 130 |
| IoU | 0.4428 | 130 |
| object-F1 | 0.6816 | 130 |
| count_ratio | 0.6274 | 130 |

**Relative Dice = Dice(B,G)/Dice(A,G) = 0.6105/0.6982 = 0.8745 — FAIL (< 0.95
threshold).** `relative_dice.csv`: `this_dice=0.61054, baseline_dice=0.69818,
relative_dice=0.87447, threshold=0.95, pass=False`. H3/RQ2 does not hold at this
operating point: normalising with the best general-purpose Phase 1 config costs
~12.5% of HoVer-Net's nucleus-detection agreement relative to the unmodified
image. Consistent with, and reinforces, the P2-06 finding that diffusion
resynthesis does not preserve pixel-exact/structural fidelity as well as doing
nothing — this is the same pattern showing up in a downstream-task metric
(nucleus detection) rather than only in raw pixel metrics (SSIM/PSNR/MAE),
which strengthens rather than complicates that earlier finding. Report as a
genuine negative result, not explained away — per this project's standing
convention (CLAUDE.md) of reporting negative findings plainly.

**Not yet done / possible follow-up (not required to close this ticket):**
Relative Dice was only computed at the single P1-09 "best general-purpose"
operating point. Whether a different config (e.g. A5's outlier-rescue strength,
or literal ControlNet-only A1) changes the verdict is untested — could be a
worthwhile discussion-section caveat or a small follow-up ablation, not
currently scoped as a ticket.

## P2-09 — Clinical utility: downstream classifier delta
**Status:** ✅ DONE (2026-08-19) — classifier trained (val_acc 0.9468), scored
against all 26 methods. **Genuinely new finding: diffusion beats every
classical baseline on downstream classifier accuracy** (A3@0.50: +0.0625 vs.
raw Hamamatsu, best classical Reinhard only +0.0457) — the opposite pattern
from every other metric in this project. See full table and caveats below.
**Source:** `sec:experiments` "Clinical Utility"
**Description:** Standard diagnostic classifier trained on raw Aperio, evaluated on
raw Hamamatsu + all baseline methods (P2-11) + A0–A5 outputs. Primary metric:
recovery delta = performance(Proposed) − performance(Raw Hamamatsu). Primary dataset
metric: atypia score accuracy at ×20 (macro-AUC/F1 secondary); mitosis F-measure at
×40 as cross-magnification generalisation check.

**Scope decision:** atypia classification only, not the mitosis F-measure arm —
that needs an object-detector architecture (centroid/bbox prediction + IoU
matching), a materially larger, different build than a classifier. Split off as
follow-up **P2-09b** (not yet ticketed in detail) rather than blocking this on
it, same scope-split precedent as P3-03/P3-03b.

**Label data (found this session, previously completely unused by any script
in this repo):** MITOS-ATYPIA-14's `<slide>/atypia/x20/<slide>_<frame>_
cna_score_decision.csv` files (bare integer 1–3, no header) carry the adjudicated
atypia score per ×20 frame. Present for the 5 held-out/testing slides already on
the cluster; present for all 11 official training slides only locally
(`D:\Research\Mitos\data\mitos\mitos_atypia_2014_training_aperio\`) until this
session — `mitos_atypia_2014_training_aperio.zip` (11.6GB, Aperio side only,
Hamamatsu side not needed for training) uploaded and extracted on the cluster
under `/datasets/mhoosen/stain-norm/mitos_atypia_train_aperio/`. Real quirks
found and handled while building the manifest join (`src/data/
build_atypia_manifest.py`, not assumed from documentation — verified directly
against real files): (a) 3 of 300 training decision files are genuinely empty,
no adjudicated score at all (`A10_01C`, `A14_00A`, `A17_02B`) — skipped, not
treated as parse errors; (b) the training archive's directory layout is
**inconsistent across slides** — A03 is doubled (`A03/A03/atypia/x20/...`,
matches the pre-existing CLAUDE.md note about a different, earlier A03
extraction) but the other 10 training slides are not doubled
(`A10/atypia/x20/...` directly) — the manifest builder resolves the slide root
relative to each label file's own parent chain rather than assuming either
layout. Result: **297/300 training frames labelled** (score distribution
`{1: 23, 2: 222, 3: 52}`), **120/124 testing frames labelled**
(`{1: 38, 2: 60, 3: 22}`) — both manifests cross-checked against a real raw
label file (not just the script's own summary output) before being trusted.

**Training methodology (`src/train/train_atypia_classifier.py`):**
- **Model**: `torchvision.models.resnet18` (ImageNet-pretrained), final `fc`
  replaced with a 3-class linear head, fine-tuned end-to-end. A defensible
  "standard diagnostic classifier" baseline — the proposal doesn't mandate a
  specific architecture.
- **Data granularity**: each training example is a 512×512 tissue crop (same
  `grid_offsets`/`tissue_fraction` tiling convention used everywhere else in
  this project), inheriting its **parent frame's** atypia score as a weak,
  frame-level label — MITOS-ATYPIA-14 assesses atypia per ROI, not per
  sub-tile, so this is the standard assumption for histopathology patch
  classification. This also means the eval side needs no frame-level
  aggregation step later: each existing eval crop can be scored directly
  against its frame's true label.
- **Class imbalance**: training labels are heavily skewed (`2` is ~75% of
  frames) — `CrossEntropyLoss` is class-weighted (inverse frequency) to avoid a
  degenerate "always predict 2" classifier that would still report a
  misleadingly high raw accuracy.
- **Train/val split**: **by slide**, not by crop, to avoid leaking crops from
  the same frame across the split — the 2 alphabetically-last training slides
  (A17, A18) held out for validation by default, giving 250 train / 47 val
  frames → 1000 train / 188 val tissue crops after tiling. Train class counts:
  `{score1: 88, score2: 704, score3: 208}`.
- **Checkpointing**: every `save_every` steps, tracks `best.pt` by val accuracy
  separately from the final-step checkpoint — matters in practice (see below).

**Smoke test (job 44082, `mscluster54`, COMPLETED 3:36):** 5 steps, real finite
decreasing-ish loss, checkpoint saved and verified (44.8MB, real ResNet18 size),
before committing to the full run.

**Full training run (job 44146, `mscluster61`, COMPLETED 53:31, 2000 steps):**
train accuracy reaches 1.000 by ~step 1550 (expected — weak frame-level labels
shared across many crops of the same frame are easy to fit exactly). **Val
accuracy peaked at 0.9468 at step 400**, then drifted down with continued
training (0.894 at step 1400, down to 0.824 by the final step 2000) — a real,
visible overfitting trend past the peak, which is exactly why `best.pt`
(step-400 weights) is the checkpoint used for evaluation, not `final.pt`.
Verified directly: `best.pt` is 44.8MB and `torch.load` confirms
`step=400, val_acc=0.9468`, not just trusting the training log's own printout.

**Scoring (job 44205 FAILED — real bug, not a fluke; job 44281 fixed +
COMPLETED, `mscluster55`, 8:32):** job 44205 crashed inside the optional
`raw_aperio` sanity-check path with `FileNotFoundError` — traced directly:
`slurm/score_atypia_classifier.slurm`'s `TESTING_ROOT` was set one directory
level too deep (`mitos_heldout/mitos_atypia_2014_testing_aperio`), but
`eval_manifest.csv`'s `aperio_path` column is already relative to
`mitos_heldout/` (e.g. `mitos_atypia_2014_testing_aperio/A06/...`) — joining
doubled that path segment. Fixed (`TESTING_ROOT=${DATA_ROOT}/mitos_heldout`),
re-verified against the real file location, resubmitted as job 44281.
`per_crop.csv` has 12,481 real rows across all 26 methods (9 diffusion rungs ×
3 strengths, 3 classical baselines, raw Aperio/Hamamatsu) — verified directly,
not just trusted the exit code.

**Result — a genuinely new finding, the opposite pattern from every other
metric measured this project.** Recovery delta = accuracy(method) −
accuracy(raw Hamamatsu), **A06 excluded from every method uniformly** (not
relying on each method's own inconsistently-triggered auto-outlier flag, per
CLAUDE.md's standing rule — recomputed directly from `per_crop.csv`'s
per-crop `correct` column, n=416 non-A06 crops per method):

| Method | Accuracy (excl. A06) | Δ vs raw Hamamatsu |
|---|---|---|
| **A3 @0.50** (ControlNet+LoRA) | **0.4760** | **+0.0625** |
| A1 @0.50 (ControlNet only) | 0.4712 | +0.0577 |
| A2 r4/r8 @0.40 | 0.4688 | +0.0553 |
| A3 @0.30 | 0.4663 | +0.0529 |
| A1 @0.30 | 0.4639 | +0.0505 |
| A3 @0.40 | 0.4591 | +0.0457 |
| **Reinhard (best classical)** | **0.4591** | **+0.0457** |
| raw Aperio (sanity check) | 0.4279 | +0.0144 |
| raw Hamamatsu (baseline) | 0.4135 | 0.0000 |
| Histogram matching | 0.4159 | +0.0024 |
| **Macenko** | **0.3462** | **−0.0673** |

**Every diffusion rung near the top of this table beats every classical
baseline** — Reinhard is the strongest classical method here (+0.0457) but
still loses to A1/A3 at every strength tested; Macenko actively *hurts*
downstream classification (worse than doing nothing at all); histogram
matching barely helps. This is the reverse of the LAB-Wasserstein/SSIM/
Relative Dice/round-trip story throughout this project, where classical
methods won by 5–8×. Cross-checked against macro-F1/macro-AUC (not just
accuracy) for the top config: A3@0.50 macroF1 0.376 / macroAUC 0.540 vs. raw
Hamamatsu's 0.248 / 0.435 (notably *below* random-chance AUC) — consistent
with the accuracy-based finding, not an artefact of one metric.

**Caveats, stated plainly, not hidden:**
- The classifier itself is a fairly weak/noisy instrument — even **raw Aperio
  (its own training-domain sanity check) only scores 42.79%**, barely above
  the 33% random-chance floor for 3 classes, and only +1.4 points over doing
  nothing. This whole comparison sits on a noisy measuring instrument; the
  *relative* ranking (diffusion > Reinhard > raw > Macenko) is the reliable
  part, not the absolute numbers.
- **A1 (ControlNet alone, no colour LoRA at all) is among the top performers**
  — suggests structural conditioning specifically, not colour adaptation, may
  be driving a real chunk of this benefit. Worth a dedicated look (does A1's
  gain come from the classifier reading structure more reliably, independent
  of colour, since A1 never touches colour at all?) before concluding the
  full pipeline gets the credit.
- A0 (frozen base, no adapters) is inconsistent by strength (+0.029 → +0.022
  → −0.034) — not a reliable positive on its own.
- This is one downstream task (atypia classification) on one classifier
  architecture (ResNet18) trained once — not yet a robustness-checked finding
  the way the colour/structure metrics are (which used windowed-metric and
  confound-check follow-ups). Worth flagging as a promising, not yet
  fully-hardened, positive result.

**Status: DONE.** This is the first clearly positive result for the diffusion
pipeline anywhere in this project, on arguably the most clinically-relevant
metric measured — a real counterpoint to the otherwise consistent
classical-beats-diffusion pattern, worth featuring prominently in the write-up
precisely because it doesn't fit the rest of the story.

**STALE (2026-08-28) — the headline table above, and the Extension below,
must both be treated as not-yet-valid clinical-utility evidence.** Newly
written `tickets/P2-12_atypia_classifier_evaluation_hardening.md` §0
empirically confirms (job 47468, direct pixel comparison) that every
method scored above ran A→H (Aperio input, Hamamatsu-styled output) — the
opposite of this ticket's own stated design (see the "Description" field
at the top of this ticket and the proposal's own §"Clinical Utility" text:
the classifier is Aperio-trained, so "raw Hamamatsu" and "the proposed
A0-A5 outputs" being scored as parallel substitutable items only makes
sense if those outputs are H→A). This plausibly explains this section's
own already-noted anomaly below (A1, with no colour LoRA at all, among the
top performers) — a structural, not colour, effect, consistent with the
scored outputs still being literally Aperio-content underneath. Status
stays DONE as a record of what was computed and why it doesn't yet answer
the intended question; P2-12 defines the corrected rerun.

**Extension (2026-08-28): the same `atypia_r18` checkpoint (inference only,
no retraining) re-scored on P1-10/P1-11/P1-16 (Phase 1) and P3-06/P3-07
(Phase 3), none of which existed when this ticket originally closed.**
Job 47418 (`score_atypia_p1_p3_p16`, COMPLETED). Required a small additive
patch to `score_atypia_classifier.py`/`score_atypia_classifier.slurm` (an
optional `TAGS` env-var override, plus handling for the newer
`infer_colour_source_ddim_inversion.py`-family manifest schema which has no
`strength` column at all, unlike the original `infer_colour_lora.py`/
`infer_baseline.py` manifests this script was written against) — additive
only, the original 26-method run's behaviour and output are unchanged.

Same A06-outlier discipline as this ticket's own headline table (A06
excluded uniformly from every method, not each method's own inconsistently-
triggered auto-flag): recovery delta = accuracy(method) −
accuracy(raw_hamamatsu), n=416/method excl. A06. **This run's own
`raw_hamamatsu` excl-A06 accuracy (0.4135) matches this ticket's original
headline-table value exactly** — confirms it's scored on the same
underlying labelled crop set, so the comparison below is apples-to-apples
with the table above, not a separately-scaled number.

| Method | acc (excl. A06) | Recovery Δ |
|---|---|---|
| **P3-07** (SDXL, native 1024×1024, `tickets/PHASE3-TICKETS.md`) | 0.4904 | **+0.0769** |
| **P1-16** (raw-source-detail/colour-residual fusion, `tickets/P1-16_source_detail_colour_residual_fusion.md`) | 0.4423 | **+0.0288** |
| P1-11 (DDIM-inversion, `tickets/PHASE1-TICKETS.md`) | 0.4231 | +0.0096 |
| P3-06 (SDXL transfer, `tickets/PHASE3-TICKETS.md`) | 0.4207 | +0.0072 |
| P1-10 (source-conditioned, `tickets/PHASE1-TICKETS.md`) | 0.4199 | +0.0064 |
| raw_aperio (sanity check) | 0.4279 | +0.0144 |
| raw_hamamatsu (baseline) | 0.4135 | 0.0000 |

**P3-07 (+0.0769) is now the single best recovery-delta result in this
project on this metric — beats both the original headline diffusion config
(A3@0.50, +0.0625) and every classical baseline (best: Reinhard +0.0457).
Read this together with `tickets/PHASE3-TICKETS.md` P3-07's own colour
result, not instead of it: P3-07's colour recovery is Δlab=-5.60 —
negative, confirmed real (D1 diagnostic ruled out a baseline artefact),
every slide flips negative — so this is not a scanner-normalisation win.**
Two things line up to explain it instead: P3-07 has the best structural
SSIM of any SDXL config (0.4313, closing most of the gap to SD1.5's
0.4485), and atypia scoring is a morphology task, not a colour-matching
one — the same "structure over colour" pattern already flagged below for
A1; and P3-07's negative colour recovery means its output stayed closer
to Aperio's own colour statistics than to Hamamatsu's, and this
classifier's own raw_aperio sanity check already scores higher (0.4279)
than real raw_hamamatsu (0.4135) — i.e. it's already more comfortable
with Aperio-flavoured colour. **Not evidence P3-07 is the best
normalisation method — evidence the classifier rewards structure +
Aperio-like colour, which P3-07 does best of the five tested, partly
because its colour normalisation is broken.**

P1-16 (+0.0288) sits between histogram matching (+0.0024) and Reinhard
(+0.0457) — unlike P3-07, P1-16's colour recovery is genuinely positive
(Δlab +7.81, ~89% of P1-11's own), so this one IS a case of the
classifier rewarding a method that also actually normalises colour.
**P1-10, P1-11, and P3-06 are all weak here — barely above raw_hamamatsu
and well below every classical baseline**, despite P1-10/P1-11 being this
project's best-ever results on SSIM/LAB-recovery; this downstream-
classifier metric evidently does not track structural/colour fidelity the
way those metrics do (consistent with this ticket's own caveat above that
this is a noisy instrument measuring something distinct from the rest of
the project's metrics).

A06 remains dramatically easier than the other four slides for every
method on this classifier, including the sanity checks (`raw_aperio` A06
acc=0.984, `raw_hamamatsu` A06 acc=0.484 — the two sanity checks disagree
by 50 points on A06 alone) — reinforces that A06 must never be pooled in
without exclusion for this metric specifically, more so than for the
structural/colour metrics.

Full per-crop and per-slide data: `/datasets/mhoosen/stain-norm/eval/
atypia_classifier_scores_p1_p3_p16/{summary.csv,per_crop.csv}`.

## P2-10 — CAMELYON17 multi-centre generalisation
**Status:** ✅ DONE (2026-08-20) — FAIL verdict. D_pre=56.70, D_post=57.52,
only 1/10 centre pairs improved. H2/RQ4's cross-hospital generalisation claim
does not hold at the P1-09 best operating point. Full table below.
**Source:** `sec:experiments` "Colour Accuracy" — CAMELYON17 subsection; `tab:camelyon`
**Description:** Pairwise LAB Wasserstein between all 5 centres, before (D_pre, 10
distances) and after (D_post, 10 distances) normalisation. Success = D_post < D_pre.
3 patients/centre, 100 random 512×512 patches/patient, pooled per centre.

**Extraction (local, not cluster — reasoning below):** `data/camelyon17/
camelyon_patches.py` is a pre-existing local extraction pipeline (tiatoolbox +
a Philips-tifffile fallback for one scanner brand) that already had 1/15
patients extracted before this session. **Why local, not cluster**: raw
CAMELYON17 is 232GB, genuinely too large to move — per-patient extraction
(tissue-thresholded 512×512 patch sampling from whole-slide TIFFs) is
disk/CPU-bound classical image processing, not model training/inference, so it
doesn't collide with CLAUDE.md's "never train/infer locally" rule the way
running the actual diffusion pipeline would; only the small resulting patch set
(order ~1GB) needed to reach the cluster, not the raw slides.

**Two real bugs found and fixed during extraction (not hypothetical, hit
directly on real files):**
1. **OOM on large Philips-format slides.** The original fallback path
   (`series.levels[0].asarray()`) tried to materialise a whole slide into RAM —
   confirmed needing 52.1GB/40.9GB for two of `patient_080`'s nodes, on a 32GB
   local machine. Fixed by switching to a zarr-backed lazy read
   (`level.aszarr()`) that only loads the sampled 512×512 window per attempt —
   verified directly against a real node file (non-degenerate 512×512×3 uint8
   patch, values 150–249) before trusting it at scale. One further wrinkle:
   `aszarr()` on a single pyramid level still returns a multiscale zarr
   *Group* keyed `'0'`..`'N'`, not a plain Array — has to be descended into
   (`group['0']`) explicitly, confirmed by direct inspection, not assumed.
2. **Attempt-budget undershoot on sparse-tissue nodes.** `MAX_ATTEMPTS_MULT`
   raised from 25→60 (500→1200 random-crop attempts per node before giving up)
   — materially improved yield on most patients, but `patient_061` stayed
   genuinely low (47/100 total across two extraction passes) even after a
   dedicated top-up attempt: 4 of its 5 nodes are near-empty regardless of
   attempt budget (confirmed: 0/1/1/20/0 patches per node on the top-up run) —
   real sparse tissue on that slide, not a bug, accepted as-is.

**Disk management (local):** only 122.6GB free vs. potentially 165GB needed to
unzip all 10 remaining patient archives at once — processed **one patient at a
time** (unzip → extract ~100 patches → delete that patient's large `.tif`,
keep the zip as a re-extractable backup — user-confirmed policy) rather than
unzipping everything upfront. Final result: **all 15 patients across 5 centres,
2,103 patches total, 863MB** (well above the ticket's 1,500-patch target).

**Upload (this session):** `scp -r` to `/datasets/mhoosen/stain-norm/
camelyon17_patches/`, verified byte-for-byte — file count (2,103) and total
size (863MB) match the local copy exactly.

**D_pre (job 44165, `mscluster23`, COMPLETED 2:01):** `src/eval/
score_camelyon_wasserstein.py` pools every patch for a centre by
`np.concatenate` (not `np.stack` — `metrics.py`'s `lab_wasserstein` calls
`cv2.cvtColor` internally, which rejects a 4D stacked array with "Bad number of
channels" — this was the plan's original assumption and it was wrong, caught
and fixed by testing directly against real uploaded patches before writing the
real script, then independently re-verified: self-distance ≈0.19 near-zero,
cross-centre ≈34.5 real, on a separate local check). All `C(5,2)=10` pairwise
distances computed on the raw patches:

| pair | LAB total |
|---|---|
| centre_0 vs centre_1 | 37.03 |
| centre_0 vs centre_2 | 42.55 |
| centre_0 vs centre_3 | 27.69 |
| centre_0 vs centre_4 | 65.67 |
| centre_1 vs centre_2 | 74.85 |
| centre_1 vs centre_3 | 22.68 |
| centre_1 vs centre_4 | 99.85 |
| centre_2 vs centre_3 | 62.67 |
| centre_2 vs centre_4 | 42.05 |
| centre_3 vs centre_4 | 91.93 |
| **mean** | **56.70** |

Verified `pairwise.csv` has exactly 10 rows before trusting this table.

**Normalisation (job 44203 smoke test, then job 44283, `mscluster55`,
COMPLETED 15:04):** all 2,103/2,103 patches normalised at the same P1-09 best
general-purpose config used throughout this project (colour LoRA + ControlNet
+ LCM, A2H direction, strength 0.20), applied uniformly across all 5 centres.
Verified non-degenerate output (real pixel stats, not blank/NaN).

**D_post (job 44364, `mscluster22`, COMPLETED 1:53):**

| pair | D_pre | D_post | improved? |
|---|---|---|---|
| centre_0 vs centre_1 | 37.03 | 36.80 | ✓ |
| centre_0 vs centre_2 | 42.55 | 43.35 | ✗ |
| centre_0 vs centre_3 | 27.69 | 28.20 | ✗ |
| centre_0 vs centre_4 | 65.67 | 66.63 | ✗ |
| centre_1 vs centre_2 | 74.85 | 76.36 | ✗ |
| centre_1 vs centre_3 | 22.68 | 23.08 | ✗ |
| centre_1 vs centre_4 | 99.85 | 100.88 | ✗ |
| centre_2 vs centre_3 | 62.67 | 64.40 | ✗ |
| centre_2 vs centre_4 | 42.05 | 42.17 | ✗ |
| centre_3 vs centre_4 | 91.93 | 93.36 | ✗ |
| **mean** | **56.70** | **57.52** | **1/10** |

**Verdict: FAIL** (`D_post_mean >= D_pre_mean`). Not a borderline/noisy result
— 9 of 10 pairs got slightly *worse*, and the one improved pair only moved by
0.23 (36.80 vs 37.03), a small fraction of typical pair-to-pair variation.
H2/RQ4's cross-hospital generalisation claim does not hold at this operating
point: a colour LoRA trained on exactly one scanner pair (Aperio/Hamamatsu),
applied uniformly to 5 completely unseen hospital centres, does not pull them
closer together — if anything it adds small, inconsistent per-centre drift
that doesn't converge. Consistent with the project's broader pattern (H2's
first half already showed the pipeline losing to classical methods even on
its trained domain; this shows it doesn't transfer a colour-normalising
effect to novel scanners either). Verified `pairwise.csv`/
`d_pre_post_comparison.csv` both have exactly 10 rows before trusting this
table.

**Status: DONE.**

## P2-11 — Baseline method comparisons
**Status:** ✅ Macenko/Reinhard/Histogram Matching DONE (2026-08-10, jobs 40521/
40619/40628 score). StainNet/StainGAN/ParamNet still TODO — no existing code in
this repo, needs new inference wrappers + pretrained weights.
**Source:** `tab:baselines`
**Description:** Macenko, Reinhard, Histogram Matching, StainNet, (pretrained StainGAN
if reproducible), ParamNet — run against the same held-out set and clinical classifier
for a fair comparison table. (Clinical classifier arm blocked on P2-09, not yet built.)
**Code:** `src/eval/baseline_methods.py` (verified against a reference Macenko/
Reinhard implementation, one documented deviation — closed-form pseudo-inverse
instead of constrained LASSO for Macenko's concentration solve), `src/eval/
infer_baseline.py` (CPU-only, writes the same manifest schema as
`infer_colour_lora.py` so `score_outputs.py` needed zero changes), `slurm/
infer_baseline.slurm` (runs on `stampede`, no GPU).
**Results:** full table + methodological caveat in `docs/results/RESULTS_SUMMARY.md`
("Classical baseline comparison (P2-11)"). Headline: all three classical methods
beat every diffusion rung on A06 recovery (histogram_matching +69.12 vs best
diffusion A5\@0.70's +24.43), **but this is confirmed to be a metric-construction
confound, not a clean win** — verified directly (not just from the summary CSV)
that `match_histograms` forces near-identical output color statistics onto every
crop regardless of content (checked across 5 slides), and the eval metric
(`lab_wasserstein`) is itself a pure marginal-distribution distance, so the
method and the metric optimize close to the same quantity by construction.
Macenko/Reinhard fit summary statistics rather than the full histogram, so they
carry a milder version of the same bias (smaller but still large A06 numbers).
**Any write-up conclusion from this table must state the confound explicitly** —
do not present "classical beats diffusion" as a clean finding on its own.
**Follow-up (2026-08-10):** the spatial/local metric this caveat called for
(windowed LAB-Wasserstein, 64×64 tiles) has been implemented, backfilled across
all 18 configs via a full rescoring pass, and tested directly against the
confound. Result: the confound is real (classical's global-vs-windowed gap is
~1 LAB unit larger than the raw baseline's) but small — it does not overturn
classical's advantage. Windowed recovery deltas: Macenko +8.09, Reinhard
+5.08, Histogram Matching +2.41, vs best diffusion config only +0.18 (several
diffusion configs go *negative* under the windowed metric even though their
global delta is positive). A second new metric, CIEDE2000, was also added but
turned out to be redundant with SSIM/PSNR/MAE (P2-06) rather than an
independent check — see `docs/results/RESULTS_SUMMARY.md`'s P2-11 follow-up
section for both full tables. Net: across three independent metric families
(global colour, windowed colour, pixel-exact structure), classical wins —
report this as a genuine finding, not an artefact to explain away.

---

## P2-12 — Harden P2-09 clinical-utility / atypia-classifier evaluation

**Status:** 🔄 IN PROGRESS (2026-08-28) — code hardening implemented and
smoke-tested (job 47545): direction gate, strict pairing, frame-level
aggregation, seed-averaging, batched inference, integrity report all
verified against real data (12/15 acceptance criteria confirmed; 3 need a
`train_atypia_classifier.py` smoke run). No real rerun submitted yet. Full
ticket: `tickets/P2-12_atypia_classifier_evaluation_hardening.md`.

**Motivation:** before treating P2-09's results (the original 26-method
table or this session's P1-10/P1-11/P1-16/P3-06/P3-07 extension above) as
final, close several methodological/robustness gaps in `score_atypia_
classifier.py`/`train_atypia_classifier.py` — unpaired recovery_delta,
crop-level (not frame-level) primary metric, per-method outlier policy,
unbounded GPU batching, raw-accuracy checkpoint selection, unenforced
manifest assumptions.

**Critical finding surfaced while writing the spec, not something this
ticket set out to find:** empirically confirmed (job 47468, a real A0
output is 2x closer in pixel-MAE to its raw-Aperio source than to its
Hamamatsu reference) that every method ever scored by `score_atypia_
classifier.py` — A0-A5, the classical baselines, and this session's P1-10/
P1-11/P1-16/P3-06/P3-07 extension — has run **A→H** (Aperio input,
Hamamatsu-styled output), the opposite of the proposal's own stated design
(§"Clinical Utility": classifier trained on Aperio, evaluated on raw
Hamamatsu *and* "the proposed A0-A5 outputs" as parallel substitutable
items — only coherent if those outputs are H→A). **Every existing P2-09
recovery_delta number, including this session's extension above, must be
treated as not supporting the proposal's clinical-utility claim as
currently computed** — full evidence, mechanism, and exactly what a
corrected rerun costs per architecture (cheap re-inference for A0-A5;
expensive new training for P1-10/P1-11/P3-06/P3-07/P1-16) in the ticket
file's §0 and §11. Does **not** affect `score_outputs.py`'s SSIM/LAB/
colour-recovery numbers for any of these methods — those remain valid.

---

**Reminder (CLAUDE.md):** always report both the pooled `ALL` aggregate and the
robust-outlier-excluded aggregate. A06 is a confirmed genuine colour-gap outlier —
never let it silently dominate a headline number.
