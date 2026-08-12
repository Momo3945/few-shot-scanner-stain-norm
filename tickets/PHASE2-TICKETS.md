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
**Status:** TODO — depends on both A2H and H2A LoRAs at the chosen rank (P1-03a/c done
for rank 8)
**Source:** `sec:experiments` "Round Trip Reconstruction"
**Description:** A06 → (LoRA A→H) → Ĥ06 → (LoRA H→A) → Â06 ↔ Original A06.
**IMPORTANT (proposal is explicit):** this test uses a **deterministic 50-step DDIM
sampler**, NOT LCM — LCM's stochastic drift would contaminate the structural deviation
measurement. LCM is reserved strictly for the unidirectional deployment pipeline.

## P2-08 — Structural safety: HoVer-Net vs Lizard (Relative Dice)
**Status:** IN PROGRESS (2026-08-12) — wrapper code written, smoke-tested
end-to-end (inference + Dice(A,G) scoring both run successfully on 5 images).
Full 130-image Dice(A,G) gate and the normalisation (B) side are still open.
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
**Next step:** full run — `sbatch slurm/infer_hovernet.slurm lizard_original ""
"" ` (no limit, all 130 `dpath_*`+`glas_*` images) then `sbatch
slurm/score_lizard.slurm lizard_original` again (overwrites the 5-image
summary with the full-130 one) for the real Dice(A,G) gate value. Step 5 (the
normalisation side, B) still needs design work before Relative Dice can be
computed. Will ask for confirmation with the exact command before submitting,
per project rules.

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

**Reminder (CLAUDE.md):** always report both the pooled `ALL` aggregate and the
robust-outlier-excluded aggregate. A06 is a confirmed genuine colour-gap outlier —
never let it silently dominate a headline number.
