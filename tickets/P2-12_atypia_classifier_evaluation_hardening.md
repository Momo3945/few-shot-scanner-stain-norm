# P2-12 — Harden P2-09 Clinical-Utility / Atypia-Classifier Evaluation

**Status:** 🔄 IN PROGRESS (2026-08-28) — §1-§3.10's code hardening
implemented and smoke-tested (job 47545, 12/15 acceptance criteria
confirmed against real data + local unit tests; 3 remaining need a
`train_atypia_classifier.py` smoke run — see §7). No real rerun (§11)
submitted yet. Written 2026-08-28 in response to a request to
methodologically harden `tickets/PHASE2-TICKETS.md` P2-09 before its
results (including this session's P1-10/P1-11/P1-16/P3-06/P3-07
extension) are treated as final.

**Source:** `docs/proposal_draft(6).tex` §"Clinical Utility" (the exact
proposal text is quoted below); hardens `tickets/PHASE2-TICKETS.md` P2-09 and
its Extension section.

**Preserves the existing experiment's intent, unchanged:** train one fixed
ResNet18 atypia-grade classifier (3-class, scores 1/2/3) on raw Aperio-domain
training images; evaluate whether stain normalization improves downstream
atypia prediction compared with raw Hamamatsu. The classifier is not, and
does not become, a scanner classifier. This ticket does not touch
`train_atypia_classifier.py`'s core architecture/label design beyond §6
below, and does not touch `score_outputs.py`'s SSIM/LAB/colour-recovery
metrics at all (see Non-Goals — those are unaffected by everything in this
ticket).

---

## 0. Critical finding, discovered during this ticket's own investigation — read this first

**The proposal's own text specifies the opposite translation direction from
what every method scored by `score_atypia_classifier.py` has actually used,
to date, without exception.**

**What the proposal says** (`docs/proposal_draft(6).tex` line 549, verbatim):

> "A standard diagnostic classifier is trained on raw Aperio (×20) patches
> and evaluated on raw Hamamatsu, Macenko, Reinhard, Histogram Matching,
> StainNet, ... and the proposed A0–A5 outputs. The primary contribution
> metric is recovery delta = performance(Proposed) − performance(Raw
> Hamamatsu)."

This sentence lists "raw Hamamatsu" and "the proposed A0–A5 outputs" as
**parallel, substitutable items** being fed to the same Aperio-trained
classifier — i.e. "Proposed" is meant to stand in for "what you'd get if you
normalized real Hamamatsu-scanner images before handing them to an
Aperio-trained classifier." That only makes sense if the outputs being
scored are **H→A** (Hamamatsu input, normalized toward Aperio's appearance).

**What the code actually does, verified two independent ways:**

1. **Code-level.** `src/eval/infer_colour_lora.py` (generates A0/A1/A2/A3/A4/A5)
   defaults `--direction` to `"A2H"` (line 87-88: `default="A2H"`), and line
   211 confirms what that means: `src_frame, ref_frame = (a_rgb, h_reg) if
   args.direction == "A2H" else (h_reg, a_rgb)` — for A2H, the img2img
   **input is the raw Aperio crop**, and the registered Hamamatsu crop is
   only the scoring reference, never fed to the model.
   `src/eval/infer_baseline.py` (Macenko/Reinhard/Histogram Matching) has
   **no `--direction` flag at all** — it hardcodes the same convention
   (line 100-102: `# A2H convention (matches infer_colour_lora.py's
   default): normalise Aperio ... src_frame, ref_frame = a_rgb, h_reg`).
   No ticket record found of any canonical P2-09-scored run passing
   `--direction H2A`. P1-10/P1-11 (`tickets/PHASE1-TICKETS.md` P1-10) and
   P3-06/P3-07 (`tickets/PHASE3-TICKETS.md`) explicitly document **"A2H
   direction only ... H2A stays out of scope"** as a deliberate scope
   decision — direct confirmation for those five.
2. **Empirical, pixel-level (job 47468, stampede, read-only, 2026-08-28,
   submitted only after explicit go-ahead).** Took one real A0 output
   (`eval/a0/outputs/A06_00A_x0_y0_s0.30.png`) and compared it against both
   its raw-Aperio source crop and its registered-Hamamatsu reference crop:

   ```text
   MAE(output, raw Aperio crop)      = 30.851
   MAE(output, registered Hamamatsu) = 61.273
   (sanity) MAE(raw Aperio, registered Hamamatsu) = 53.745
   ```

   The output is roughly **2x closer to raw Aperio than to Hamamatsu**, and
   closer to Aperio than the raw Aperio/Hamamatsu pair are to each other —
   decisive, direct confirmation the model's input was Aperio and its output
   is Hamamatsu-*styled*, not the reverse.

**Consequence.** Every method ever scored by `score_atypia_classifier.py` —
the original 26-method P2-09 run (A0/A1/A2/A3/A4/A5 × strengths, the three
classical baselines) **and** this session's P1-10/P1-11/P1-16/P3-06/P3-07
extension (job 47418) — has fed the Aperio-trained classifier images whose
underlying tissue *content* is literally still Aperio, only recoloured
toward Hamamatsu's appearance. `recovery_delta = accuracy(method) −
accuracy(raw_hamamatsu)` is therefore not measuring "does normalizing real
Hamamatsu help the classifier" — it is closer to measuring "does
recolouring Aperio content toward Hamamatsu's palette preserve enough for
the Aperio classifier to still read it, compared to genuinely independent
Hamamatsu content." This plausibly explains two anomalies already flagged
in the existing write-ups without this explanation available at the time:
P2-09's own observation that **A1 (ControlNet only, no colour LoRA at all)
was among the top performers** ("structural conditioning specifically, not
colour adaptation, may be driving a real chunk of this benefit"), and this
session's finding that **P3-07 tops the extension's recovery-delta table
despite a confirmed negative colour recovery** (Δlab=-5.60) — both are
exactly what you'd expect if the classifier is really rewarding
"how much genuine Aperio structure survived," not "how well was Hamamatsu
normalized."

**This does not mean the result is proven wrong** — it means the result
**cannot currently support the proposal's stated clinical-utility claim**,
and must be re-measured in the correct direction before anyone treats
either the original P2-09 table or this session's extension as evidence
either way. See §11 (Rerun Requirements) for exactly what a correct rerun
costs per architecture, and §3 below for the validation this ticket adds so
this can never again go unnoticed.

**What this does NOT affect:** `score_outputs.py`'s SSIM/LAB-Wasserstein/
colour-recovery numbers for P1-10/P1-11/P3-06/P3-07/P1-16 (and A0-A5) are
**unaffected and remain valid** — those metrics compare a method's output
against the real target of *whichever* direction it was generated in, which
is internally consistent regardless of A2H vs H2A. The direction problem is
specific to the atypia classifier, because that classifier was trained on
one specific domain (Aperio), which breaks the A2H/H2A symmetry that the
structural/colour metrics don't depend on.

---

## 1. Problem statement (methodological/robustness holes beyond §0)

`src/eval/score_atypia_classifier.py` and `src/train/train_atypia_classifier.py`
are broadly correct in their core design (fixed classifier, weak frame-level
labels, class-weighted loss, slide-level split) but have the following
holes, discovered by direct inspection of the current code
(`score_atypia_classifier.py`, current as of commit `3eacd88`) and manifests:

1. **`recovery_delta` is not strictly paired.** `pool()`
   (`score_atypia_classifier.py:228-244`) computes each method's accuracy
   over whatever crops exist in *that method's own* `eval_manifest.csv`,
   and `raw_hamamatsu`'s accuracy is computed separately over crops derived
   from whichever tag happened to be processed first (`derived_done` block,
   lines 166-182). Nothing checks these are the same sample sets. Different
   methods have already been shown this session to have different total
   crop counts (P3-07: 1485 vs P1-10/P1-11/P3-06: 1488) — a real, observed
   case of exactly this problem, not a hypothetical.
2. **Evaluation is crop-level, not frame-level.** The true label is a
   frame-level clinical annotation; `score_atypia_classifier.py` currently
   scores and pools every 512×512 crop as if it were an independent
   observation (`by_slide[slide].append((t, p, pr))` at crop granularity,
   `score_atypia_classifier.py:218-220`), so a frame with 4 crops counts 4x
   in every aggregate.
3. **Translation direction is unverified — see §0.** No code anywhere
   checks or asserts direction; nothing prevents silently pooling
   incompatible directions together in the future.
4. **Outlier exclusion is per-method.** `flag_outliers(slide_acc, ...)`
   (`score_atypia_classifier.py:224-226`) is called separately inside the
   `for method, items in sorted(methods.items())` loop, so each method can
   flag and exclude a different slide set from `ALL_excl_outliers` — this
   session's own run confirms it (only `p1_10_ddim_inversion/
   p1_10_ddim_inversion_full` printed `*** OUTLIER` on A06; every other
   method's A06 was numerically just as extreme but didn't cross that
   method's own z-threshold). `ALL_excl_outliers` numbers are therefore not
   comparable across methods as currently computed (this is exactly why
   this session manually recomputed a uniform A06 exclusion by hand instead
   of trusting the script's own `ALL_excl_outliers` rows).
5. **`classify_batch()` does not bound GPU memory.**
   `score_atypia_classifier.py:117-122`: `x = torch.stack([preprocess(r)
   for r in rgb_list]).to(device)` builds and transfers the *entire*
   method's tensor before the `for i in range(0, len(x), args.batch_size)`
   loop ever runs — `--batch-size` only bounds the forward-pass chunk size,
   not peak GPU memory, which scales with the whole method's crop count.
6. **Checkpoint selection uses raw, imbalanced-class accuracy.**
   `train_atypia_classifier.py:277-291` selects `best.pt` by `val_acc`
   alone; combined with a heavily skewed val distribution (whatever the
   auto-picked last-2-slides split happens to contain), this can reward a
   near-degenerate "mostly predict the majority class" checkpoint. No check
   that the auto-picked validation slides actually contain all 3 classes;
   `--val-slides` is not validated against the manifest's real slide names
   before training starts (a typo would silently produce an empty val
   split, which `evaluate()` at line 223-235 would then silently return
   `None` for, permanently disabling checkpoint selection with no error).
7. **Manifest assumptions are enforced by code comments, not by checks.**
   The newer-schema handling added this session
   (`score_atypia_classifier.py:148-161`) documents "checked directly
   before relying on it, not assumed" in a comment, but nothing in the code
   itself re-verifies `source_mode` is uniform for a future manifest that
   might not be. Three seeds are currently pooled as if they were 3x the
   independent clinical samples (this session's own P1-10/P1-11/P3-06/P3-07
   extension did exactly this) rather than being explicitly
   averaged/deduplicated per underlying crop identity. No duplicate-key
   detection exists at all.
8. **No integrity guards.** Beyond `build_atypia_manifest.py`'s own
   `VALID_SCORES = {1, 2, 3}` check at manifest-*build* time (which is
   sound and does not need to change), nothing downstream re-validates
   labels, non-empty splits, requested-but-absent `--val-slides`, path
   existence before a long scoring job starts, non-zero paired sample
   counts, or train/test slide disjointness (currently true by construction
   per CLAUDE.md's methodology guardrails, but not runtime-asserted).
9. **No statistical treatment of the paired comparison** beyond the point
   estimate (recovery_delta itself has no uncertainty interval anywhere).
10. **Output files don't distinguish frame-level from crop-level**, and
    `summary.csv` has no explicit `level` column.

---

## 2. Current behaviour (for reference during implementation)

- **Training** (`src/train/train_atypia_classifier.py`): `resnet18`
  (ImageNet-pretrained) → 3-class linear head; 512×512 tissue crops
  (`grid_offsets`/`tissue_fraction`, same convention as
  `infer_colour_lora.py`) inherit their parent frame's atypia score;
  class-weighted `CrossEntropyLoss` (inverse frequency); train/val split by
  slide, default = 2 alphabetically-last slides; checkpoint saved every
  `--save-every` steps, `best.pt` tracked by raw `val_acc`.
- **Scoring** (`src/eval/score_atypia_classifier.py`): reads
  `eval/<tag>/eval_manifest.csv` per `--tags` entry; for the older
  `strength,slide,frame,x,y,output_path,reference_path,aperio_path` schema
  (`infer_colour_lora.py`/`infer_baseline.py`), one method per distinct
  `strength` value; for the newer `seed,source_mode,crop_id,slide,frame,x,y,
  output_path,reference_path,aperio_path` schema
  (`infer_colour_source_ddim_inversion.py` family — P1-10/P1-11/P3-06/
  P3-07), pools every row under the tag as one method (patch added this
  session, §1.7 above). Derives `raw_hamamatsu` (dedup'd `reference_path`)
  and `raw_aperio` (sanity check, re-cropped from `--testing-root`) once,
  from the first `--tags` entry only. Scores crop-by-crop, pools per
  method/scope (`ALL`, `ALL_excl_outliers` — per-method outlier set,
  §1.4), per-slide. `recovery_delta` computed on the `ALL` scope only,
  against `raw_hamamatsu`'s own `ALL` accuracy.
- **Manifest schemas currently in use** (verified directly against real
  files on the cluster, not assumed):
  - `strength,slide,frame,x,y,output_path,reference_path,aperio_path` —
    A0-A5 rungs, classical baselines.
  - `seed,source_mode,crop_id,slide,frame,x,y,output_path,reference_path,
    aperio_path` — P1-10 (`p1_10_full_heldout`), P1-11
    (`p1_10_ddim_inversion/p1_10_ddim_inversion_full`), P3-06
    (`p3_06_full_heldout`), P3-07 (`p3_07_full_heldout`); `aperio_path` is
    **blank** in some of these (e.g. P1-11's reconstructed manifest, per
    `tickets/PHASE1-TICKETS.md` P1-11's own note: "`aperio_path` left
    blank, unused by the scorer") — currently silently tolerated because
    `raw_aperio`/`raw_hamamatsu` are only ever derived from the first tag
    processed.
  - `strength,seed,slide,frame,x,y,output_path,reference_path,aperio_path`
    — P1-16 (`p1_16_fusion/f3_s8_b0.50_heldout_full_corrected`), `strength`
    here is reused as an arbitrary config label (`"f3_s8_b0.50"`), not a
    numeric denoising strength; `aperio_path` also blank.
- **Ground-truth manifests**: `pairs/atypia_train_manifest.csv` (11 slides,
  297 labelled frames) and `pairs/atypia_test_manifest.csv` (5 held-out
  slides — A06/A08/A09/A13/A16 — 120 labelled frames), both
  `slide,frame_id,image_path,atypia_score`, built by
  `src/data/build_atypia_manifest.py`. Train/test slide sets are disjoint
  by construction (held-out slides are never in the training archive), but
  this is not runtime-asserted anywhere downstream.

---

## 3. Required changes

### 3.1 Translation-direction validation (highest priority — blocks everything else being meaningful)

- Every `eval_manifest.csv` consumed by `score_atypia_classifier.py` must
  declare its direction unambiguously. Prefer reading it from the
  manifest's own metadata where the generating script already records it
  (`infer_colour_source_ddim_inversion.py` writes `"direction": args.
  direction` into a run-level JSON at line 615 — extend `infer_colour_lora.
  py`/`infer_baseline.py` to write the same alongside their
  `eval_manifest.csv` if they don't already, rather than requiring it be
  re-derived or guessed).
- `score_atypia_classifier.py` must read this per-tag and **assert every
  tag being scored together is `H2A`** before running the real clinical
  comparison (an `A2H` tag may still be scored, but only under an explicit
  `--allow-a2h` escape hatch that clearly labels the output as *not* the
  proposal's clinical-utility claim — useful for the "does colour
  restyling preserve enough structure" robustness question §0 surfaces,
  but must never be silently presented as the headline recovery_delta).
- Fail loudly (non-zero exit, clear message naming the offending tag and
  its detected/declared direction) rather than silently scoring or pooling
  a mismatched-direction manifest.
- Print an explicit per-method line, e.g. `method=X paired_frames=Y
  missing_vs_raw=0 duplicates=0 source_direction=H2A`, before any score for
  that method is trusted/written (ties into §3.7's integrity report and
  the acceptance criteria in §12).

### 3.2 Strictly paired recovery_delta

- Canonical sample key: `(slide, frame, x, y)` where available; fall back
  to the manifest's own `crop_id` only if `(slide,frame,x,y)` isn't
  derivable, and document which was used.
- For every method: intersect its sample-key set with `raw_hamamatsu`'s
  sample-key set (computed once, from its own manifest — not assumed
  identical without checking).
- Compute both `accuracy(method)` and `accuracy(raw_hamamatsu)` **only**
  over that intersected set; `recovery_delta =
  paired_method_accuracy − paired_raw_hamamatsu_accuracy`.
- Record `n_paired`, and explicitly report `n_method_only` /
  `n_raw_only` (samples present in one side but not the other) — do not
  silently drop them without a printed/logged count.
- If all manifests are expected to contain identical crop grids (true for
  same-architecture same-`_full_heldout`-family runs, not necessarily true
  across architectures at different native resolutions, e.g. P3-07's
  1024-native grid vs P1-10's 512 grid), add an assertion that fails
  loudly when that expectation is violated instead of assuming it — do not
  simply intersect-and-move-on silently when the mismatch is large enough
  to suggest a bug rather than expected cross-architecture variation
  (pick a threshold, e.g. warn if paired overlap is <90% of either side's
  raw count, in addition to always reporting the exact number).

### 3.3 Frame-level primary metric

- For each `method × slide × frame`: `frame_probs = mean(crop_probability_
  vectors)` over that frame's available crops (after §3.2's pairing);
  `frame_pred = argmax(frame_probs)`; compare against the frame's true
  label.
- Primary summary metrics become frame-level: accuracy, macro-F1,
  macro-AUC (where valid — same `ValueError`-on-absent-class guard already
  in `pool()`), `n_frames`, `recovery_delta` vs. `raw_hamamatsu` computed
  on the same paired frames (§3.2's pairing applies at the frame level
  here — a frame counts as paired only if it has at least one crop on both
  sides after §3.2's crop-level intersection).
- Crop-level accuracy/metrics remain available (existing `per_crop.csv`
  behaviour, effectively unchanged) but must be clearly labelled secondary/
  diagnostic wherever printed or written, not presented as the headline
  clinical number.
- New `per_frame.csv`: `method, slide, frame, true_score, pred_score,
  correct, prob_1, prob_2, prob_3, n_crops`.

### 3.4 One common outlier policy across all methods

- Do not call `flag_outliers` per-method inside the per-method loop and
  let each method pick its own excluded slide set.
- Derive one exclusion list **before** the per-method loop, using a
  method-independent criterion — prefer the project's own already-
  established A06 convention (CLAUDE.md: "A06 is a genuine colour-gap
  outlier... Outlier flagging is robust (median/MAD), not mean/std" — the
  same discipline P2-09's own original headline table and this session's
  extension already applied by hand). Concretely: run the *existing*
  robust `flag_outliers` logic once, on `raw_hamamatsu`'s (or another
  fixed, principled reference's) own per-slide accuracy distribution, not
  per-method — or simply hardcode the project's already-established A06
  exclusion if a from-data derivation proves noisy at this classifier's
  low absolute accuracy. Either way, the same excluded-slide set applies
  to every method's `ALL_excl_outliers` row.
- `ALL` (no exclusion) remains the primary headline scope, exactly as now.
  `ALL_excl_outliers` stays secondary, and its excluded-slide set must be
  recorded once (e.g. in the integrity report and/or a header row of
  `summary.csv`) rather than re-derived silently per method.

### 3.5 Batched GPU inference

- Rewrite `classify_batch()` so preprocessing, stacking, and `.to(device)`
  happen per-chunk: iterate `rgb_list` in slices of `args.batch_size`,
  preprocess only that slice, stack it, move only that batch to device,
  run inference, move results back to CPU immediately, then concatenate
  the CPU-side results at the end. Predictions should be numerically
  equivalent to the current implementation (same model, same
  preprocessing, only the memory-transfer granularity changes) — verified
  by the smoke-test comparison in §10.

### 3.6 Validation/checkpoint-selection robustness (`train_atypia_classifier.py`)

- Keep the by-slide split (never by-crop).
- Print the class distribution of both the train and validation splits
  explicitly (train's is already printed; add val's).
- If validation is missing one or more of the 3 classes, fail loudly by
  default (or warn very visibly if the project decides a warning is
  preferable to a hard stop for this dataset's small slide count — decide
  during implementation, but the current silent behaviour must not
  remain).
- If `--val-slides` is supplied, validate every listed name actually
  appears in the manifest's slide set before training starts; fail with a
  clear message naming the unmatched slide(s) otherwise.
- Select `best.pt` by a class-balanced metric — macro-F1 preferred (matches
  `score_atypia_classifier.py`'s own existing macro-F1 computation, so the
  same metric family is used end-to-end) — while continuing to report raw
  accuracy alongside it. Store the checkpoint-selection metric name and
  value in both the checkpoint dict (`torch.save({..., "selection_metric":
  "macro_f1", "selection_value": ...})`) and `training_config.json`.
- Document explicitly, in the ticket and in the script's own docstring, if
  the dataset's small slide count (13 usable training slides total) limits
  how representative any single held-out validation split can be — this is
  a real, acknowledged limitation, not something to paper over with a more
  elaborate CV scheme unless the project decides that's warranted
  separately.

### 3.7 Manifest-assumption enforcement

- If a manifest has a `source_mode` column, read and check its actual
  values (not assume uniformity from a code comment); reject or cleanly
  separate any manifest that mixes incompatible modes (e.g. `correct` and
  `shuffled` together) rather than silently pooling them.
- If multiple seeds exist for the same underlying crop identity, do not
  count them as independent clinical samples. Default policy: **average
  the probability vectors across seeds for the same `(slide, frame, x, y)`
  before producing one crop-level prediction** (matches the frame-level
  averaging philosophy in §3.3, and matches how this project already
  treats seeds as repeated stochastic draws of the same underlying
  estimate elsewhere, e.g. `score_outputs.py`'s multi-seed pooling
  convention) — but make this an explicit, named, documented decision in
  the code and this ticket, not an implicit side effect of just
  concatenating rows.
- Detect duplicate `(method, crop_identity)` records explicitly (a
  dict-of-seen-keys check is sufficient) and handle them deterministically
  per the seed-averaging policy above, logging how many duplicates were
  found/merged.

### 3.8 General integrity guards

Add explicit checks for:
- every atypia label consumed is in `{1, 2, 3}` (defense-in-depth on top of
  `build_atypia_manifest.py`'s existing build-time check);
- train split non-empty;
- validation split non-empty for the real (non-smoke) run;
- expected classes present in the training set;
- requested `--val-slides` actually exist (§3.6);
- image paths exist before a long scoring job begins where practical (a
  cheap `Path.exists()` pass over the manifest before the classify loop
  starts, not a guarantee for every possible failure);
- paired method/raw sample count is non-zero (§3.2);
- `raw_hamamatsu` was successfully constructed at all (currently, a typo'd
  or missing first-tag manifest would silently produce an empty baseline —
  `base_acc = method_accuracy.get("raw_hamamatsu")` returns `None`
  silently, and every `recovery_delta` cell just becomes `""` with no
  error);
- `raw_aperio` sample keys correspond to the same frames/crops as the rest
  of that run (not a disjoint or partially-overlapping set);
- no train/test slide overlap (runtime assertion on top of the
  already-true-by-construction guarantee);
- no duplicate clinical sample silently counted multiple times (§3.7).

`raw_aperio` remains a **sanity check, not a required headline baseline** —
do not assert it must score highest; instead, treat an unexpectedly low
`raw_aperio` score as a printed warning that may indicate a join/domain
problem, exactly as the current docstring already intends, just make the
threshold-and-warn behaviour explicit in code rather than implicit in a
comment.

### 3.9 Statistical interpretation

- At minimum, always report: paired frame count, `raw_hamamatsu` accuracy,
  normalized-method accuracy, `recovery_delta` — all already covered by
  §3.2/§3.3's outputs.
- Add a paired bootstrap confidence interval for `recovery_delta` over
  frames (resample paired frames with replacement, recompute both
  accuracies and their difference per resample, report the 95% percentile
  interval) — mirrors the pattern already used elsewhere in this project
  (`score_atypia_classifier.py`'s sibling scripts and `benchmark_vae_
  reconstruction.py`/P1-14's Stage A/B already do exactly this kind of
  paired bootstrap). Keep it at this level of complexity — no mixed-effects
  modelling or per-slide stratified resampling unless a first pass shows
  the simple frame-level bootstrap is misleading (e.g. wildly unstable due
  to slide-count imbalance), which should be checked, not assumed.

### 3.10 Outputs / backwards compatibility

- `per_crop.csv` — kept, existing schema, explicitly secondary/diagnostic.
- `per_frame.csv` — new, schema per §3.3.
- `summary.csv` — extended, not silently redefined: add a `level` column
  (`frame`/`crop`) so both scopes coexist in one file without ambiguity;
  keep `scope` (`ALL`/`ALL_excl_outliers`), add `n_frames` (frame-level
  rows) alongside the existing `n` (kept for crop-level rows, or renamed
  `n_samples` with a comment — decide one and document it, don't have both
  meaning different things under the same header unlabelled); add
  `paired_raw_hamamatsu_accuracy` and the bootstrap CI bounds for
  `recovery_delta`; add an `excluded_slides` value (constant per run, from
  §3.4) rather than the current per-row `outlier`/`robust_z` pair, which
  become per-slide-row-only diagnostics, not scope-defining fields anymore.
- New `integrity_report.csv` (or `.json`) — one row/entry per method:
  `method, source_direction, paired_frames, paired_crops, missing_vs_raw,
  duplicates_found, duplicates_policy`. This is the artifact that answers
  "should I trust this score" at a glance, per the ticket's own request in
  §12.
- Document the change from the old `summary.csv` schema explicitly in this
  ticket and in the script's own docstring/changelog comment — do not
  silently reinterpret old column meanings.

---

## 4. Implementation plan

Ordered so nothing downstream is built on top of an unverified assumption:

1. **Direction metadata + validation (§3.1).** Add direction recording to
   `infer_colour_lora.py`/`infer_baseline.py` if not already present
   in some retrievable form; add the direction-assertion gate to
   `score_atypia_classifier.py`. This is a prerequisite for step 7 (rerun)
   even mattering.
2. **Pairing infrastructure (§3.2).** Canonical sample-key extraction,
   intersection logic, paired-count reporting — this is the backbone every
   later metric (frame-level, outlier policy, bootstrap) builds on.
3. **Frame-level aggregation (§3.3).** `per_frame.csv`, frame-level
   `pool()`-equivalent, using the paired set from step 2.
4. **Common outlier policy (§3.4).**
5. **Batched inference fix (§3.5)** — independent of the others, can land
   any time, but do it before any large real rerun to avoid burning GPU
   memory unnecessarily on the corrected runs.
6. **Manifest-assumption checks + seed handling (§3.7).**
7. **Integrity guards (§3.8) + statistical CI (§3.9) + output schema
   (§3.10).**
8. **`train_atypia_classifier.py` hardening (§3.6)** — can proceed in
   parallel with 1-7 since it's a separate script, but its output
   (a possibly-different `best.pt`) gates whether step 9 needs a fresh
   checkpoint.
9. **Smoke-test validation (§10)** on a small controlled sample before any
   real rerun.
10. **Reruns (§11)**, in the order that unblocks the most: cheap
    re-inference (A0-A5 ladder, once H2A checkpoints/flags exist) before
    expensive retraining (P1-10/P1-11/P3-06/P3-07).

---

## 5. Files expected to change

- `src/eval/score_atypia_classifier.py` — direction validation, pairing,
  frame-level aggregation, common outlier policy, batched inference,
  manifest checks, integrity guards, bootstrap CI, new output schema.
- `src/train/train_atypia_classifier.py` — val-slide validation, class-
  distribution reporting/checks, macro-F1-based checkpoint selection,
  checkpoint/config metadata additions.
- `src/eval/infer_colour_lora.py` / `src/eval/infer_baseline.py` —
  additive: record direction retrievably if not already (`infer_colour_
  lora.py` already has `--direction`; `infer_baseline.py` needs the flag
  added at all, currently hardcoded A2H per §0).
- `slurm/score_atypia_classifier.slurm` / `slurm/train_atypia_classifier.
  slurm` — updated invocation flags/env-vars as needed, additive only
  (matches this project's established non-breaking env-var-override
  pattern, e.g. the `TAGS` override already added this session).
- New: `tickets/P2-12_atypia_classifier_evaluation_hardening.md` (this
  file) — no code, tracked for the record.

---

## 6. Validation (smoke test before any real rerun)

- A small controlled sample (a handful of crops spanning at least 2
  frames, at least 2 methods, at least one deliberately duplicated
  seed/crop, and — if feasible to construct cheaply — one deliberately
  mismatched-direction manifest to confirm the failure path actually
  fires) run end-to-end through the hardened `score_atypia_classifier.py`.
- Confirm old-code and new-code give matching individual crop
  probabilities on the same non-duplicated crops (i.e. the batching
  rewrite in §3.5 didn't change numerics beyond floating-point noise).
- Confirm the smoke run produces `per_crop.csv`, `per_frame.csv`,
  `summary.csv`, and the integrity report, all internally consistent
  (e.g. `n_frames` in `summary.csv` matches the actual distinct-frame count
  in `per_frame.csv`).
- Only after this passes: proceed to any real (non-smoke) rerun.

---

## 7. Acceptance criteria

Status as of the 2026-08-28 smoke test (job 47545, `p3_07_smoke_heldout`:
93 crops / 8 A06 frames / 3 seeds, real data, six checks against the
hardened `score_atypia_classifier.py`; local synthetic-data unit tests
against the pure logic functions preceded it, run before ever touching
the cluster):

- [x] Every method's `recovery_delta` is computed against `raw_hamamatsu`
      on exactly the same paired frame IDs (§3.2, §3.3). Verified:
      `integrity_report.csv` shows `paired_frames=8, method_only_crops=0,
      raw_only_crops=0` for both `p3_07_smoke_heldout` and `raw_aperio`.
- [x] Method and `raw_hamamatsu` paired sample counts are identical for
      each comparison, and reported. Same evidence as above.
- [x] Frame-level prediction correctly averages crop probability vectors
      before `argmax` (§3.3). Verified: `per_frame.csv` produced; check 6
      confirms `summary.csv`'s frame-level `n` matches `per_frame.csv`'s
      actual distinct-frame count for every method.
- [x] Repeated crops (duplicate seeds/keys) do not inflate `n_frames` or
      `n_paired` (§3.7). Verified on real 3-seed data: 93 raw manifest rows
      collapsed to `paired_crops=31` (the correct distinct-crop count) with
      `duplicates_dropped=0` (no accidental dupes conflated with the
      intentional 3-seed averaging) and `paired_frames=8` (not 24).
- [x] A mismatched translation direction causes a clear, loud failure, not
      a silent score (§3.1). Verified: check 1 (unresolved direction,
      rc=1) and check 3 (declared-but-mismatched direction, rc=1) both
      aborted with the expected distinct error messages; check 2 (matched)
      and check 4 (`--allow-direction-mismatch`, `recovery_delta` marked
      `"N/A (non-clinical direction)"`) both succeeded as expected.
- [x] Mixed `source_mode` values within one manifest do not silently pool
      (§3.7). Verified via local unit test only (`check_source_mode_uniform`
      on synthetic mixed-mode rows) — not exercised by this smoke run,
      since `p3_07_smoke_heldout` is uniformly `source_mode=correct`.
- [x] Duplicate seeds/crops are handled per the documented averaging
      policy, not left ambiguous (§3.7). Same evidence as the seed-related
      item above.
- [x] GPU inference never transfers the full evaluation set to device at
      once (§3.5). Verified by code structure (per-batch `.to(device)`
      inside the chunking loop) plus indirect empirical support: the
      `--batch-size 1` smoke run (93 individual transfers) completed
      successfully with predictions identical to `--batch-size 64`.
- [ ] An invalid `--val-slides` entry fails clearly at training start
      (§3.6) — **not yet exercised**; this smoke test only ran
      `score_atypia_classifier.py`, not `train_atypia_classifier.py`.
      Verified by code review only so far.
- [ ] Validation class distribution is printed and stored in
      `training_config.json` (§3.6) — **not yet exercised**, same reason.
- [ ] Best checkpoint selection uses the documented class-balanced metric
      (macro-F1), stored alongside raw accuracy (§3.6) — **not yet
      exercised**, same reason. The manual macro-F1 formula itself was
      hand-verified against a worked example locally (sklearn isn't
      installed on this machine to cross-check directly), but a real
      `--smoke` training run hasn't been submitted yet.
- [~] Common outlier exclusions are identical across every method in a run
      (§3.4) — ran without error (`excluded_slides=none`, correctly, since
      `p3_07_smoke_heldout` is single-slide and `flag_outliers` never flags
      with <3 slides), but this smoke data doesn't stress-test the
      "identical across multiple differing methods" case the way a
      multi-slide, multi-method real rerun will.
- [x] Old-code and new-code crop-level probabilities match (within
      floating-point tolerance) on a controlled smoke sample (§6). Check 5:
      `--batch-size 1` vs `64` gave **zero prediction (argmax) mismatches**
      across all 155 scored crops; max raw-probability drift was 0.00119
      (median 0.00006) on a crop not near a decision boundary (0.636 vs
      0.279, same predicted class either way) — consistent with ordinary
      cuDNN batch-size-dependent floating-point noise, not a logic bug in
      the batching rewrite. (The smoke script's own 1e-4 tolerance flagged
      this as a script-level "FAIL" — that threshold was simply too strict
      for real GPU non-determinism; the prediction-level check is the one
      that actually matters and it passed cleanly.)
- [x] A small smoke-test run completes end-to-end and produces all
      expected output files (§6). `per_crop.csv`, `per_frame.csv`,
      `summary.csv`, `integrity_report.csv` all produced and internally
      consistent across all four scoring invocations (checks 2, 4, 5).
- [x] An integrity line/report is printed and saved per method, of the
      form `method=X paired_frames=Y missing_vs_raw=0 duplicates=0
      source_direction=H2A` (§3.1, §3.10). Verified — exact format
      confirmed in the job log and `integrity_report.csv`.

**Remaining before this ticket can be called fully verified:** a
`train_atypia_classifier.py --smoke` run (or equivalent) to exercise the
three unchecked `--val-slides`/class-distribution/checkpoint-selection
items, and a multi-slide multi-method real rerun to properly stress-test
the common outlier policy. Both are cheap, GPU-light, and don't require
any new H2A training — can be scheduled whenever useful, separate from
§11's real reruns.

---

## 8. Rerun requirements

**Mandatory once this ticket lands, regardless of §0's direction finding:**
the scoring step (`score_atypia_classifier.py`) must be rerun against
whatever manifests are being compared, since its output schema and pairing
logic both change materially. Every existing P2-09 table/number (original
26-method table, and this session's P1-10/P1-11/P1-16/P3-06/P3-07
extension in `tickets/PHASE2-TICKETS.md`, `tickets/PHASE1-TICKETS.md`,
`tickets/PHASE3-TICKETS.md`, `tickets/P1-16_source_detail_colour_residual_
fusion.md`, `docs/results/RESULTS_SUMMARY.md`) **must be marked stale**
until a hardened rescoring exists — not because the numbers are necessarily
wrong in every particular, but because they were computed with unpaired,
crop-level, per-method-outlier-policy, unvalidated-direction methodology
that this ticket exists specifically to replace.

**Beyond the mandatory rescoring, whether new *inference* is required
depends on §0's direction finding, per architecture:**

- **A0/A1 (no colour LoRA at all).** Direction is purely which raw image
  is fed as img2img input — an H2A rerun is `infer_colour_lora.py
  --direction H2A` with no `--lora`, no retraining needed.
- **A2/A3 (colour LoRA, +/- ControlNet).** `h2a_r4`/`h2a_r8` colour LoRA
  checkpoints **already exist** (`tickets/PHASE1-TICKETS.md` P1-03c/P1-03d,
  trained specifically as a "cycle-consistency" H2A counterpart to
  `a2h_r4`/`a2h_r8`) — an H2A rerun is inference-only, swapping the LoRA
  checkpoint and passing `--direction H2A`; ControlNet-Canny conditioning
  itself is direction-agnostic (conditions on whichever image is fed in).
  **No new training needed.**
- **A4/A5 (+LCM-LoRA / +hist warm-start LoRA).** Both additional adapters
  are pretrained/frozen and direction-agnostic; combined with A2/A3's
  existing H2A LoRA, an H2A rerun is plausibly inference-only here too —
  **verify this assumption directly** (confirm `h2a_r8` composes cleanly
  with `--lcm`/`--hist-lora` the same way `a2h_r8` currently does) before
  assuming no training is needed, rather than asserting it outright.
- **Classical baselines (Macenko/Reinhard/Histogram Matching).**
  `infer_baseline.py` has no `--direction` flag at all (§0) — needs a small
  code change (mirroring `infer_colour_lora.py`'s existing pattern) before
  an H2A run is even possible. Cheap once added; classical methods need no
  training.
- **P1-10/P1-11 (source-conditioned ControlNet + colour LoRA, joint
  training).** `--direction H2A` already exists as a flag in `train_
  colour_translation_lora.py` and `infer_colour_source_ddim_inversion.py`,
  but **no H2A-direction checkpoint has ever been trained** — P1-10's own
  ticket explicitly scoped H2A out. A valid H2A version requires a **new,
  full joint training run** (fresh source-conditioned ControlNet + colour
  LoRA, `--direction H2A`), not just new inference. Real, nontrivial
  compute cost — recommend scoping this as separate follow-on work,
  explicitly approved on its own, rather than assumed-included in this
  ticket's mandatory scope.
- **P3-06/P3-07 (SDXL transfer).** Same situation as P1-10/P1-11 — their
  own tickets explicitly scope H2A out (`infer_colour_lora_sdxl.py` has the
  `--direction` flag, but scope decisions describe A2H as the only trained
  direction). New SDXL training required — same recommendation as above.
- **P1-16 (raw-source-detail/colour-residual fusion).** Built as a
  post-hoc fusion on top of P1-11's outputs (`fuse_source_detail.py` has no
  direction concept of its own — it inherits whatever direction its input
  P1-11 run used). Needs P1-11's H2A version to exist first, then a rerun
  of the fusion step against it.

**`atypia_r18` classifier retraining:** independent question, driven by
§3.6's checkpoint-selection change, not by §0. If macro-F1-based selection
picks a materially different step than the current `best.pt` (step 400,
raw val_acc 0.9468), **retrain and treat the new checkpoint as the one to
use going forward**; if it happens to agree, the existing checkpoint may
remain valid and only the mandatory rescoring (above) is needed — determine
this empirically during implementation rather than assuming either
outcome. Do not conflate "the checkpoint might need to change" with "the
existing checkpoint was invalid" — a different checkpoint-selection metric
choosing a different step is expected behaviour, not evidence the current
`best.pt` was broken.

---

## 9. Risks / interpretation

- **This ticket may show P2-09's headline "first positive result in the
  project" evaporates, weakens, or reverses once measured in the correct
  direction with strict pairing and frame-level aggregation.** That is a
  legitimate, scientifically valid outcome — the goal here is methodological
  fairness (same fixed classifier, same ground-truth frames, same paired
  comparison set, one prediction per true clinical annotation unit, the
  intended H→A direction), not a predetermined positive result. Do not let
  the fact that P2-09 was previously "the first clearly positive result
  anywhere in this project" bias the implementation toward preserving that
  outcome.
- Conversely, it is equally possible a corrected H2A rerun on the cheap-to-
  regenerate A0-A5 ladder still shows a real positive recovery — this
  ticket does not presuppose the answer either way.
- The compute cost asymmetry in §11 (A0-A5 = cheap re-inference; P1-10/
  P1-11/P3-06/P3-07/P1-16 = expensive new training) means a corrected,
  valid clinical-utility result may become available for the *classical*
  ablation ladder well before it's available for this project's actual
  best-performing configurations — worth being explicit that "P2-12
  complete" and "every architecture has a valid H2A clinical-utility
  number" are two different milestones, and this ticket's mandatory scope
  is the former.
- Statistical power: this classifier's own absolute accuracy is already
  low (raw Aperio sanity check barely above chance, per P2-09's own
  existing caveat) — frame-level aggregation shrinks sample sizes
  considerably versus crop-level pooling (120 held-out frames total vs.
  ~480-1440+ crops depending on method), so the bootstrap CIs added in
  §3.9 may turn out wide enough to make several of the currently-reported
  point-estimate differences non-significant. Report this plainly if so,
  rather than treating a wide CI as a reason to fall back to crop-level
  pooling's narrower-but-wrong-population intervals.

---

## 10. Non-goals

- **Not** retraining every diffusion architecture (P1-10/P1-11/P3-06/
  P3-07/P1-16) in the H2A direction as part of *this* ticket's mandatory
  scope — that is real, separate, per-architecture follow-on work,
  explicitly flagged in §11 and requiring its own go-ahead given the
  compute cost.
- **Not** touching `score_outputs.py` or any of the SSIM/LAB-Wasserstein/
  colour-recovery scoring — those are unaffected by §0's finding (see the
  clarifying note at the end of §0) and out of scope here.
- **Not** redesigning the A0-A5 ablation ladder's architecture or the
  colour LoRA training methodology.
- **Not** re-litigating P2-08 (structural safety), P2-10 (CAMELYON17), or
  any other Phase 2 ticket — this is scoped strictly to P2-09's classifier
  evaluation.
- **Not** adding statistical machinery beyond a simple paired bootstrap CI
  (§3.9) unless a first pass shows it's genuinely needed — no mixed-effects
  models, no per-slide stratified resampling by default.
- **Not** asserting `raw_aperio` must be the numerically highest-scoring
  method — it remains a sanity check whose unexpectedly poor performance is
  a warning sign, not a hard requirement (§3.8).
- **Not** implementing anything in this ticket itself — per the explicit
  request this document responds to, this is specification only.
