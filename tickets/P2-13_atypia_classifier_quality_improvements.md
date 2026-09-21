# P2-13 — Atypia-Classifier Quality Improvements (`atypia_r18_v2`)

**Status:** ✅ CLOSED, negative result (2026-09-20). Trained `atypia_r18_v2`
(jobs 56514 smoke, 56528 real training) and rescored the full H2A eval set
against it (job 57154, `eval/atypia_classifier_scores_h2a_v2/`). **Verdict:
`atypia_r18_v2` is not an improvement — it is worse than the frozen
`atypia_r18` on both sanity checks, and its raw_hamamatsu baseline is bad
enough to make its own recovery_delta numbers uninterpretable. The frozen
`atypia_r18/best.pt` and P2-12 §8.5's H2A verdict are unchanged and remain
the citable result.** See §Result below for the full comparison and the
leading hypothesis for why.

**Source:** `docs/proposal_draft(6).tex` §"Clinical Utility" — same citation
basis as P2-09/P2-12. Motivated by `tickets/P2-12_atypia_classifier_evaluation_hardening.md`
§8.5/§9: the corrected H2A `recovery_delta` table showed no genuine
clinical-utility benefit for any method, and §9 explicitly flags that the
classifier's own raw-Aperio sanity accuracy is "barely above chance" —
i.e. the classifier's low absolute accuracy limits statistical power to
detect any real effect, confounding "no benefit" with "underpowered
instrument."

## Scope

**Does not touch or replace the frozen `atypia_r18/best.pt` checkpoint** —
that remains the instrument P2-12 §8.5's already-reported H2A
`recovery_delta` table was scored against, and stays the citable result.
This ticket trains a new sibling checkpoint, `atypia_r18_v2`, to test
whether a few concrete, low-risk quality levers materially raise the
classifier's own accuracy/macro-F1:

1. **Train-only augmentation** (`--augment`): random horizontal/vertical
   flip, random 0/90/180/270° rotation, mild colour jitter. Tissue has no
   canonical orientation, so flips/rotations are safe; the current script
   had no augmentation at all.
2. **Class-balanced sampling** (`--balanced-sampling`): `WeightedRandomSampler`
   (inverse class frequency) in the train loader, in addition to the
   existing inverse-frequency loss weighting — the dataset is heavily
   imbalanced (roughly 1:23, 2:222, 3:52 frames).
3. **More crops per frame**: `--max-crops-per-frame 0` (all tissue crops,
   vs. the default cap of 4) — no code change, just a different flag value.
4. **A validation split that contains all 3 classes**: the current
   auto-picked "last 2 alphabetically" split is missing class 2 entirely
   (confirmed in P2-12 §7's acceptance-criteria evidence) — pick
   `--val-slides` explicitly from a real per-slide class-count check.

## Non-goals

- No changes to `score_atypia_classifier.py` — it already takes
  `--checkpoint` as a path, so pointing it at `atypia_r18_v2/best.pt` needs
  no code change.
- Not re-litigating P2-12's direction/pairing/frame-level-aggregation/
  outlier-policy methodology — this ticket is purely about classifier
  quality, not scoring methodology.
- Not a guaranteed win — if `atypia_r18_v2` doesn't clear
  `atypia_r18`'s val macro-F1 by a meaningful margin, the frozen checkpoint
  and P2-12 §8.5's verdict stand as-is; this ticket's result gets recorded
  either way (§ below), not just on success.

## Implementation (done)

- `src/train/train_atypia_classifier.py`: added `--augment` (train-only
  flip/rotation/`ColorJitter`, `CropDataset` gained an `is_train` flag so
  validation crops are never augmented) and `--balanced-sampling`
  (`WeightedRandomSampler` over inverse class frequency, mutually exclusive
  with `shuffle=True`). Both default off; `vars(args)` already flows both
  into `training_config.json` with no further change needed.
- `slurm/train_atypia_classifier.slurm`: additive `AUGMENT`,
  `BALANCED_SAMPLING`, `MAX_CROPS_PER_FRAME`, `VAL_SLIDES` env-var
  overrides, all unset by default (existing invocation unchanged).

## Steps taken

1. Read-only cluster check of per-slide class counts in
   `pairs/atypia_train_manifest.csv` (2026-09-18) found the current
   auto-picked val split (A17, A18) has **zero score-3 frames**, confirming
   the gap flagged in P2-12 §7. Slide A15 alone covers all 3 classes
   ({1:1, 2:21, 3:2}); chose `--val-slides A15,A17,A18`, keeping A12 (19 of
   the dataset's 23 total class-1 frames) in training.
2. Smoke test: job 56368 hit a new bad GPU node (`mscluster45`, "Unable to
   determine the device handle for GPU0" — same sub-signature CLAUDE.md
   already tracked for 5 other nodes, now a 6th; added to the exclude list).
   Resubmitted (job 56469) hit a path bug — passed a relative manifest path
   that doesn't resolve from the script's `src/train` cwd. Fixed, resubmitted
   (job 56514): passed cleanly, `--augment`/`--balanced-sampling` ran without
   error.
3. Real training run (job 56528): 2000 steps, `MAX_CROPS_PER_FRAME=0`
   (2711 train / 828 val tissue crops, vs. the baseline's 1000/188). Best
   checkpoint at step 1200: `val_macro_f1=0.419, val_acc=0.882` on the new,
   harder A15/A17/A18 split — not directly comparable to the old
   checkpoint's `val_acc=0.9468` (step 400), since that used the easier,
   class-3-free A17/A18 split and predates P2-12's macro-F1 selection code
   entirely (its `training_config.json` has no `selection_metric` field at
   all — it was never retrained under the hardened pipeline).
4. Rescoring (job 57154) needed two fixes over the original plan: node
   `mscluster57` hit the same intermittent GPU fault (7th hit of this
   sub-signature, added to the exclude list; `score_atypia_classifier.py`
   has no fail-fast GPU check, so it silently fell back to CPU rather than
   aborting — a known, previously-flagged gap, not touched here per
   Non-Goals), and the `a0` source tag (used via `--raw-hamamatsu-tag a0`)
   predates P2-12's `run_metadata.json` direction stamping, so it needed an
   explicit `--tag-direction a0=A2H`. Ran clean after both fixes.

## Result (2026-09-20)

Held-out test-set (120 frames, frame-level) sanity-check comparison,
`atypia_r18` (frozen, job 47727's original scoring) vs. `atypia_r18_v2`
(job 57154, this ticket):

| Metric | `atypia_r18` (frozen) | `atypia_r18_v2` (new) |
|---|---|---|
| raw_aperio accuracy | 0.533 | 0.508 |
| raw_aperio macro-F1 | 0.326 | 0.270 |
| raw_hamamatsu accuracy | 0.442 | **0.192** |
| raw_hamamatsu macro-F1 | 0.235 | **0.132** |

Held-out test set's own majority-class baseline (always predict score 2,
60/120 frames) is **0.50 accuracy** — meaning `atypia_r18`'s raw_aperio
score (0.533) is genuinely only barely above the naive baseline (matches
P2-12 §9's "barely above chance" characterization), and `atypia_r18_v2`'s
raw_hamamatsu score (0.192) is **worse than random 3-class guessing**
(0.333), let alone the majority baseline.

**`atypia_r18_v2` is strictly worse on every held-out sanity metric — not
a marginal or mixed result.** Its `recovery_delta` table (`eval/
atypia_classifier_scores_h2a_v2/summary.csv`) shows large positive deltas
for several methods (up to +0.342 for `h2a_a5_s0.50`), but this is an
artefact, not a finding: `recovery_delta` is defined relative to
`raw_hamamatsu`'s own accuracy, and a baseline this broken (worse than
chance) makes almost anything look like an improvement over it. **This
table must not be read as evidence of clinical utility and does not
supersede P2-12 §8.5.**

**Leading hypothesis for the regression (not further investigated —
out of scope beyond documenting it):** this is a Hamamatsu-domain-specific
failure (same-domain raw_aperio accuracy only dropped slightly, 0.533→0.508)
that lines up with the two levers added together — mild `ColorJitter`
augmentation on a classifier whose entire downstream job is reading
colour-shifted (scanner-normalized) images, and doubled inverse-frequency
correction (balanced sampling *and* loss weighting together) on an already
tiny dataset (13 slides). Either could plausibly push the decision boundary
away from real Hamamatsu-domain colour statistics specifically. A follow-up
ablating `--augment` and `--balanced-sampling` independently (2 more runs)
would isolate which lever is responsible, but this ticket stops here per
its own scope — it set out to test whether these two levers helped, found
they didn't, and recorded why plausibly not.

**Conclusion:** no change to the project's citable clinical-utility result.
`atypia_r18/best.pt` remains the classifier used for P2-12 §8.5's H2A
`recovery_delta` table. `atypia_r18_v2` and its scoring output are kept on
disk as a documented negative result, not deleted, not used downstream.
