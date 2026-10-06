# Learned baselines — StainNet, ParamNet, StainGAN (P2-11b)

**What this tests:** the three learned (non-diffusion) methods listed in the
proposal's `tab:baselines`, trained on **exactly the same 50 Aperio/Hamamatsu
coordinate-corresponding pairs the colour LoRA trains on**
(`pairs/train`, all from slide A03), then run through the identical
registration / crop / scoring pipeline as every other method in this folder
(496 held-out crops, `score_outputs.py` unchanged). Both directions: A2H and H2A.

**Status:** ✅ DONE for colour + pixel-structure scoring and the Lizard
Relative Dice structure-safety arm for all seven normalisers (2026-10-05/06). Atypia-classifier and
CAMELYON17 arms are not run (the classifier is not a valid instrument anyway,
P2-14 gate failed).

| Method | Architecture | Training (this project) |
|---|---|---|
| **StainNet** | 3 × 1×1 conv, 32 ch (1,283 params) — a per-pixel colour map | paired L1, SGD 0.01, cosine, bs 10, random 256 crops; 1,500 steps |
| **ParamNet** | ResNet18 predicts the weights of a per-image 1×1-conv colour map | CycleGAN-style (upstream recipe), 20,000 steps, bs 1 |
| **StainGAN** | standard CycleGAN (ResNet-9 generator, PatchGAN) | unpaired, LSGAN + cycle 10 + identity 5, 20,000 steps, bs 1 |

Model definitions adapted from the authors' public code (pinned commits in
`src/baselines/models.py`). **StainGAN has no released weights** (its repo
lists them as a TODO), so it is trained here, like the other two.
**Upstream MITOS-trained checkpoints were deliberately not used**: their
train/test split is unknown and may include held-out slides A06/A08/A09/A13/A16.

## Headline (crop-weighted over the four typical slides A08/A09/A13/A16; A06 reported separately)

Full table: `aggregate_typical_slides.csv` (regenerate with
`python learned_baselines/aggregate_typical_slides.py` from `docs/results/`).

**A2H**

| Method | Lab W. (typical) | SSIM (typical) | Δlab (typical) | A06 Lab W. | A06 Δlab |
|---|---|---|---|---|---|
| Raw (no model) | 25.41 | 0.748 | 0 | 94.84 | 0 |
| Macenko | 21.02 | 0.649 | +4.39 | 50.30 | +44.5 |
| Reinhard | 26.79 | 0.701 | −1.38 | 36.34 | +58.5 |
| Histogram Matching | 31.33 | 0.664 | −5.92 | 25.72 | +69.1 |
| StainNet | 58.02 | 0.518 | −32.61 | 38.58 | +56.3 |
| **ParamNet** | **8.55** | 0.769 | **+16.86** | 76.15 | +18.7 |
| **StainGAN** | 13.55 | **0.772** | +11.86 | 29.92 | +64.9 |

**H2A**

| Method | Lab W. (typical) | SSIM (typical) | Δlab (typical) | A06 Lab W. | A06 Δlab |
|---|---|---|---|---|---|
| StainNet | 53.80 | 0.398 | −28.38 | 80.19 | +14.7 |
| **ParamNet** | **8.14** | 0.750 | **+17.28** | 71.76 | +23.1 |
| StainGAN | 14.67 | 0.748 | +10.75 | 67.44 | +27.4 |

(H2A classical summaries: `../classical_baselines/h2a_*/eval_summary.csv`, pulled
from the P2-12 runs so every row above comes from a file in this repo.)

## How to read this

- **ParamNet and StainGAN beat every classical method on typical-slide colour
  and SSIM in A2H** — unlike the diffusion rungs, which lose to classical.
  ParamNet is the most consistent: Δlab +9.8 to +19.1 across the four typical
  slides in A2H (A13 is the weak one at +9.8) and +16.2 to +19.7 in H2A.
  StainGAN is better on the A06 outlier in A2H (+64.9 vs ParamNet's +18.7) but
  has more slide-to-slide variance (typical-slide Δlab +9.4 to +16.0 in A2H,
  +6.3 to +15.6 in H2A).
- **StainNet is worse than raw on every typical slide, both directions.** It
  only "wins" on A06. Its training loss converged (a 150,000-step retrain
  plateaued at the same L1, ~0.24 vs 0.30 for the do-nothing mapping), so this
  is not undertraining: a global per-pixel colour map fit on one slide's 50
  crops does not generalise to the other slides' stain variation.
- **Raw SSIM caveat:** the raw reference row (25.41 / 0.748) is computed over
  the full 1,296 typical-slide crops, the methods over their 432-crop subset,
  so the SSIM comparison to raw is approximate. The method-vs-method rows are
  exact (same crops).
- **Everything here is single-seed, single training set (one slide, 50
  pairs).** Lab Wasserstein is a colour-distribution metric with the
  construction confound documented under P2-11; SSIM only tests pixel-level
  agreement. Neither says the generators did not alter nuclei — that is what
  the Lizard Relative Dice run is for.

## Structure safety — Lizard Relative Dice, all seven normalisers (P2-08 protocol, A2H)

Same protocol as P2-08: normalise the 130 Lizard held-out images (reflect-pad,
disjoint 512 px tiles, output shape == source shape), run the pretrained
HoVer-Net, score Dice vs the human nucleus masks, Relative Dice =
Dice(B,G)/Dice(A,G) with Dice(A,G) = 0.69818 (original images). Pass = >= 0.95
(pre-committed in the proposal). 130/130 images, no shape-mismatch skips, for
every method. Paired bootstrap over images (10,000 resamples, seed 0):
`lizard/relative_dice_bootstrap.csv`. Change size measured over all 130 images
for every method (`lizard/change_magnitude_*.csv`): MAD = mean absolute pixel
difference vs the original (0-255), dE76 = mean per-pixel CIELAB distance.

| Method | Dice(B,G) | Relative Dice | 95% CI | Verdict | MAD (mean) | dE76 (mean) | Object-F1 | Count ratio |
|---|---|---|---|---|---|---|---|---|
| **ParamNet** | 0.6974 | **0.9989** | [0.996, 1.001] | **pass** | **1.77** | **1.65** | 0.736 | 0.678 |
| Reinhard | 0.6441 | 0.9225 | [0.905, 0.939] | fail (narrow) | 27.75 | 19.76 | 0.652 | 0.544 |
| *Diffusion (P2-08, A4 LCM @0.20)* | *0.6105* | *0.8745* | *[0.867, 0.882]* | *fail* | *15.78* | *10.75* | *0.682* | *0.627* |
| Histogram Matching | 0.5538 | 0.7931 | [0.763, 0.822] | fail | 27.63 | 20.84 | 0.540 | 0.415 |
| Macenko | 0.5142 | 0.7365 | [0.684, 0.787] | fail | 23.77 | 18.11 | 0.512 | 0.402 |
| StainGAN | 0.5056 | 0.7241 | [0.692, 0.756] | fail | 20.79 | 16.85 | 0.503 | 0.384 |
| StainNet | 0.4342 | 0.6220 | [0.567, 0.675] | fail | 27.54 | 16.18 | 0.439 | 0.355 |
| *(original images, reference)* | *0.6982* | *1.000* | | | *0* | *0* | *0.733* | *0.668* |

(Macenko passed 1 of 775 tiles through unchanged as degenerate, 0.13%; the
other two classical methods 0. `lizard/fallbacks_*.json`.)

**What this shows**

1. **ParamNet's pass is "it barely touched the images", confirmed at full scale
   (all 130, not the earlier 5-image spot check).** Mean change 1.77 grey
   levels, dE76 1.65 (about the threshold of a noticeable colour difference);
   only 2 of 130 images change by more than 4 grey levels (max 7.9). Even its
   most-changed third (MAD 2.55) keeps Relative Dice 0.9955. It leaves these
   out-of-domain images almost untouched while it recovers +17 delta-lab on the
   MITOS held-out crops, so a pass here is not evidence that it is
   structure-safe where it changes colour a lot.
2. **No method that substantially changes colour passes.** Every method that
   moves the images by 11-40 dE76 fails the 0.95 gate; the closest is Reinhard
   (0.9225, CI [0.905, 0.939], excludes 0.95). The proposal's H3/RQ2 threshold is
   not met by any normaliser that actually normalises, classical or learned.
3. **The diffusion pipeline is mid-pack, not uniquely unsafe:** 0.8745 beats
   Histogram Matching, Macenko, StainGAN and StainNet, and loses to Reinhard and
   (trivially) ParamNet.
4. **Colour-shift size does NOT by itself explain the Dice loss** (this
   corrects my first reading, which said the failures conflate structural
   damage with out-of-distribution colour shift). Reinhard changes the images the
   most (MAD 27.75, dE76 19.76) yet loses only ~8% of Dice, while the diffusion
   pipeline changes them the least among the active methods (MAD 15.78, dE76
   10.75) and loses ~12.5%; StainNet shifts about as much as Reinhard (27.54)
   and loses 38%. So the damage depends on the *kind* of change: a global
   per-tile LAB mean/std shift (Reinhard) keeps nuclei detectable, while
   resynthesis (diffusion, StainGAN) and learned nonlinear per-pixel maps
   (StainNet) do not. This supports reading diffusion's loss as real
   structural/content change (consistent with P2-06 SSIM and P2-07 round-trip),
   not a colour artefact. It is an across-method comparison of different kinds of
   change, so "supports", not "proves".
5. **Within a method, bigger change does not reliably mean bigger Dice loss**
   (Spearman of per-image change vs Dice drop: StainNet -0.36, Reinhard -0.29,
   StainGAN +0.40, Macenko +0.23, ParamNet +0.25, diffusion +0.07 (n.s.),
   Histogram Matching -0.12 (n.s.); mixed signs). Per-image change size is not a
   reliable predictor of damage; `change_magnitude_summary.csv` also has Relative
   Dice by change tercile for each method.

**Caveats:** Lizard has no scanner-pair ground truth, so this measures
structure-safety only, not whether the colour change was correct; HoVer-Net is
one instrument (PanNuke-trained) and its Dice(A,G) is only 0.70; single normalisation
seed/run per method; classical methods were applied per 512 px tile (their
fitted statistics are per tile, as in the held-out eval), and ParamNet's colour
map is predicted per tile too (possible tile seams, not inspected).

Jobs: learned normalise 64549 / 64552 / 64555 (StainNet / ParamNet / StainGAN),
HoVer-Net 64550 / 64553 / 64556, score 64551 / 64554 / 64557. Classical
normalise 64650 / 64653 / 64656 (Macenko / Reinhard / Histogram Matching),
HoVer-Net 64651 / 64654 / 64657, score 64652 / 64655 / 64658 (smokes 64647-9).
Change-size job 64659. Code: `src/eval/normalize_lizard_learned.py`,
`src/eval/normalize_lizard_classical.py`, `src/eval/lizard_change_magnitude.py`
and the matching `slurm/` files (HoVer-Net and scoring reuse
`slurm/infer_hovernet.slurm`, `slurm/score_lizard.slurm`). Per-image CSVs,
summaries and bootstrap: `lizard/`.

## Files

Per method folder (`stainnet/`, `paramnet/`, `staingan/`, `h2a_*`):
`eval_summary.csv` (per-slide numbers, same schema as every other run) and
`train_metadata.json` (steps, seed, no-held-out-selection note). Full outputs
and weights are on the cluster: `eval/<tag>` and
`baselines/<direction>_<method>/` under `/datasets/mhoosen/stain-norm/`.

Note on `stainnet/`: A2H StainNet is the 1,500-step run (cluster dir
`eval/stainnet_1500step_underfit`, name kept for provenance; the "underfit"
label turned out to be wrong, see above). A 150,000-step retrain
(`baselines/a2h_stainnet`) exists but was not run on the held-out set
because training loss had already plateaued.

**Code:** `src/baselines/models.py`, `src/train/train_learned_baseline.py`,
`src/eval/infer_learned_baseline.py`, `slurm/train_learned_baseline.slurm`,
`slurm/infer_learned_baseline.slurm`.
**Jobs (A2H):** train 64217 (StainNet), 64243 (ParamNet), 64244 (StainGAN);
infer 64242 / 64277 / 64278; score 64282 / 64343 / 64344.
**Jobs (H2A):** train 64405 / 64409 / 64412; infer 64406 / 64410 / 64413;
score 64407 / 64411 / 64414.
