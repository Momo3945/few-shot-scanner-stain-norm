# How to read this folder

This folder holds the measured results for a project that tests whether AI
image generation (a "diffusion model") can make microscope slide photos from
one scanner look like they came from a different scanner — a common problem
in digital pathology, since different scanner brands render the same tissue
sample with slightly different colours. The idea being tested: could an AI
model learn to translate between scanners, instead of (or better than) the
existing simpler, non-AI colour-correction methods?

**Start with `RESULTS_SUMMARY.md`** in this same folder — it's the full
written report, in order, with the actual findings explained. Everything
else in this folder is the raw data and images that report is built from.
This README is just a map and a glossary.

## Where things are

- **`RESULTS_SUMMARY.md`** — the main report. Read this first.
- **`qualitative/`** — actual before/after images, so you can look at the
  results yourself rather than just trust a number. Good place to start if
  you'd rather see pictures than read numbers.
- **`phase1_ablation/`** — the core experiment's raw numbers. "Phase 1" tested
  six increasingly complex versions of the AI pipeline (named A0 through A5,
  see glossary below) against a "do nothing" comparison.
- **`classical_baselines/`** — raw numbers for the older, non-AI methods
  (Macenko, Reinhard, Histogram Matching) that the AI pipeline is compared
  against. These are simple, deterministic colour-recipe adjustments, not AI.
- **`analyze.py`** — the script used to double-check trends in the numbers.

Every result folder (`a0/`, `a3_ext_s02/`, `macenko/`, etc.) has up to three
files:
- `eval_summary.csv` — the headline numbers, averaged. **This is the file
  quoted throughout `RESULTS_SUMMARY.md`.**
- `eval_per_crop.csv` — the same numbers, but for every individual image
  crop rather than averaged. Useful for double-checking an average isn't
  hiding something.
- `eval_manifest.csv` — bookkeeping: which output file matches which input.

## Glossary — the A0–A5 names

The AI pipeline was built up one piece at a time, so each version's result
shows what that specific piece contributed:

| Name | What it is |
|---|---|
| **A0** | The frozen, untrained AI model on its own — the "does the base model already do this" control. |
| **A1** | A0 + a component that keeps the tissue's structure/edges intact during generation. |
| **A2** | A0 + a small trained add-on ("LoRA") that teaches the model this specific scanner's colours. `a2h_r4`/`a2h_r8` are two versions of this trained with different capacity. |
| **A3** | A1 + A2 combined — structure-preservation and colour-learning together. |
| **A4** | A3 + a speed optimisation (fewer generation steps, ~6x faster). |
| **A5** | A4 + an extra add-on pretrained on histopathology images in general (not just this scanner pair), to see if general tissue knowledge helps. |

Folders with an `_ext_sNN` suffix (e.g. `a4_ext_s02`) are the same
configuration re-run at a different "strength" setting — `s02` = strength
0.20, `s07` = strength 0.70, `s67` = a small 0.6/0.7 test. **Strength**
controls how much the AI is allowed to change the image: low strength keeps
the original photo mostly intact (safer, less colour change); high strength
lets it regenerate more of the image (bigger colour change, more risk of
altering the actual tissue structure).

## Glossary — the CSV columns

| Column | Plain meaning |
|---|---|
| `ssim` | How well the tissue's *structure* (nuclei, shapes, edges) was preserved, from 0 (unrecognisable) to 1 (pixel-perfect). This project's central concern — a scanner-colour fix that also blurs or invents tissue structure would be dangerous for real diagnostic use. |
| `psnr`, `mae` | Two more ways of measuring the same thing as `ssim` (structure/pixel accuracy) — included for completeness, tell a similar story. |
| `lab_total` | How different the *colour palette* is from the real target-scanner photo — lower is a closer colour match. This is a "distribution" measure (does the overall colour mix match), not a per-pixel one. |
| `wlab_mean` | The same colour-distance idea as `lab_total`, but checked in small local tiles instead of the whole image at once — catches a method "cheating" by matching the overall colour mix while still being locally wrong. |
| `recovery_delta_lab` | The actual headline number for colour success: *(colour error doing nothing) − (colour error after this method)*. Positive = the method made the colour more accurate than not touching the image at all; negative = it made things worse. |
| `de2000_mean` | A perceptual colour-difference measure — turned out to track `ssim` closely rather than adding new information (see `RESULTS_SUMMARY.md`'s note on this). |
| `robust_z`, `outlier` | Statistical outlier-detection columns — used to flag one test slide (**A06**) that turned out to have a much bigger starting colour gap than the other four, so it's always reported separately rather than averaged in and skewing the picture. |

## The short version of what was found

The AI pipeline reliably shifts colour in the right direction (better than
doing nothing), and one configuration (source-conditioned training, see
`RESULTS_SUMMARY.md`'s P1-10 section) achieved the best colour-recovery
result of any AI configuration tested. But on *structural* accuracy
(`ssim`), the simple non-AI methods (`classical_baselines/`) consistently
beat every AI configuration tested — because they mathematically never move
a pixel, while the AI regenerates the image and can't help but drift
slightly from a pixel-perfect match. `RESULTS_SUMMARY.md` explains this
finding in full, including why it happens and what was done to rule out
measurement artefacts.
