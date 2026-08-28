# How to read this folder

This folder holds the measured results for a project that tests whether AI
image generation (a "diffusion model") can make microscope slide photos from
one scanner look like they came from a different scanner — a common problem
in digital pathology, since different scanner brands render the same tissue
sample with slightly different colours. The idea being tested: could an AI
model learn to translate between scanners, instead of (or better than) the
existing simpler, non-AI colour-correction methods?

**Start with `RESULTS_SUMMARY.md`** in this same folder — it's the full
written report, in order, with the actual findings explained. This README
is a map: one row per experiment, pointing at where its numbers and images
actually live.

## Index

Every ticket folder has the same shape: a `README.md` (mechanism + headline
number + pointers), an `eval/` with the real pulled CSVs where a new pull
was needed, and `images/` where a diagnostic image exists. Folders marked
"data lives elsewhere" don't duplicate CSVs already sitting in
`phase1_ablation/` or `classical_baselines/` — their README says exactly
where to look instead.

| Folder | What it is | Status |
|---|---|---|
| `phase1_ablation/` | The A0–A5 ablation ladder + strength-sweep follow-ups + raw baseline | ✅ done |
| `classical_baselines/` | Macenko, Reinhard, Histogram Matching | ✅ done |
| `p1_10_source_conditioning/` | Training-time source conditioning (fresh ControlNet branch) | ✅ done — mixed |
| `p1_11_ddim_inversion/` | Deterministic DDIM-inversion initialisation | ✅ done — best SD1.5 result at the time |
| `p1_12_generic_lcm/` | Few-step LCM acceleration on P1-10 | ✅ closed — speed/quality tradeoff |
| `p1_13_lcm_distillation/` | Task-specific LCM-LoRA distilled from P1-10 (P1-13 + P1-13b) | ❌ closed negative |
| `p1_14_vae_decoder_swap/` | Drop-in VAE decoder swap, isolated reconstruction test | ✅ done  |
| `p1_15_alt_decoder/` | P1-11 decoded with P1-14's winning VAE | ⚠️ smoke data exists, not written up |
| `p1_16_source_fusion/` | Post-hoc raw-source + colour-residual fusion | ✅ **first config to beat classical on SSIM** |
| `p1_17_differential_diffusion/` | Per-pixel spatially-varying denoising strength | ❌ closed negative |
| `p2_06_ground_truth_comparison/` | SSIM/PSNR/MAE vs. real target, all methods (data lives elsewhere) | ✅ done |
| `p2_07_cycle_consistency/` | Round-trip A→H→A reconstruction | ❌ done |
| `p2_08_hovernet_relative_dice/` | Nucleus-detection agreement vs. HoVer-Net | ❌ done — fails 0.95 threshold |
| `p2_09_atypia_classifier/` | Downstream ResNet18 atypia classifier | ✅ **diffusion beats every classical baseline** |
| `p2_10_camelyon17/` | Cross-hospital generalisation, 5 unseen centres | ❌ done |
| `p2_11_baseline_confound/` | Windowed-metric confound check (data lives elsewhere) | ✅ done |
| `p3_04_sdxl_a4/` | SDXL port of the original A4 ladder config | ❌ done |
| `p3_05_sdxl_a5_warmstart/` | SDXL port of the A5 histopathology warm-start | ❌ done |
| `p3_06_sdxl_source_cond/` | SDXL transfer of P1-10's fresh-ControlNet architecture | ❌ done |
| `p3_07_sdxl_native1024/` | P3-06 retrained at native 1024×1024 | 🔄 reopened, D1 diagnostic now resolved |
| `analyze.py` | Reproducible per-crop trend analysis behind the P1-09 strength-sweep picks | — |

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
doing nothing). For most of the project's history, the simple non-AI
methods (`classical_baselines/`) consistently beat every AI configuration
on structural accuracy (`ssim`) — because they mathematically never move a
pixel, while the AI regenerates the image and can't help but drift slightly
from a pixel-perfect match. **That changed with `p1_16_source_fusion/`**:
using the AI only to estimate a colour shift, then compositing it onto the
untouched raw source, clears every classical baseline on SSIM at full
held-out scale — the same trick classical methods use, borrowed
deliberately. Separately, `p2_09_atypia_classifier/` is the one place
diffusion beat classical *without* that trick, on a real downstream
clinical task. `RESULTS_SUMMARY.md` explains all of this in full, including
why it happens and what was done to rule out measurement artefacts.
