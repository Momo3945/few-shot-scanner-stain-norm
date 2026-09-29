# Method

## Problem

Whole-slide images of the same tissue, scanned on different digital
pathology scanners (here: Aperio and Hamamatsu), show systematic
colour/appearance differences purely from scanner hardware and processing —
not from the tissue itself. This "scanner shift" is a known confound for
downstream computational pathology (classifiers, nucleus detection) trained
or validated across scanners. The task is **scanner-to-scanner stain
normalisation**: map an image from one scanner's colour appearance to the
other's while preserving the underlying tissue structure.

## Architecture

A frozen **Stable Diffusion 1.5** base is adapted with three lightweight,
composable pieces, added incrementally as an ablation ladder (A0–A5):

| Component | Role |
|---|---|
| **Colour LoRA** | Low-rank adapter trained on paired crops to learn the target scanner's colour/appearance distribution. The primary mechanism for the colour transformation. |
| **ControlNet (Canny)** | Structural conditioning — keeps tissue/nucleus boundaries intact during generation, added in later ablation rungs. |
| **LCM-LoRA** | Latent Consistency Model adapter for few-step (4–8 step) inference, traded off against the full 50-step DDIM path for speed. |

Direction is explicit and trained separately: **A2H** (Aperio→Hamamatsu) and
**H2A** (Hamamatsu→Aperio, needed for cycle-consistency checks and for a
downstream classifier that expects Aperio-style input), each at LoRA rank 4
and rank 8.

### Source conditioning (P1-10) and DDIM inversion (P1-11)

The base ablation ladder's colour LoRA never sees the source image during
*training* — direction is only imposed at *inference* time via img2img
strength. A corrective architecture adds a small ControlNet-style branch
conditioned on the source crop during training itself, so the model learns a
genuine `P(target | source)` mapping rather than an unconditional
target-domain prior. Paired with this: replacing the original random-noise
img2img initialisation with **DDIM inversion** (deterministically running the
trained model's own noise prediction backwards from the source image) gives a
source-specific starting point for generation, recovering real structural
fidelity that random-noise initialisation was giving up.

### Post-hoc source-detail fusion (P1-16)

The single biggest remaining gap to classical stain-normalisation methods
(Macenko, Reinhard, Histogram Matching) turned out to be architectural, not a
training problem: resynthesising an *entire* image crop through a VAE
encode/decode caps structural fidelity (SSIM) at a hard ceiling regardless of
backbone or conditioning quality (measured directly with a zero-denoising
VAE-only reconstruction test). The fix does not touch the diffusion model at
all: it fuses the untouched source image's fine structure back onto the
diffusion output's learned colour transformation, entirely as post-processing
outside the UNet. This is the project's best-performing configuration — see
[Key results](../README.md#key-results) and
[`docs/results/RESULTS_SUMMARY.md`](results/RESULTS_SUMMARY.md).

## Training data

Training pairs are **≤50 coordinate-corresponding crops** (few-shot) from a
*single* Aperio/Hamamatsu slide pair (MITOS-ATYPIA-14's A03/H03), registered
by tissue coordinate — not pixel-exact, since affine registration is reserved
for held-out evaluation only, never applied to training data.

## Evaluation

Multi-axis, always reported per-slide plus an outlier-excluded aggregate
(one held-out slide, A06, is a genuine, confirmed colour-gap outlier — never
allowed to stand alone as a pooled number):

- **Colour fidelity** — LAB-histogram Wasserstein distance against real
  paired ground truth (MITOS-ATYPIA-14 provides physical scans of the *same*
  tissue on both scanners, the strongest available test).
- **Structural preservation** — SSIM/PSNR/MAE against the registered
  ground-truth target, plus a HoVer-Net/Lizard Relative-Dice check that the
  normalisation doesn't distort nucleus geometry enough to break downstream
  detection.
- **Clinical utility** — accuracy delta of a downstream atypia classifier
  trained on one scanner's images and evaluated on normalised images from
  the other.
- **Multi-centre generalisation** — inter-centre colour variance on
  CAMELYON17 (5 centres), independent of the MITOS-ATYPIA-14 training/eval
  split.
- **Cycle consistency** — round-trip (A→H→A) reconstruction fidelity against
  the original.

Held-out slides (A06/A08/A09/A13/A16) are never used in any training —
enforced as a standing invariant, checked at every stage.

## Compute cost

The project's success criteria are colour/structural/clinical quality, not
speed — but the trade-off against classical stain-normalisation methods is
real and worth stating plainly. Measured directly from completed cluster
job logs at matched evaluation scale (496 held-out crops): the three
classical baselines combined run **CPU-only** in ~3h27m; a full 50-step
diffusion pass is comparable or slower on a GPU; DDIM inversion (the
mechanism behind this project's best quality results) is clearly slower
again. Few-step LCM inference is the one diffusion configuration that beats
classical on raw speed, but still requires a GPU and trades away colour
fidelity on the harder checkpoints. **This project's diffusion methods win
on quality, not speed or compute cost** — full breakdown and the node-speed
caveats behind these numbers: [`docs/results/RESULTS_SUMMARY.md`](results/RESULTS_SUMMARY.md#addendum-2026-09-30-compute-cost--diffusion-vs-classical-baselines).

Full architecture detail per ablation rung, and the complete numeric
results, live in [`docs/results/RESULTS_SUMMARY.md`](results/RESULTS_SUMMARY.md).
