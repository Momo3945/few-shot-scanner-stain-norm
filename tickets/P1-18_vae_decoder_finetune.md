# P1-18 — Raising the VAE Ceiling: Decoder-Only Fine-Tune, then Full-VAE Fine-Tune

**Status (2026-10-26):** Stage 1 DONE (G1.0/G1.1 pass, G1.2 fails, G1.3 SSIM passes but wLAB
fails, G1.4 fails). **Stage 2a DONE, negative** (anchored full-VAE fine-tune, λ = 1 and 10):
G2.1 fails (self-recon +0.002/+0.004 Aperio, +0.001 Hamamatsu over Stage 1; bar +0.02), G2.3
fails (translated SSIM −0.002/−0.003), G2.4 fails (fused SSIM unchanged, wLAB +0.41/+0.93 vs
the final pipeline). See "Stage 2a results" below; paper appendix A-D, Table IX. **Not run:**
Stage 2b (unanchored + downstream retrain) and G1.5 (CAMELYON17), both listed in the paper.
*Status (2026-10-19):* Stage 1 DONE; G1.5 and Stage 2 open, Stage 2 awaiting the supervisor.
*Earlier status:* 🔄 IN PROGRESS — proposed 2026-10-05, **supervisor-directed**
(Richard Klein asked for both steps, in this order: decoder-only first, then
full VAE). **Stage 0: 0.1, 0.2, 0.4, 0.5 done (job 64246); 0.3 (overfit control)
pending the training script.** No training run yet. Headline so far: `ft_mse`
shares stock's encoder bit-for-bit, so it is already a decoder-only fine-tune.

**Scope note (README rule 2):** supervisor-directed, not a proposal-scoped task,
so it carries no proposal label. The proposal describes a frozen VAE
(§Pipeline Overview, steps 2 and 5); the supervisor has said this does not need
to be treated as a constraint here. Just keep VAE-adapted results labelled as
such (not part of the A0–A5 ladder). P1-14/P1-15 swapped in a *pretrained*
decoder; this is the first time the project trains a VAE component. Everything
else (U-Net, ControlNet, colour LoRA, LCM-LoRA) stays frozen.

**Output folder:** `/datasets/mhoosen/stain-norm/vae/` created 2026-10-05
(empty), one dir per run.

## Motivation

Every diffusion result in this project is bounded by what the VAE can
reconstruct, because the whole crop is resynthesised through it:

```text
                                   SSIM(input, encode->decode(input)), canonical 496 held-out Aperio crops
lossless autoencoder (ideal)       1.000     <- what an ideal VAE would give (dotted line in the figure)
sd-vae-ft-mse   (P1-14)            0.6182
stock SD1.5 VAE (P1-14)            0.5759
```

(The older 0.5393 figure from P1-10 is a different quantity: encode-decode of
Aperio scored against the *registered Hamamatsu* target. Do not mix the two on
one axis.)

The baseline figure is already rendered from the real P1-14 Stage B per-crop
scores: `docs/report/figures/vae_ceiling.png` (script
`docs/report/make_vae_ceiling_figure.py`). Both VAEs sit at 0.46–0.66 per slide,
far from 1.0 — "SSIM is 1 for a perfect autoencoder, and we are nowhere near it."

Where P1-14 → P1-15 already showed the lever works: swapping in a better decoder
raised self-reconstruction SSIM by +0.042 and translated SSIM by +0.061
(0.4960 → 0.5567, 496/496 crops). That decoder was trained by Stability on
natural images (LAION); a decoder trained on **H&E** should have more headroom.

## Why decoder-only has a bound (and why Stage 2 exists)

- The VAE compresses 512×512×3 into 64×64×4 (48×). What the *encoder* discards
  no decoder can recover, so Stage 1 has a bound below 1.0; Stage 0.3 measures
  it (decoder overfit control) so the headroom is known before Stage 2.
- Full-VAE fine-tuning (Stage 2) can raise that bound, but it **moves the latent
  space** the P1-10 colour LoRA and source ControlNet were trained on. That is
  Stage 2's central risk and is handled explicitly below.

## Scientific Questions

1. **Stage 1:** with the encoder frozen (latent space unchanged, so P1-10/P1-11
   remain valid with zero retraining), how much of the self-reconstruction gap
   does an H&E-trained decoder close beyond `sd-vae-ft-mse`, and does that
   carry through to translated outputs (P1-15) and to the fused pipeline (P1-16)?
2. **Stage 2:** does also fine-tuning the encoder raise the ceiling meaningfully
   beyond Stage 1, and can that survive without retraining the downstream
   diffusion components?

## Hypotheses and expected failure modes

- **H1 (Stage 1):** an H&E-trained decoder beats `ft_mse` self-reconstruction
  SSIM by ≥ +0.02 on held-out crops, consistently on every slide.
- **H2 (Stage 1 → translation):** the gain transfers to P1-11 translated outputs.
  *Caveat:* decoder training sees latents of real images; translated latents are
  slightly off that distribution. P1-14→P1-15 transferred (+0.042 → +0.061),
  but that is one precedent, not a guarantee.
- **H3 (fusion absorbs it):** P1-16 fusion lifts SSIM to ~0.729 by borrowing
  source detail outside the decoder. Self-ensembling gave +0.047 pre-fusion and
  **exactly 0 after fusion** (PHASE1-TICKETS P1-11 future-work). A better decoder
  may be absorbed the same way, so **every arm is scored both pre- and
  post-fusion**; a pre-fusion-only win must not be reported as a pipeline win.
- **Risk — scanner/dataset overfitting:** few-shot spirit of the project; the
  decoder sees little data. Mitigated by frame-level validation split, the
  Stage 0.3 overfit control, and an out-of-domain no-regression check.
- **Risk — hallucinated texture:** a decoder trained with perceptual/GAN losses
  can invent plausible nuclei. No GAN loss in this ticket (see Loss).

## Isolation rules

- **Held-out slides A06/A08/A09/A13/A16 are never used in any VAE training or
  validation-for-selection.** Leak check required (same pattern as every other
  trainer here). Hyper-parameters are chosen on an internal validation split only;
  the 496-crop held-out set is scored once per frozen config.
- Training pairs are coordinate-corresponding, not pixel-exact — **irrelevant
  here**, because VAE fine-tuning is pure self-reconstruction (input = target),
  so no pair registration is used or needed.
- The U-Net, ControlNet, colour LoRA, LCM-LoRA, and every existing validated
  inference script are not modified. New scripts are standalone (P1-14/P1-15
  convention).
- Training loss is **not** a success signal (project guardrail). Success is
  held-out SSIM/PSNR/MAE/LAB from the scorer plus the translated-output metrics.

## Stage 0 — Setup and diagnostics (no training)

0.1 **Weight diff, stock vs `ft_mse` encoder.** If `ft_mse` is already a
decoder-only fine-tune of stock (its encoder weights identical), then P1-15 was
effectively "Stage 1 with a pretrained decoder" and Stage 1 here is a direct
continuation. Cheap CPU check; record the result either way. Do not assume.

0.2 **Data inventory.** Count the usable training crops from sources that never
touch held-out slides: `pairs/` (MITOS A03/H03, **both** domains — the decoder
must handle Hamamatsu-coloured latents too, since that is its output domain),
`hist_lora_pool/` (mitos/pannuke/tcga). Derive counts from `train_manifest.csv`
and the pool manifests; do not assume them. Split **by frame** (not by crop) into
train / internal-val to avoid spatial leakage.

0.3 **Capacity / bottleneck control (the "ceiling" measurement).** Following
the project's mandatory `OVERFIT_N` control: train the decoder on a tiny set
(`OVERFIT_N=16` crops) to convergence and score reconstruction on those same
crops. This is the best a decoder can do against the *frozen encoder's* latent,
i.e. the encoder-bottleneck bound. If even the overfit run is far below 1.0, the
bound is the encoder and Stage 2 is the only route higher; if it approaches 1.0,
there is real decoder headroom and the generalisation gap is the problem.

0.4 **Dependency check.** LPIPS is not currently used anywhere in this repo.
Confirm the `lpips` package and its VGG/AlexNet weights are available in the
`stainnorm` env **without touching `torch`** (install with `--no-deps` if
needed, never upgrade torch — CLAUDE.md). The cluster may be offline for weight
downloads, so fetch via a `fetch_models`-style `sbatch`, and verify by byte size,
not exit code (the SDXL-weights incident).

0.5 **Extend the benchmark to score arbitrary VAEs.** Add optional
`--extra-vae label=path` (repeatable) to `src/eval/benchmark_vae_reconstruction.py`
with default behaviour unchanged, so new checkpoints are scored by the *same*
code path as P1-14 (its own advice: never compare across different historical
code paths). Also enable the **Hamamatsu-side** held-out self-reconstruction
(registered Hamamatsu frames are already loaded there) so both domains are
reported on held-out, not only Aperio.

### Stage 0 results (2026-10-05)

Raw outputs: `docs/results/p1_18_vae_finetune/stage0/` (copy of
`/datasets/mhoosen/stain-norm/vae/stage0/`).

- **0.1 weight diff — DONE (job 64246, COMPLETED 00:02:30, `mscluster25`;
  `src/eval/vae_stage0_checks.py`).** Stock vs `sd-vae-ft-mse`: `encoder` (106
  tensors) and `quant_conv` (2) are **bit-identical**; `decoder` (138) and
  `post_quant_conv` (2) are all different (decoder relative L2 diff 1.06,
  `post_quant_conv` 0.13). So **`ft_mse` is already a decoder-only fine-tune of
  the stock VAE**: its latents equal stock's exactly, and P1-15 was effectively
  "Stage 1 with a pretrained (LAION) decoder". Consequences: (i) Stage 1 here is a
  direct continuation of what P1-14/P1-15 already measured, with H&E data instead
  of LAION; (ii) "stock encoder" and "`ft_mse` encoder" are the same thing, so the
  G1.0 frozen-encoder hash check can compare against either; (iii) the optional
  stock-init control arm is a genuine replication of what `ft_mse` did, not a
  different latent space.
- **0.2 data inventory — DONE (read-only cluster queries).** No held-out slide
  appears in any candidate source (checked from filenames/manifests).
  - `pairs/train` (512px, 50 pairs = 100 images) and the 1024px sets
    `train_1024_full96` (96 pairs) / `train_1024_overlap144` (144 pairs) are
    **all slide A03/H03, all drawn from the same 24 frames**. The larger sets are
    overlapping re-crops of those frames, so they add crops, **not independent
    content**. Do not count them as more data.
  - `hist_lora_pool/` is the real source of diversity: **3,000 patches, all
    512×512**: 500 MITOS (11 training slides A03–A18 excl. held-out, **253
    Aperio + 247 Hamamatsu**, ~22–23 per slide per scanner), 1,000 PanNuke, 1,500
    TCGA-BRCA (16 slides; the 10 label-reference slides are already excluded).
  - Hamamatsu content is scarce: 247 pool patches + the A03/H03 pair crops. This
    matters because the decoder must render Hamamatsu-coloured latents (its output
    domain for A→H), so Hamamatsu should be reported separately and not drowned
    out by the PanNuke/TCGA bulk (consider sampling weight for the H&E-Hamamatsu
    subset).
  - **Proposed split (to confirm before training, by frame/slide, never by crop):**
    validate on 2 whole MITOS training slides (both scanners) + 2 TCGA slides + a
    PanNuke subset; train on the rest plus A03/H03 pair frames, de-duplicated
    against pool patches from the same A03/H03 frames. Exact slide choice is set in
    the training script and logged.
- **0.4 LPIPS — DONE (same job).** `lpips 0.1.4` installed into `stainnorm` with
  `--no-deps`; **torch unchanged (2.5.1+cu121 before and after)**. VGG16 backbone
  cached at `/datasets/mhoosen/torch_cache/hub/checkpoints/vgg16-397923af.pth`,
  **553,433,881 bytes** (real, verified by size, not exit code). Sanity forward:
  LPIPS(x, x) = 0.0, LPIPS(x, random) ≈ 0.35. Note LPIPS's own linear-head
  weights ship inside the pip package and were loaded fine.
- **0.5 benchmark extension — code DONE, not yet exercised on the cluster.**
  `src/eval/benchmark_vae_reconstruction.py` gained `--extra-vae LABEL=PATH`
  (repeatable; paired stats vs `ft_mse` in a new `paired_extra` summary key) and
  `--heldout-both-domains`; default behaviour unchanged by construction (new keys
  only, same code path). Compiles locally and on the cluster; first real use will
  be Stage 1's scoring, so it should be treated as unvalidated until then.
- **Trainer written + smoke-tested (job 64286, COMPLETED 00:03:02, `mscluster56`,
  `SMOKE=1`, 5 steps).** `src/train/train_vae_decoder.py` +
  `slurm/train_vae_decoder.slurm`. Verified from the log, not the exit code:
  leak check OK (**train 2,715 = 257 Aperio + 252 Hamamatsu + 2,206 TCGA/PanNuke;
  val 385**, no held-out slide, no overlap); `ft_mse` loaded, scaling_factor
  0.18215, **49,490,199 trainable params** (decoder + post_quant_conv); losses
  finite; **encoder hash identical before/after (G1.0 mechanism works)**;
  steady-state **~1.3 s/step** at batch 8 / 256px crops (so 3,000 steps ≈ 70 min +
  validation, inside the 3 h default limit). The smoke run only validated 8
  PanNuke images (list order), so the MITOS validation path was not exercised yet;
  the code was then changed so smoke validates 2 images per domain.
- **Finding from the smoke log — PanNuke is a poor training/validation source.**
  Init (`ft_mse`) SSIM on the PanNuke validation images was **0.948** vs ~0.6 on
  held-out MITOS, and the manifest's source folder is `upscale_results_train`:
  these 512px patches are upscaled low-resolution images, i.e. smooth and
  trivially reconstructable. Training on them risks teaching the decoder that blur
  is acceptable, and averaging them into the selection metric would hide whether
  MITOS actually improved. Fixes made: validation domains split into
  aperio / hamamatsu / tcga / pannuke; **model selection (`select_ssim`) now uses
  the MITOS domains only**; new `--pannuke-weight` (default 1.0, i.e. no change
  of behaviour yet) so it can be lowered or zeroed. **Decision pending:** run the
  real Stage 1 with PanNuke down-weighted (suggested 0.0–0.3) or at 1.0 as a
  documented variant.
- **0.3 overfit/bottleneck control — RUN, but NOT converged, so it is only a
  lower bound on headroom (job 64408, COMPLETED 00:12:42, `mscluster87`;
  `OVERFIT_N=16`, 600 steps, lr 1e-5, train == val on 8 Aperio + 8 Hamamatsu
  MITOS crops).** Validation SSIM on those same crops: Aperio **0.5980 → 0.6208**,
  Hamamatsu **0.6586 → 0.6783**, `select_ssim` **0.6283 → 0.6496 (+0.021)**; PSNR
  20.38 → 21.10 / 22.24 → 23.00, MAE 18.53 → 16.93 / 14.86 → 13.50. Encoder hash
  unchanged. Caveats that stop this being read as the bottleneck bound: (i) the
  training loss was still falling (L1 0.148 → ~0.118) and SSIM was still creeping
  up between steps 500 and 600 (0.6483 → 0.6496); (ii) the run applied random
  rot90/flip augmentation and random 256px crops, so it was not a true memorisation
  test; (iii) validation was logged only at steps 0, 500, 600, so there is no
  curve. What it does show: with these settings even *seen* crops gain only about
  +0.02 SSIM over `ft_mse`, versus the +0.042 that LAION training gave `ft_mse`
  over stock — so G1.1's +0.02 held-out gate looks hard to reach at this training
  budget. A converged control (no augmentation, higher lr, more steps, val every
  ~200 steps) is needed before drawing any conclusion about the encoder bottleneck.
- **0.3b converged control — DONE (job 64546, COMPLETED 00:36:51, `mscluster55`;
  `OVERFIT_N=16 NO_AUGMENT=1 LR=1e-4`, 2000 steps, same 8 Aperio + 8 Hamamatsu
  crops, val every 200 steps; output `vae/d1_overfit16_conv/`).** `select_ssim`
  (mean of Aperio/Hamamatsu SSIM on the training crops themselves):

  | step | 0 (`ft_mse`) | 200 | 400 | 600 | 800 | 1000 | 1200 | 1400 | 1600 | 1800 | 2000 |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | select SSIM | 0.6283 | 0.6804 | 0.7244 | 0.7672 | 0.7978 | 0.8199 | 0.8383 | 0.8510 | 0.8630 | 0.8707 | **0.8790** |

  Final: Aperio 0.5980 → **0.8756**, Hamamatsu 0.6586 → **0.8824**; PSNR 20.38 →
  25.11 / 22.24 → 26.86; MAE 18.53 → 9.65 / 14.86 → 7.86. Monotonic, stable at
  lr 1e-4 (no divergence, finite losses throughout), encoder hash unchanged. **Not
  plateaued**: still +0.008 per 200 steps at the end. Interpretation, kept
  deliberately narrow: (i) the first control (lr 1e-5 + augmentation, +0.021) was
  badly under-trained, and its pessimism should be discarded; (ii) the frozen
  encoder's latent clearly carries enough information for SSIM ≥ 0.88 on these
  crops, so **the frozen latent is not what limits SSIM at ~0.63**; the decoder is.
  (iii) This is **memorisation of 16 crops by a 49M-parameter decoder**, so it
  bounds what the decoder *can fit*, not what it will *generalise* to unseen
  slides; the real Stage 1 run on ~2.7k images and held-out scoring answer that,
  and the held-out gain will be much smaller than +0.25. (iv) Consequence for
  Stage 2: since the latent is not the limiting factor on seen crops, the case for
  touching the encoder (and the downstream-retrain risk) is weaker than assumed
  when the ticket was written; revisit after the Stage 1 held-out result.
- **Learning rate for the real Stage 1 arms: 1e-4** (stable in 0.3b; 1e-5 is
  demonstrably too slow), identical in both PanNuke arms. `best_val` selection on
  the internal validation split guards against over-training on MITOS crops.

## Stage 1 — Freeze encoder, fine-tune decoder only

**Mechanism:** only `vae.decoder` (and `post_quant_conv`) receive gradients; the
encoder and `quant_conv` are frozen, so the latent `z` for any image is
**bit-identical** to today's. Therefore P1-10's LoRA/ControlNet and P1-11's
inversion are valid unchanged and *no downstream retraining is needed* — the
only change is the last decode, exactly P1-15's isolation.

- **Init:** primary = `sd-vae-ft-mse` (the project's current best decoder);
  optional control arm = stock decoder, to show whether any gain is from our
  data or merely re-deriving `ft_mse`.
- **Encoder:** the stock encoder (as in P1-15). Verify frozen-ness by hashing
  encoder state dict before/after training and logging equality.
- **Training data:** both scanner domains, frame-level split from Stage 0.2.
  Latents can be precomputed once from the frozen encoder (`.mode()`, matching
  the project's deterministic convention) which makes decoder training cheap.
- **Loss (default):** L1 + LPIPS (+ optional SSIM term, since SSIM is the
  reporting metric). No adversarial loss in Stage 1 — it adds instability and the
  hallucination risk above. If Stage 1 gains are blurry-limited, a small GAN term
  is a *separate, later* arm, not part of the first run.
- **Optimiser:** small LR (start 1e-5, AdamW), fp32 or bf16 training (SD1.5 VAE
  can NaN in fp16 training; the project already learned this for SDXL), gradient
  checkpointing if 512px batches exceed 24 GB on a 3090. Final choices set by
  the SMOKE run, not guessed.
- **Cluster conventions:** `train_vae_decoder.slurm` on `bigbatch`, always with
  the bad-node `--exclude` list; env-var overrides `SMOKE=1`, `OVERFIT_N=N`;
  `PYTHONUNBUFFERED=1`; fail-fast on `torch.cuda.is_available()`; resumable
  checkpoints. Outputs to `/datasets/mhoosen/stain-norm/vae/<run>/` — **a new
  top-level folder, so CLAUDE.md's "Cluster data layout" list must be updated
  when the first run is created.** Scoring on `stampede` via its own tiny
  `.slurm`, never over bare ssh.
- **Every sbatch needs explicit go-ahead** (CLAUDE.md rule); this ticket
  authorises none.

### Stage 1 arms (pre-registered 2026-10-05, before any real run)

Two runs, identical except for the PanNuke sampling weight (supervisor-agnostic
design decision by the student; PanNuke here is upscaled/easy, see Stage 0
results):

| Arm | Run name | `PANNUKE_W` | Everything else |
|---|---|---|---|
| D1-pn0 | `d1_pn0` | 0.0 (PanNuke never sampled) | init `ft_mse`, decoder-only, L1 1.0 + LPIPS 0.3, `MITOS_W` 3.0, batch 8, 256px crops, 3000 steps, seed 0; **lr 1e-4** (from converged control 0.3b), identical in both arms; `VAL_EVERY=250` |
| D1-pn03 | `d1_pn03` | 0.3 | same |

**Pre-registered decision rule (fixed before looking at any result):**
1. The winner is chosen on the **internal validation split** (`select_ssim` =
   mean of Aperio and Hamamatsu validation SSIM, at each arm's best-validation
   step), **not** on the held-out slides, so the held-out set is never used to
   choose between configurations.
2. A difference smaller than **0.003 `select_ssim`** is treated as a tie; on a
   tie prefer the arm with the better Hamamatsu SSIM (the scarce, output-domain
   side), then the lower `PANNUKE_W`.
3. Both arms' internal-validation curves are reported. The 496-crop held-out
   benchmark is run **once per frozen config**: the winner always; the other arm
   too if cheap, reported as an ablation and never used to reverse the choice.
4. Also reported for context: per-domain validation (TCGA, PanNuke) so any blur
   regression on the easy sources is visible, plus the A06/A08 qualitative panels
   for sharpness.

### Stage 1 training results (2026-10-06; internal validation only, held-out not yet scored)

Jobs 64639 (`d1_pn0`, COMPLETED 01:07:02, `mscluster56`) and 64640 (`d1_pn03`,
COMPLETED 01:05:57, `mscluster58`); lr 1e-4, 3000 steps, val every 250 on 385
images never trained on (46 Aperio + 45 Hamamatsu MITOS from slides A10/H10/A15/H15,
194 TCGA, 100 PanNuke). Raw logs: `docs/results/p1_18_vae_finetune/stage1/<arm>/`.
Encoder hash unchanged in both (G1.0 mechanism passes).

| | init `ft_mse` | `d1_pn0` best (step 1250) | `d1_pn03` best (step 2250) | `d1_pn0` mean steps ≥1000 | `d1_pn03` mean steps ≥1000 |
|---|---|---|---|---|---|
| select SSIM (Aperio+Hamamatsu mean) | 0.6451 | **0.6720** (+0.0268) | **0.6736** (+0.0285) | 0.6701 (sd 0.0015) | 0.6700 (sd 0.0026) |
| Hamamatsu SSIM | 0.6799 | 0.7066 | 0.7088 | — | — |
| PanNuke SSIM / PSNR (final) | 0.9620 / 35.85 | 0.9589 / ~31–32 | 0.9639 / ~34–35 | — | — |

- **Pre-registered rule outcome:** best-val difference pn03 − pn0 = **+0.0017 <
  0.003 → TIE**; tie-break 1 (Hamamatsu SSIM at the best step) favours
  **`d1_pn03`** (0.7088 vs 0.7066), so **`d1_pn03` `best_val` is the designated
  winner**. Stated honestly: the two arms are **statistically indistinguishable**
  (plateau means 0.6701 vs 0.6700; checkpoint-to-checkpoint sd 0.0015–0.0026 is as
  large as the gap), so the winner label reflects the pre-registered tie-break, not
  a real difference in quality. The best step is also chosen from 13 noisy
  validation points, so the +0.027/+0.029 "best" figures are optimistically
  biased; the plateau mean (**≈ +0.025**) is the fairer estimate of the gain.
- **Learning curve:** nearly all of the gain arrives by step 250 and then plateaus
  (select SSIM 0.666–0.674 from step 250 to 3000, no trend, no collapse). More steps
  at this lr do not help; **the generalising gain is ≈ +0.025 SSIM over `ft_mse`,
  versus the 0.88 reachable by memorising 16 crops** (0.3b), i.e. the limit on
  unseen slides is data/generalisation, not capacity or the frozen latent.
- **PanNuke blur worry — not borne out:** PanNuke SSIM stays ≈ 0.96 in both arms
  (no blurring). What differs is PSNR on PanNuke: excluding it (`pn0`) lets PSNR drop
  from 35.9 to ~31–32 dB, weight 0.3 keeps it at ~34–35 dB. Including a little
  PanNuke protects easy images at no cost to MITOS.
- **G1.1 outlook:** an internal-validation gain of ≈ +0.025 is only just above the
  +0.02 gate and carries selection bias and slide-to-slide variance, so the held-out
  result could land either side of the gate. Not decided until the 496-crop run.

### Stage 1 held-out self-reconstruction result (2026-10-06, job 64645)

Job 64645 (`p1_18_heldout`, COMPLETED 01:43:49, `mscluster58`): canonical 496
held-out crops, both scanner domains, 6 VAEs, the P1-14 code path with the new
`--extra-vae` / `--heldout-both-domains` flags (first real use; worked). Sanity
check: stock and `ft_mse` Aperio SSIM **0.5759 / 0.6182 reproduce P1-14 exactly**.
Data: `docs/results/p1_18_vae_finetune/stage1_heldout/` (`per_crop.csv`,
`summary.json`); figure `docs/report/figures/vae_ceiling.png`.

Frozen winner `d1_pn03_best` vs `ft_mse`, paired per crop:

| Domain / group | n | `ft_mse` SSIM | winner SSIM | ΔSSIM | min per-crop Δ | % crops improved | ΔPSNR | ΔMAE | ΔLAB total | ΔwLAB |
|---|---|---|---|---|---|---|---|---|---|---|
| Aperio ALL | 496 | 0.6182 | 0.6460 | **+0.0278** | +0.0161 | 100.0 | +0.75 | −1.56 | +0.21 | +0.42 |
| Aperio excl. A06 | 432 | 0.6348 | 0.6628 | +0.0280 | +0.0211 | 100.0 | +0.76 | −1.60 | +0.08 | +0.31 |
| Aperio A06 | 64 | 0.5061 | 0.5321 | +0.0261 | +0.0161 | 100.0 | +0.68 | −1.27 | **+1.08** | +1.15 |
| Aperio A08 / A09 / A13 / A16 | 112/96/64/160 | — | — | +0.0271 / +0.0282 / +0.0291 / +0.0281 | ≥ +0.021 | 100.0 | +0.73 to +0.78 | −1.47 to −1.73 | +0.08 / 0.00 / **+1.29** / −0.34 | — |
| Hamamatsu ALL | 496 | 0.7573 | 0.7787 | **+0.0214** | **−0.0260** | 98.4 | +0.53 | −0.88 | **+0.58** | +0.61 |
| Hamamatsu excl. A06 | 432 | 0.7673 | 0.7888 | +0.0215 | +0.0137 | 100.0 | +0.54 | −0.90 | +0.39 | +0.45 |
| Hamamatsu A06 | 64 | 0.6901 | 0.7106 | +0.0205 | −0.0260 | **87.5** | +0.49 | −0.77 | **+1.80** | +1.73 |
| Hamamatsu A08 / A09 / A13 / A16 | 112/96/64/160 | — | — | +0.0210 / +0.0240 / +0.0209 / +0.0207 | ≥ +0.0137 | 100.0 | +0.51 to +0.58 | −0.81 to −1.08 | +0.12 / +0.36 / +0.77 / +0.45 | — |

All paired 95% CIs sit well above zero (e.g. Aperio [+0.0276, +0.0280], Hamamatsu
[+0.0208, +0.0219]). Ablation checkpoints (pooled ΔSSIM vs `ft_mse`, Aperio /
Hamamatsu): `d1_pn03_best` +0.0278 / +0.0214; `d1_pn0_best` +0.0267 / +0.0199;
`d1_pn03_final` +0.0213 / +0.0168; `d1_pn0_final` +0.0236 / +0.0188. The
pre-registered winner is also the best on held-out, but the margin over
`d1_pn0_best` (≈ 0.001) is far inside the noise, and the held-out result could
not have changed the choice.

**Gate G1.0 (encoder frozen): PASS** (hash identical, both arms). **G1.1
(self-reconstruction, ≥ +0.02 over `ft_mse`, positive on every slide, CI above 0,
PSNR/MAE not worse): PASS, narrowly on Hamamatsu.** Aperio clears it everywhere
(+0.026 to +0.029, 100% of crops improved). Hamamatsu pooled +0.0214 and every
slide at +0.0205 to +0.0240, so the mean gate holds, but the margin is about
0.001 on A06, A08, A13 and A16, and **8 of 64 A06 Hamamatsu crops got worse**
(min −0.026), so "positive on every crop" does not hold there. PSNR and MAE
improve on every group.

**Open concern — colour fidelity (feeds G1.2, G1.3):** the SSIM-selected decoder
drifts colour slightly relative to its input: identity LAB distance rises
**+0.21 (Aperio) and +0.58 (Hamamatsu)** pooled, with A06 (+1.08 / +1.80) and
A13 (+1.29 Aperio) worst; only A16 Aperio improves. The *final* checkpoints drift
far more (Hamamatsu +1.7 to +3.9 LAB per slide; Aperio +0.1 to +2.1), and the
best-vs-final spread shows colour is not controlled by anything in the training
objective (L1 + LPIPS; LPIPS is largely colour-blind) and is not stable across
checkpoints. Colour is this project's primary metric, so a structure gain bought
with colour drift is not automatically a pipeline win. Whether the drift matters
at ~25 wLAB translation scale is exactly what G1.3 (translated drop-in) tests;
a colour-aware loss term (e.g. L1 in LAB space or on low-pass colour) is the
obvious remedy if G1.3 shows a colour cost, and is **not** done yet.

### G1.3 / G1.4 execution plan (2026-10-06)

**G1.2 (artifact panels) was deferred, not run:** the benchmark script's
qualitative-panel code only handles stock/`ft_mse`, so it needs a code change; G1.3
was started first per the plan to test "final output improves" directly. G1.2 is
still owed (and is where the colour-drift concern should be looked at by eye).

**G1.3 run history (2026-10-06):** job 64762 landed on `mscluster61`, was measured
at 14.4 s per crop-seed (~2.2x slower than normal, projecting ~7 h vs the 5 h cap)
and was **cancelled by hand after 1:13** (`CANCELLED`, not a failure of the method;
partial outputs in `.../p1_18_d1_pn03_full/` are to be ignored). Resubmitted as
**job 64803** with `mscluster61` added to the exclude list, new output dir
`eval/p1_15_alt_decoder/p1_18_d1_pn03_full_r2/`. Everything below applies to the
rerun.

**G1.3 (translated drop-in), submitted 2026-10-06 as job 64762** (`p1_18_g13`):
`slurm/infer_p1_11_alt_decoder.slurm` with the new optional `ALT_DECODER=
.../vae/d1_pn03/best_val` env override (default unchanged), the validated P1-15
script untouched (it already accepts `--alt-decoder-repo` as a path, aborts on any
latent-channel/scaling mismatch, and always encodes with the stock VAE). `correct`
source, all 496 crops, seeds 0/1/2, `--time=05:00:00`; output
`eval/p1_15_alt_decoder/p1_18_d1_pn03_full/` (each crop's single shared latent is
decoded by stock and by the new decoder). Scoring is a separate job
(`score_p1_15_alt_decoder.slurm`, ~2 h CPU). Comparison targets: P1-11 stock
decoder 0.4960 / 26.01 wLAB; P1-15 `ft_mse` decoder 0.5567 / 25.21 wLAB (per-slide
table in P1-15). Pass: SSIM > 0.5567 **and** wLAB ≤ 25.21.

**G1.4 (fusion) — tooling prepared, nothing run:** F3 computes
`Δ = LAB(H_pred) − LAB(A_V)`, blurs it (σ=8) and adds β·Δ_colour (β=0.50) to the
untouched raw source, so the **fused image's fine structure comes from the raw
Aperio crop, not from the decoder**; only the low-frequency colour field depends on
the decoder. Prediction: a sharper decoder will therefore add little after fusion
(the self-ensemble gain vanished completely, H3). Because A_V is subtracted, it must
come from the *same decoder* as H_pred so decoder-specific colour offsets cancel.
New pieces (all additive, defaults unchanged): `src/eval/vae_only_newdecoder.py` +
`slurm/vae_only_newdecoder.slurm` (stock-encoder + new-decoder reconstruction A_V of
the 496 held-out source crops, read from the checksum-verified full identity run
`eval/p1_10_ddim_inversion/identity_full_heldout`, no registration needed, minutes
of GPU); `fuse_source_detail.py`/`.slurm` gained `--pred-method` /
`--pred-source-mode` (env `PRED_METHOD`/`PRED_SOURCE_MODE`) to pick the
`alt_decoder` / `correct_alt_decoder` rows of the G1.3 manifest. Sanity arm: fusing
the same run's `stock_decoder` rows with the *original* identity manifest must
reproduce P1-16's 0.729. Order: A_V job (independent of G1.3, can run now) → after
G1.3: fuse stock rows (sanity) + fuse new-decoder rows → score both
(`score_outputs.slurm`, ~7.5 min each) → compare.

### G1.2 result: no artefacts, but the decoder over-smooths (2026-10-18)

Panels: `docs/report/figures/p1_18_g12_recon.jpg` (original | stock | `ft_mse` | P1-18,
self-reconstruction) and `p1_18_g12_translated.jpg` (Aperio source | real Hamamatsu |
the same P1-11 latent through stock / `ft_mse` / P1-18), A06 and A08 crop (0,0), seed
0, full crop plus a 128 px zoom (nearest-neighbour). Script
`docs/report/make_p1_18_panels.py`; inputs fetched to `docs/report/cluster_raw/p1_18/`.
Checked before use: the P1-14 original and stock tiles are pixel-identical to the
identity run's source crop and stock VAE-only output (max |Δ| = 0).

- **No checkerboarding, ringing or invented nuclei.**
- **Over-smoothing:** by eye the P1-18 decoder is visibly softer than stock and
  `ft_mse` (stromal fibres, chromatin texture). Laplacian variance (fine-detail
  energy) relative to the input, A06 / A08: reconstruction stock 1.11 / 1.17,
  `ft_mse` 0.52 / 0.59, **P1-18 0.21 / 0.30**.
- **The target scanner is just as soft:** real Hamamatsu carries 0.18 / 0.24 of the
  Aperio source's fine detail; translated output stock 0.91 / 1.02, `ft_mse` 0.44 /
  0.51, **P1-18 0.21 / 0.27**. So much of P1-18's translated SSIM gain (G1.3) is
  matching Hamamatsu's softer rendering, not recovering structure. The decoder was
  trained on both scanners, which is the plausible source of the softness.
- **Colour shift introduced by the fine-tune:** mean blue-channel offset vs input,
  A06 / A08: stock −0.7 / −0.6, `ft_mse` −1.0 / −1.3, **P1-18 −5.0 / −3.3** levels.
- **G1.2 verdict: FAIL on over-smoothing** (no artefacts). Rests on two crops; a
  496-crop Laplacian-variance comparison is the cheap way to confirm it and is not run.

### G1.3 result: translated drop-in — structure passes strongly, colour gate fails (2026-10-06)

Inference job 64803 (COMPLETED 03:35:17, `mscluster52`, 6.3 s/crop-seed; the first
attempt 64762 was cancelled on slow `mscluster61`), scoring job 64932 (COMPLETED
00:42:01). 496 crops x 3 seeds, `correct` source, the validated P1-11/P1-15 trajectory;
each crop's single shared latent is decoded by stock and by the new decoder. Manifest
2,976 rows. **Validity check:** the stock-decoder rows are bit-identical to P1-15's
(max |ΔSSIM| = 0, max |ΔwLAB| = 0 over all 1,488 crop-seeds), so the per-crop
comparison against the P1-15 `ft_mse` decoder is a genuine paired comparison. Data:
`docs/results/p1_18_vae_finetune/g13/`.

| Decoder (same latents) | SSIM | wLAB | LAB total | ΔE2000 | PSNR | MAE |
|---|---|---|---|---|---|---|
| stock (= P1-11 canonical) | 0.4960 | 26.01 | 25.10 | 11.16 | 16.79 | 28.97 |
| `ft_mse` (P1-15) | 0.5567 | **25.21** | 24.53 | 10.26 | 17.68 | 26.26 |
| **new `d1_pn03_best`** | **0.6163** | 26.34 | 25.86 | **9.96** | **18.47** | **23.87** |

Per slide (wLAB, new vs `ft_mse`): ALL excl. A06 18.44 vs 17.57; A06 79.61 vs 76.79;
A08 19.33 vs 18.02; A09 16.72 vs 16.64; A13 22.40 vs 21.44; A16 17.27 vs 16.25.
Paired new − `ft_mse` (1,488 crop-seeds): **ΔSSIM +0.0596, new better on 100% of
crop-seeds, every slide +0.054 to +0.064**; **ΔwLAB +1.13, new better on only 19.4%**
(A06 +2.81 / 6.3%, A09 +0.09 / 45.8% ≈ tie); ΔΔE2000 −0.30 (better), ΔPSNR +0.79.
Paired new − stock: ΔSSIM +0.1203 (100%); ΔwLAB +0.33 overall — better on non-outlier
slides excl. A06 (−0.48) but **A06 +5.78 (worse on 100% of A06 crop-seeds)**.

**G1.3 gate (SSIM > 0.5567 AND wLAB ≤ 25.21): SSIM passes by a wide margin (+0.060);
the wLAB condition FAILS (26.34, +1.13).** Per the ticket's own rule (P1-15
"Failure Interpretation"), improved SSIM with worse windowed LAB is *a trade-off,
not an automatic replacement*. Notes: ΔE2000 (pixelwise colour error) improves on
every slide while LAB/windowed-LAB (distribution distance) worsens, consistent with
a small systematic colour offset (the blue-channel shift of ~−2.8 levels seen in the
A_V reconstructions) on top of better-aligned structure; A06 (the colour-gap
outlier) is hit hardest. The new decoder partly undoes the colour gain `ft_mse` had
made over stock on the non-outlier slides (excl. A06 wLAB: stock 18.92 → `ft_mse`
17.57 → new 18.44).

### G1.4 result: fusion absorbs the SSIM gain and the colour drift gets through (2026-10-06)

Jobs 64933/64934 (stock-decoder rows, sanity arm) and 64935/64936 (new-decoder rows),
all COMPLETED; F3 at P1-16's frozen sigma=8, beta=0.50; 496 crops, **seed 0 only**
(the identity run that supplies A_raw/A_V has seed 0 only; 992 translate rows for
seeds 1-2 are skipped with a logged warning — identical to P1-16's own protocol
and to both arms). Raw data: `docs/results/p1_18_vae_finetune/g14/`.

**Sanity arm passes:** fusing the *stock-decoder* rows with the original identity
manifest reproduces P1-16's held-out result to every printed digit (ALL SSIM
0.72895, wLAB 27.07792, recovery Δlab 7.8096, every per-slide row identical), so the
new `--pred-method`/`--pred-source-mode` plumbing and the scorer are verified.

| Scope | SSIM stock / new / Δ | wLAB stock / new / Δ | recovery Δlab stock / new / Δ |
|---|---|---|---|
| ALL (496) | 0.7290 / 0.7289 / **−0.0001** | 27.08 / 27.74 / **+0.66** | 7.81 / 7.08 / **−0.73** |
| ALL excl. A06 (432) | 0.7455 / 0.7455 / 0.0000 | 18.65 / 19.10 / +0.44 | — |
| A06 | 0.6171 / 0.6164 / −0.0007 | 83.94 / 86.06 / **+2.12** | 11.25 / 9.09 / −2.15 |
| A08 / A09 / A13 / A16 | unchanged to 4 dp | +0.25 / +0.90 / **−0.16** / +0.54 | −0.34 / −0.98 / **+0.13** / −0.64 |

Paired per crop: SSIM change ≈ 0 (mean −0.00008, new better on 50.8% — a coin flip);
**wLAB worse on 85.1% of crops** (mean +0.66, median +0.53, new decoder better on only
14.9%); LAB total worse on 87.3%. Only A13 improves on colour.

**Interpretation:** (i) **H3 confirmed for structure:** the fused image takes its
fine structure from the raw Aperio crop, so the decoder's +0.025 SSIM disappears
entirely after fusion (exactly the pattern the self-ensemble showed). (ii)
**Unexpected, and contrary to the prediction written above:** the decoder's colour
drift is *not* cancelled by using the same decoder for A_V; colour recovery gets
*worse* after fusion. A plausible reason (not tested here) is that F3's subtraction
only cancels an offset the decoder applies identically to both images, whereas the
drift measured in self-reconstruction differs between Aperio-coloured and
Hamamatsu-coloured content (identity LAB +0.21 Aperio vs +0.58 Hamamatsu), so a
residual passes through the low-pass colour field. (iii) **G1.4 verdict: FAIL —
no pipeline benefit, small colour cost.** Limited to seed 0, one decoder checkpoint,
one fusion setting.

### Stage 1 gates (in order; stop at the first failure and report)

| Gate | Pass condition |
|---|---|
| G1.0 encoder frozen | encoder weight hash unchanged; `z` for 20 crops bit-identical |
| G1.1 self-recon (held-out, 496 crops, both domains) | mean ΔSSIM ≥ **+0.02** over `ft_mse`, positive on **every** slide incl. A06, paired bootstrap CI above 0; PSNR/MAE not worse |
| G1.2 artifacts | A06 + A08 panels: no checkerboarding, ringing, colour drift, invented nuclei (visual, not just scored) |
| G1.3 translated drop-in | re-decode P1-11's translation (P1-15 script path, new `--decoder-path`; same seeds/steps/f=1.00) — SSIM > P1-15's 0.5567 **and** wLAB ≤ 25.21, per-slide table, A06 reported separately |
| G1.4 fused pipeline | P1-16 F3 (σ=8, β=0.50, unchanged) on the new decoder's output — compare to 0.729 pooled; report whether the gain survives (H3) |
| G1.5 no regression off-domain | reconstruct CAMELYON17 crops (raw patches already on the cluster): SSIM/PSNR not materially below `ft_mse` (decoder must not have collapsed onto MITOS) |

Gate thresholds are practical, supplementary thresholds in the same spirit as
P1-14's +0.01 — not proposal-defined pass criteria. State them as such in
results.

## Stage 2 — Fine-tune the full VAE (encoder + decoder)

Only start after Stage 1 is reported (supervisor's stated order), and only the
sub-steps whose preconditions hold.

**Why this is harder:** updating the encoder changes `z`. The P1-10 colour LoRA
and source ControlNet were trained against the *original* latent distribution
(and the SD1.5 `scaling_factor` 0.18215 was calibrated to it). Two sub-options:

- **2a — Anchored full fine-tune (drop-in attempt).** Train encoder+decoder with
  the Stage 1 loss **plus a latent anchor**: `λ·||E_new(x) − E_orig(x)||²`
  (and/or a KL/moment-matching term to the original latent statistics), so the new
  latent stays close enough that downstream components still work. Sweep λ on the
  internal-val split; report the ceiling gained vs. latent drift
  (mean/std shift, per-channel cosine similarity to original `z`). Recompute and
  log the effective latent std / scaling factor rather than assuming 0.18215.
  *Do-no-harm test:* run P1-11 translation with the new encoder+decoder and the
  **unchanged** LoRA/ControlNet; compare to Stage 1's translated result. If
  translation degrades, 2a fails regardless of its self-recon score.
- **2b — Unanchored full fine-tune + downstream retrain (conditional).**
  Highest ceiling, but effectively **a new P1-10 run** (retrain colour LoRA +
  source-conditioning ControlNet on the new latents) and a new P1-11 inversion
  validation. Do **not** start 2b unless 2a shows (i) a self-recon ceiling
  meaningfully above Stage 1 *and* (ii) the headroom cannot be reached with a
  small anchor. 2b needs its own scoping/compute estimate and supervisor
  go-ahead; this ticket only defines the decision rule.

### Stage 2 gates

| Gate | Pass condition |
|---|---|
| G2.1 ceiling | self-recon ΔSSIM over **Stage 1's best** ≥ +0.02 (otherwise the encoder is not the bottleneck worth the risk) |
| G2.2 latent compatibility | measured drift small enough that G2.3 holds; report drift numbers, not just pass/fail |
| G2.3 do-no-harm | translated SSIM and wLAB with *unchanged* downstream ≥ Stage 1's, per slide |
| G2.4 fused pipeline | as G1.4 |

### Stage 2a results (2026-10-26) — negative: the encoder is not the bottleneck

**Design (fixed before results):** init from the Stage 1 winner `vae/d1_pn03/best_val` (stock
encoder, so the anchor reference is the original encoder). Same recipe as that winner (lr 1e-4,
3,000 steps, batch 8, crop 256, L1 + 0.3 LPIPS, MITOS_W 3, PANNUKE_W 0.3, val every 250,
`best_val` on the internal split), all 83,653,863 VAE parameters trainable, anchor
`λ·MSE(E(x), E_0(x))` on raw latents. Two arms (λ = 1, λ = 10), both run end to end, reported
as a sweep with no held-out pick. Downstream (Canny `lora/a2h_cond_r8/best` + ControlNet) unchanged;
scaling factor 0.18215 kept. Scope: 2a only, so 2b was not run (it would not fit before 2 Nov). Hard cutoff 28 Oct, met.

**Code:**
- `train_vae_decoder.py`: `--init-path`, plus latent-drift logging in `validate()` for full mode
  (relative MSE, per-channel cosine, mean/std shift, latent std);
- `train_vae_decoder.slurm`: `INIT_PATH`, `GRAD_CKPT`;
- `infer_colour_source_ddim_inversion.py`: `--vae-path`, with a `VAE_PATH` override in `infer_p1_10_ddim_inversion.slurm`;
- `vae_only_newdecoder.py`: `--full-vae`, with a `FULL_VAE` override in its slurm file;
- `docs/results/p1_18_vae_finetune/stage2/compare_stage2.py`.

All defaults unchanged.

**Jobs (all COMPLETED; bigbatch):**
- **Controls:** smoke 75610 (encoder hash changed, drift fields present) and overfit
  control 75611 (`OVERFIT_N=16`, 600 steps, 8:31): select SSIM 0.657 → 0.764, no NaN, drift rel MSE 0.006.
- **Training:** 75630 (λ1, 1:07, mscluster84) and 75631 (λ10, 1:04, mscluster86), at about 1.15 s/step.
- **Held-out self-recon benchmark:** 75632 (1:27).
- **Translation (496 crops, seed 0):** 75633 and 75638 (1:47 each), logs confirm
  `VAE replaced ... scaling_factor=0.18215`.
- **A_V with the new encoder:** 75634 and 75639.
- **F3 σ8 β0.5:** 75635 and 75640.
- **Scoring:** fused 75636 and 75641, unfused 75637 and 75642.

Data: `docs/results/p1_18_vae_finetune/stage2/` (`comparison.txt`, per-arm CSVs, `val_log.jsonl`, train logs).

**Internal validation:**
- Best `select_ssim` vs the Stage 1 init (0.6736): λ1 0.6751 (step 250), λ10 0.6761 (step 2250).
- The curve is flat or noisy from step 250 to 3000 (0.666–0.676), with no trend.
- **G2.2 drift at the best checkpoints (Aperio val):** relative MSE 0.0083 / 0.0085, cosine
  0.9957 / 0.9956, latent std 5.05 / 5.08 vs 5.08. λ = 10 barely reduces the drift compared with λ = 1.

**Held-out (496 crops; Δ vs the named baseline, paired 95% CI):**

| | Stage 1 decoder | full λ=1 | full λ=10 |
|---|---|---|---|
| G2.1 self-recon SSIM, Aperio | 0.6460 | 0.6478 (+0.0018 [+0.0016, +0.0020]) | 0.6495 (+0.0035 [+0.0033, +0.0037]) |
| G2.1 self-recon SSIM, Hamamatsu | 0.7787 | 0.7798 (+0.0011) | 0.7795 (+0.0008) |
| self-recon wLAB, Aperio / Hamamatsu | 6.22 / 3.90 | 5.92 / 4.79 | 5.50 / 3.98 |
| G2.3 translated SSIM (unfused) | 0.6163 | 0.6142 (−0.0020 [−0.0024, −0.0016]) | 0.6129 (−0.0033 [−0.0037, −0.0030]) |
| G2.3 translated wLAB (unfused) | 26.34 | 26.55 (+0.21) | 25.00 (−1.34 [−1.42, −1.26], better on every slide) |
| G2.4 fused SSIM (final pipeline 0.7289) | 0.7289 | 0.7288 (−0.0001) | 0.7289 (−0.0000) |
| G2.4 fused wLAB (final pipeline 27.08) | 27.74 | 27.49 (+0.41 [+0.36, +0.46]) | 28.01 (+0.93 [+0.88, +0.99]) |

**Gates:**
- **G2.1 FAIL** for both arms: every slide and domain sits at +0.0005 to +0.0074 against a bar of
  +0.02. Sanity check: `ft_mse` comes out −0.0278 vs Stage 1, which reproduces job 64645 exactly.
- **G2.3 FAIL** for both. SSIM is lower on every slide except A09 at λ1 (0.0000); wLAB is worse
  at λ1 and better at λ10.
- **G2.4 FAIL** for both: SSIM is unchanged, and wLAB is worse pooled. Per slide, λ1 is worse on
  4 of 5 (better only on A13) and λ10 is worse on all 5.

**Leak guard:** on A08_00A_x0_y0, the fused output differs from raw Aperio by 3.40 / 3.47 and
from Hamamatsu by 19.24 / 19.31 (mean |Δ|/255), against 20.49 between raw Aperio and Hamamatsu.

**Interpretation:** unfreezing the encoder adds almost nothing to the ceiling. This is consistent
with Stage 0.3b, where the frozen latent already allowed SSIM 0.88 when memorising 16 crops, so the
limit on unseen slides is generalisation, not the encoder. The λ = 10 arm's unfused colour gain does not
survive F3, because F3 subtracts that VAE's own A_V. Stage 2b is not motivated by these numbers and was
not run. The ceiling figure was not extended: the new arms sit within 0.004 of the Stage 1 line and
would be indistinguishable from it. Table IX in the paper gives the numbers instead (a deviation from
the "Reporting requirements" below).

## Reporting requirements

- **Per-slide always:** ALL, ALL excl. A06, A06, A08, A09, A13, A16, never pooled
  alone (project guardrail). Both scanner domains for self-reconstruction.
- **Required comparison table** (self-recon and translated): stock, `ft_mse`,
  Stage 1 decoder, Stage 2a (if run), plus the classical baselines for the
  translated rows; pre-fusion and post-fusion columns.
- **The explicit ceiling graph (supervisor's "VAE comparison line graph"):**
  extend `docs/report/make_vae_ceiling_figure.py` — add a line per arm
  (`ft_decoder`, `ft_full`) to its `LINES` list using each arm's `per_crop.csv`.
  Per-slide SSIM on the x-axis, a **dotted line at SSIM = 1.0** for the lossless
  ideal, and (new) a dotted line at the Stage 0.3 overfit-control value as the
  **encoder-bottleneck bound for decoder-only training**. The baseline (stock vs
  `ft_mse` vs ideal) is already rendered at `docs/report/figures/vae_ceiling.png`.
- **Interpretation guard:** a higher self-recon SSIM is not evidence of a better
  pipeline unless G1.3/G1.4 agree (the P1-15 "Failure Interpretation" rule).

## Outcome interpretation

- **Stage 1 positive, survives fusion:** adopt the decoder and re-run any
  headline numbers that cite the old decoder.
- **Stage 1 positive pre-fusion only (H3):** report honestly as a structural
  improvement to the *raw diffusion output* that the fused pipeline does not
  need; do not claim a final-pipeline gain.
- **Stage 1 flat (< +0.01 over `ft_mse`):** the bound is the encoder bottleneck
  (confirm with Stage 0.3). Report that as a finding about the 48× latent
  bottleneck, which motivates Stage 2 or a different latent (e.g. SDXL/SD3.5's
  16-channel VAEs, already partly measured in P3-08/PXC), not more decoder tuning.
- **Stage 2a drifts latents / hurts translation:** the encoder cannot be touched
  without retraining the translator; stop at Stage 1 unless 2b is separately
  approved.

## New files (standalone; the only validated script touched is the additive flag in 0.5)

```text
src/train/train_vae_decoder.py       # --mode decoder|full, --init ft_mse|stock, --anchor-lambda
slurm/train_vae_decoder.slurm        # bigbatch, SMOKE / OVERFIT_N env overrides
slurm/score_vae_ft.slurm             # stampede CPU scoring (template: score_p3_07.slurm)
src/eval/benchmark_vae_reconstruction.py   # +optional --extra-vae label=path (default unchanged)
docs/report/make_vae_ceiling_figure.py     # already exists; add arms when available
docs/results/p1_18_vae_finetune/README.md  # results, written after runs (per-phase convention)
```

## Acceptance criteria (ticket-level)

1. Stage 0 complete: encoder weight-diff result, data inventory, overfit-control
   ceiling, LPIPS availability, benchmark extension — all recorded.
2. Stage 1 run and every gate reported pass **or** fail with numbers; no result
   reported from a held-out-tuned config.
3. If Stage 1 passes G1.1–G1.2, G1.3–G1.5 are run and reported pre-/post-fusion.
4. Stage 2 either run per its gates or explicitly deferred with the decision-rule
   evidence (Stage 0.3 bound + Stage 1 outcome).
5. Ceiling line graph updated with all arms, dotted ideal and bottleneck lines.
6. Held-out leak check logged. (CLAUDE.md data-layout list already updated for `vae/`.)

## Open questions for the supervisor

- Appetite for Stage 2b (retraining P1-10 components on a new latent space) if
  2a's ceiling gain is real but needs a larger latent shift than the anchor allows?
- Should the SSIM ≥ +0.02 gate be the bar, or is a smaller gain still worth
  adopting given the per-slide consistency P1-14 showed?
