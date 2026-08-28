# P1-14 — VAE Reconstruction Benchmark: Stock SD1.5 vs. `sd-vae-ft-mse`

**What this tests:** the 0.5393 VAE-only ceiling (established in P1-10) is
a decoder property, not a denoising one. Does a reconstruction-focused
pretrained VAE (`stabilityai/sd-vae-ft-mse`) raise that ceiling as a pure
drop-in swap, no retraining? Both VAEs confirmed architecturally identical
(same param count, `scaling_factor`) by reading the loaded configs
directly. Full mechanism: Pipeline Anatomy §09.

**Status:** ✅ CLOSED/POSITIVE (2026-08-28) — Stage A and Stage B (canonical
496-crop held-out set) both confirm `sd-vae-ft-mse` as a true drop-in
reconstruction-fidelity improvement, positive on every held-out slide.

**Stage A (isolated self-reconstruction, no diffusion at all, 50
internal-validation pairs):**

| Domain | Stock SSIM | `ft_mse` SSIM | ΔSSIM | 95% bootstrap CI |
|---|---|---|---|---|
| Aperio | 0.5477 | 0.5957 | **+0.0480** | [+0.0470, +0.0490] |
| Hamamatsu | 0.5862 | 0.6312 | **+0.0450** | [+0.0443, +0.0457] |

**Stage B (canonical 496-crop Aperio held-out set, job 47232):** pooled
stock SSIM=0.5759 → `ft_mse` SSIM=0.6182, mean ΔSSIM=**+0.0422** (95% CI
[+0.0419, +0.0426]). Per-slide (never let pooled ALL stand alone):

| Slide group | n | mean ΔSSIM | 95% CI |
|---|---|---|---|
| ALL | 496 | **+0.0422** | [+0.0419, +0.0426] |
| ALL excl. A06 | 432 | **+0.0416** | [+0.0412, +0.0420] |
| A06 | 64 | **+0.0462** | [+0.0453, +0.0471] |
| A08 | 112 | **+0.0419** | [+0.0412, +0.0426] |
| A09 | 96 | **+0.0439** | [+0.0433, +0.0446] |
| A13 | 64 | **+0.0413** | [+0.0405, +0.0420] |
| A16 | 160 | **+0.0403** | [+0.0396, +0.0410] |

Every slide gains a similar amount, including A06 (the largest gain, not a
degraded one) — clears the ticket's ≥+0.01 gate everywhere. Qualitative
check (A06+A08 panels, job 47259): `ft_mse` visibly crisper chromatin
texture than stock's mild blur, no checkerboarding/ringing/colour drift,
lower-amplitude abs-error heatmap. All four acceptance criteria pass.

**Files:** `eval/stage_a/`, `eval/stage_b/` — `per_crop.csv`,
`summary.json`, `vae_config_inspection.json`.

**Full narrative:** `docs/results/RESULTS_SUMMARY.md`, `tickets/
PHASE1-TICKETS.md` P1-14, `tickets/P1-14_vae_reconstruction_benchmark.md`
(all three now carry the full Stage A + Stage B write-up).
