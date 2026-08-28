# P1-14 — VAE Reconstruction Benchmark: Stock SD1.5 vs. `sd-vae-ft-mse`

**What this tests:** the 0.5393 VAE-only ceiling (established in P1-10) is
a decoder property, not a denoising one. Does a reconstruction-focused
pretrained VAE (`stabilityai/sd-vae-ft-mse`) raise that ceiling as a pure
drop-in swap, no retraining? Both VAEs confirmed architecturally identical
(same param count, `scaling_factor`) by reading the loaded configs
directly. Full mechanism: Pipeline Anatomy §09.

**Status:** 🔄 Stage A ✅ done and positive; Stage B data is now on the
cluster (pulled below) but **not yet analysed or written up** — treat as
not yet reported, not as a result, until `RESULTS_SUMMARY.md` is updated.

**Stage A (isolated self-reconstruction, no diffusion at all, 50
internal-validation pairs):**

| Domain | Stock SSIM | `ft_mse` SSIM | ΔSSIM | 95% bootstrap CI |
|---|---|---|---|---|
| Aperio | 0.5477 | 0.5957 | **+0.0480** | [+0.0470, +0.0490] |
| Hamamatsu | 0.5862 | 0.6312 | **+0.0450** | [+0.0443, +0.0457] |

Clears the ticket's own meaningful-gain gate (≥+0.01 absolute SSIM) by
nearly 5×, in both scanner domains almost equally.

**Files:** `eval/stage_a/per_crop.csv`, `summary.json` (table above),
`vae_config_inspection.json`. `eval/stage_b/per_crop.csv`, `summary.json` —
present, pulled for the record, genuinely unanalysed as of this reorg.

**Full narrative:** `docs/results/RESULTS_SUMMARY.md` (Stage A only) and
`tickets/PHASE1-TICKETS.md` P1-14, `tickets/
P1-14_vae_reconstruction_benchmark.md`.
