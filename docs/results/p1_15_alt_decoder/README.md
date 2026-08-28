# P1-15 — P1-11 with the Winning Alternate Decoder

**What this tests:** keeps P1-11's DDIM inversion, conditioning, and
translated latent completely unchanged, and swaps only the final decoding
stage to P1-14's winning VAE (`sd-vae-ft-mse`) — a decoder-only isolation,
so any SSIM change is attributable to the decoder alone. Gated on P1-14
finding a meaningful reconstruction gain (it did).

**Status:** ✅ CLOSED/POSITIVE (2026-08-28) — full 496-crop x 3-seed
held-out evaluation, all 5 slides. Pooled SSIM 0.4960 → **0.5567**
(+0.0607), alt_decoder wins **496/496 crops** on SSIM, and pooled colour
recovery also improves (windowed LAB 26.01 → 25.21). Every slide's SSIM
improves by a similar amount; on the non-outlier slides (A08/A09/A13/A16)
colour improves too, while A06 (the project's known colour-gap outlier)
trades a modest colour regression for its structural gain. All 6
acceptance criteria pass. No retraining of P1-10 required.

**Files:**
- `eval/sanity_check/` — compatibility verification + exact-latent A/B
  (job 47308).
- `eval/smoke_correct/`, `eval/smoke_shuffled/`, `eval/smoke_all_summary/`
  — Stage 1 smoke test + source-conditioning sanity check (A06+A08
  subset, jobs 47310/47337, scored by 47391).
- `eval/full_correct_summary/` — the canonical full 496-crop x 3-seed
  result (`per_crop.csv`, `summary.csv`, `paired_win_rate.csv`,
  `run_metadata.json`; jobs 47406/47453).
- `qualitative_panel.png` — A06/A08 raw / real Hamamatsu / stock decoder /
  alt decoder comparison, seed 0.

**Full narrative:** `tickets/P1-15_p1_11_alternate_decoder.md`,
`tickets/PHASE1-TICKETS.md` P1-15.
