# P3-06 — SDXL Transfer of P1-10's Source-Conditioned Architecture

**What this tests:** by the time this started, P1-10's source-conditioned
architecture had become the best SD1.5 result, so this transfers *that*
instead of the original ladder — a fresh, jointly-trained 6-channel
ControlNet on SDXL (dual text encoders, fp32 VAE, `added_cond_kwargs` for
both UNet and ControlNet). Not the same mechanism as P3-04/P3-05. Full
mechanism: Pipeline Anatomy §12.

**Status:** ❌ DONE — the mechanism transfers cleanly (ablation confirms
the ControlNet branch is genuinely used), but SDXL underperforms SD1.5 on
the actual held-out result, at strength 0.50/50-step DDIM (same operating
point both sides):

| | SD1.5 P1-10 | SDXL P3-06 |
|---|---|---|
| ALL SSIM | **0.4485** | 0.3920 |
| ALL_excl_outliers SSIM | **0.4695** | 0.4120 |
| Recovery Δlab (pooled) | **+2.89** | +1.22 |
| A06/A08/A09/A13/A16 SSIM | 0.307/0.482/0.418/0.458/0.497 | 0.257/0.410/0.372/0.404/0.441 |

Every slide's SSIM is lower on SDXL, and colour recovery — while positive
on every slide, no sign flip — is about 60% weaker than SD1.5's. More
encouraging than P3-04 (colour recovery never goes negative here), but
still not an improvement over SD1.5 at this operating point.

**Files:** `eval/full_heldout/eval_manifest.csv`,
`eval/full_heldout_summary/summary.csv` (table above), `per_crop.csv`.

**Full narrative:** `docs/results/RESULTS_SUMMARY.md` ("Update
(2026-08-25): P3-06") and `tickets/PHASE3-TICKETS.md` P3-06.
