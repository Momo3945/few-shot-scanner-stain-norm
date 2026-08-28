# P1-11 — DDIM-Inversion Inference Path

**What this tests:** replaces P1-10's random-noise img2img initialisation
with DDIM inversion — running the trained model's own noise-prediction
backwards from the real source, giving a deterministic, source-specific
starting latent instead of an arbitrary one. Same trained checkpoint as
P1-10, inference-only, no retraining. Full mechanism: Pipeline Anatomy §06.

**Status:** ✅ DONE — clean, decisive positive result, the project's
strongest joint colour+structure SD1.5 result. Full 496-crop held-out:

| Scope | SSIM: old → new | Windowed LAB: old → new | Recovery Δlab: old → new |
|---|---|---|---|
| ALL | 0.4485 → **0.4960** | 31.60 → **26.01** | +2.89 → **+8.74** |
| ALL excl. A06 | — | — | +7.43 |
| A06 (outlier) | 0.3067 → **0.3732** | — | +21.72 |

Every held-out slide improves on both structure and colour simultaneously —
first configuration in the project to do that. Still capped by the same
0.5393 VAE-only floor P1-10 established, since DDIM inversion never touches
the decoder.

**Files:** `eval/full_heldout/eval_manifest.csv`, `run_metadata.json`;
`eval/full_heldout_summary/summary.csv` (headline numbers above),
`per_crop.csv`.

**Full narrative:** `docs/results/RESULTS_SUMMARY.md` ("Update
(2026-08-22): P1-11") and `tickets/PHASE1-TICKETS.md` P1-11 /
`tickets/Claude Code Task_ Add DDIM-Inversion Inference for P1-10(2).md`.
