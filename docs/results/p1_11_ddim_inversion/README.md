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

---

## H2A follow-on + fraction-sweep diagnostic (2026-09-14/26)

**H2A training/inference/scoring complete (2026-09-15):** same
architecture, correct-direction checkpoint (`lora/h2a_cond_r8/best`).
Structurally this project's strongest H2A performer (SSIM 0.5477 pooled),
but the original `f=1.00` (full DDIM-inversion) run showed a large,
uniform colour-recovery regression on every slide (pooled recovery
Δlab -15.64) — worse than doing nothing colour-wise.

**Diagnosed, not just accepted (2026-09-26):** full inversion discards the
original pixel signal before redenoising, letting the H2A colour-LoRA's
bias run unconstrained (confirmed visually — see
`qualitative_h2a_fraction_fix.png`: the f=1.00 output is *more*
magenta/saturated than even the raw Hamamatsu source, moving away from
Aperio's paler target, not toward it). Code independently reviewed for
bugs (metric symmetry, baseline-file reuse validity, fraction-handling
correctness) — none found; this is a genuine model/setting interaction,
not a scoring artefact.

**Fixed via a fraction sweep, confirmed at full scale:**

| | f=1.00 (original) | f=0.25 (fixed) |
|---|---|---|
| SSIM (ALL) | 0.5477 | **0.5553** |
| recovery Δlab (ALL) | -15.64 | **-6.19** |
| recovery Δlab (excl. A06) | -16.05 | **-6.06** |

Every slide improved by roughly half to two-thirds; SSIM held/improved
slightly too — no structure/colour trade-off. Recovery is still net
negative, so this does not reverse P2-12's "no significant clinical-utility
benefit" verdict, but the -15.64 figure should no longer be cited as this
architecture's H2A ceiling.

**Files:** `eval/h2a_full_f025/eval_summary_final.csv` (recovery_delta
per slide), `summary.csv`; `qualitative_h2a_fraction_fix.png` (A08/A16,
raw Hamamatsu / real Aperio / f=1.00 / f=0.25 side by side).

**Full narrative:** `tickets/PHASE1-TICKETS.md` P1-11's diagnostic
follow-up section, `tickets/P2-12_atypia_classifier_evaluation_hardening.md`,
`docs/results/RESULTS_SUMMARY.md`.
