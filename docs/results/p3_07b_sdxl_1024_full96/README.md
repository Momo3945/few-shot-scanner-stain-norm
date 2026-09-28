# P3-07b — Supplementary >50-Pair Follow-Up to P3-07

**What this tests:** identical architecture, hyperparameters, and 4000-step
budget to P3-07 (native-1024 SDXL transfer of P1-10's source-conditioned
colour LoRA + fresh 6-channel ControlNet) — the ONLY variable that changes
is training-set size: all 96 non-overlapping A03/H03 1024 pairs (measured
in job 46384) instead of P3-07's capped 50. `pairs/train_1024`'s 50 are a
deterministic subset of these 96 (same `--seed 0` round-robin selection,
just uncapped), so this is a strict data superset, not a different sample.

**⚠️ Scope note — read before citing this anywhere:** `docs/proposal_draft(6).tex`'s
H1/RQ1 is a literal, capped hypothesis ("≤50 coordinate-corresponding
A03/H03 crop pairs are sufficient... from ONE slide pair"). **P3-07b is a
supplementary >50-pair follow-up, explicitly NOT evidence for or against
the formal ≤50-pair H1 claim.** P3-07 (≤50 pairs, recovery Δlab −5.60)
remains the number that answers H1 as literally stated. P3-07b exists only
to test H3 (was P3-07's regression an insufficient-data problem at native
resolution, separate from the few-shot hypothesis itself) — report it
alongside P3-07, never in place of it.

**Status:** ✅ COMPLETE (2026-08-30). Colour recovery flips positive on
every single held-out slide.

## Motivation

P3-07 (≤50 pairs) found native 1024 resolution closes most of SDXL's
structural gap vs SD1.5, but colour recovery flips negative on every slide
(ALL recovery Δlab −5.60). Five diagnostics (D1 baseline-artefact check, D2
colour-LoRA-disabled comparison, D3 RGB-vs-Canny conditioning split, D4
`controlnet_conditioning_scale` sweep, D5 colour-LoRA-scale sweep — see
`tickets/PHASE3-TICKETS.md` P3-07) ruled out a metric artefact, a
mis-trained LoRA, a fixable conditioning-channel imbalance, and every
inference-only knob available on the ≤50-pair checkpoint. The remaining
hypothesis (H3): native-1024 crops are a harder/more heterogeneous
colour-learning problem than 512, and ≤50 paired crops are insufficient
data to learn it at that resolution.

## Result

| | P3-07 (≤50 pairs) | **P3-07b (96 pairs, supplementary)** |
|---|---|---|
| ALL SSIM | 0.4313 | **0.4416** |
| ALL_excl_outliers SSIM | 0.4485 | **0.4591** |
| ALL recovery Δlab | **−5.60** | **+1.74** |
| ALL_excl_outliers recovery Δlab | −5.43 | +1.62 |

Per-slide recovery Δlab (source: `eval/full_heldout_summary/eval_summary_final.csv`):

| Slide | P3-07 (≤50) | **P3-07b (96)** |
|---|---|---|
| A06 (outlier) | −3.48 | **+2.55** |
| A08 | −5.54 | **+1.82** |
| A09 | −6.64 | **+1.31** |
| A13 | −4.81 | **+0.81** |
| A16 | −4.89 | **+2.00** |

Every slide flips sign, not a pooled-outlier artefact. SSIM improves
slightly too (not a structure-for-colour tradeoff). Only variable changed
vs P3-07 is training-pair count — strong support for H3.

**Ablation control passed** (`eval/ablation_summary/summary.csv`):
`correct` LAB 25.25 decisively beats `shuffled` 35.94 and `zero` 57.49 on
every pooled metric; `correct` wins the paired SSIM win-rate 5/8 crops vs
`zero`'s 3/8 (`shuffled` 0/8) — ControlNet is genuinely used on this
checkpoint, not ignored.

**Files:** `eval/full_heldout/eval_manifest.csv`;
`eval/full_heldout_summary/{eval_summary_final.csv,summary.csv,per_crop.csv}`
(main tables); `eval/ablation_summary/{summary.csv,per_crop.csv,paired_win_rate.csv}`
(source-conditioning control).

**Checkpoint:** `lora/a2h_cond_r8_sdxl_1024_full96/{final,best}` on the
cluster (not backed up locally — checkpoints aren't versioned in this repo,
only eval outputs).

**Full narrative:** `tickets/PHASE3-TICKETS.md` P3-07 (P3-07b section) —
extraction/training/ablation/inference job IDs, D1-D5 diagnostic history,
and the full cross-project comparison in `docs/results/RESULTS_SUMMARY.md`.
