# P3-07 — P3-06 Retrained at Native 1024×1024

**What this tests:** same P3-06 architecture, same ≤50-pair budget
(re-extracted at 1024px), same 4000 training steps — only the resolution
changed, testing whether P3-06's SSIM deficit vs. SD1.5 was a
training-resolution artefact.

**Status:** 🔄 REOPENED — a genuine trade-off, structure improves,
colour recovery flips negative, under active diagnosis. Same
strength=0.50/50-step DDIM operating point as P3-06:

| | SD1.5 P1-10 | SDXL P3-06 (512) | SDXL P3-07 (1024) |
|---|---|---|---|
| ALL SSIM | 0.4485 | 0.3920 | **0.4313** |
| A06 SSIM (outlier) | 0.3067 | 0.2567 | **0.3128** — exceeds SD1.5 |
| ALL recovery Δlab | +2.89 | +1.22 | **−5.60** (negative on every slide) |

Native resolution confirms the structural hypothesis — every slide's SSIM
improves over P3-06. But colour recovery flips to negative on every single
slide, not a pooled-outlier artefact. The source-conditioning ablation still
passes decisively at 1024px (correct 24.32 / shuffled 34.39 / zero 62.13
LAB), so this isn't a broken adapter.

**⚠️ D1 diagnostic now complete — not yet folded into `RESULTS_SUMMARY.md`
or the published Scorecard, both of which still say "running."** D1 tested
whether the −5.60 figure was a resolution-mismatched-baseline artefact
(comparing 1024px model output against a 512px-computed baseline). Result,
read directly from `eval/d1_verify/d1_summary.csv`:

| Scope | Recovery Δlab (1024-native baseline) | Recovery Δlab (as originally reported, 512 baseline) |
|---|---|---|
| ALL | −5.25 | −5.60 |
| A06 | −3.91 | −3.48 |
| A08 | −5.47 | −5.54 |
| A09 | −6.76 | −6.64 |
| A13 | −4.74 | −4.81 |
| A16 | −4.92 | −4.89 |

The two are close on every slide — **this rules out H5 (resolution-mismatched
baseline)** as the explanation. The negative colour recovery appears to be
real frozen-SDXL-base behaviour at native resolution, not a scoring
artefact. This still needs a proper write-up pass before being cited as
final — flagged to the user as a direct follow-up.

**D2 (colour-LoRA-disabled diagnostic, the other half of the diagnostic
plan) is still running** as of this reorg — its manifest is present but
empty on the cluster, not yet worth pulling.

**Files:** `eval/full_heldout/eval_manifest.csv`,
`eval/full_heldout_summary/summary.csv` (main table), `per_crop.csv`;
`eval/d1_verify/d1_summary.csv`, `d1_spot_check.csv` (D1 diagnostic table).

**Full narrative:** `docs/results/RESULTS_SUMMARY.md` ("Update
(2026-08-27): P3-07") and `tickets/PHASE3-TICKETS.md` P3-07 (D1/D2
diagnostic plan).
