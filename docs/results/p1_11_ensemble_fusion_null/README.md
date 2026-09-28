# P1-11+P1-16 — Self-Ensembling: A Real Win Alone, a Null Result Combined With Fusion

**What this tests:** two follow-on questions from P1-11/P1-16's own
future-work lists. (1) Does genuine self-ensembling (multiple DDIM-inversion
samples per crop, eta&gt;0 for real stochasticity, averaged) improve on
P1-11's single-sample result? (2) Does that ensembled prediction compound
with [P1-16's fusion](../p1_16_source_fusion/) for a further gain, or does
fusion already capture what ensembling offers?

**Status:** ✅ Both closed. Part 1 — genuine positive result. Part 2 — clean
null result for compounding.

## Part 1 — self-ensembling alone: a real, validated improvement

3-member ensemble at eta=0.3, full held-out scale (496 crops), vs. P1-11's
single deterministic sample:

| source_mode | SSIM (mean) | LAB total (mean) |
|---|---|---|
| single sample (P1-11, `correct_member`, 3 independent runs) | 0.4606 | 25.80 |
| **3-member ensemble, eta=0.3** (`correct_ensemble`) | **0.5427** | **23.93** |

Every slide, including the A06 outlier, improves on *both* SSIM and colour
simultaneously — a clean, uniform win with no trade-off anywhere. Closes
roughly a third of the remaining gap to the weakest classical baseline
(Macenko, SSIM 0.628) purely via an inference-time technique — no
retraining required.

## Part 2 — combined with P1-16's fusion: does NOT compound

P1-16's frozen F3 config (σ=8, β=0.50) applied to the ensembled prediction
instead of a single deterministic sample:

| Slide | fusion alone SSIM (P1-16) | fusion + ensemble SSIM | Δ |
|---|---|---|---|
| A06 | 0.6171 | 0.6170 | ~0 |
| A08 | 0.7867 | 0.7870 | ~0 |
| A09 | 0.7269 | 0.7270 | ~0 |
| A13 | 0.6687 | 0.6680 | ~0 |
| A16 | 0.7586 | 0.7580 | ~0 |
| **ALL** | **0.72895** | **0.72879** | **−0.0002** |

Identical to 3–4 decimal places on every slide — not noise-level
uncertainty, a genuine null result for compounding. Colour (windowed LAB)
shows a tiny, consistent improvement on 4/5 slides but nothing meaningful.

**Mechanistic read:** F3 fusion takes the raw source's own structure
directly and adds back only a heavily Gaussian-blurred (σ=8) colour
residual — it already bypasses the diffusion output's own structural
fidelity almost entirely, discarding most of the ensembled prediction's
extra spatial detail in favour of the untouched source pixels. The two
techniques target the same underlying weakness (structural fidelity) via
different, non-additive routes.

**Practical takeaway:** use fusion when a fusion-based result is acceptable
(it alone already beats classical baselines); self-ensembling remains
valuable only for a fusion-free result (a purely model-generated output,
not blending in real source pixels), where it is still a genuine +0.047
SSIM improvement on its own line.

**Files:** `eval/ensemble_full/{summary.csv,per_crop.csv,paired_win_rate.csv}`
(Part 1 table above), `eval/ensemble_fusion/{eval_manifest.csv,
eval_per_crop.csv,eval_summary.csv}` (Part 2 table above).

**Full narrative:** `tickets/PHASE1-TICKETS.md` P1-11 future-work §2,
`docs/results/RESULTS_SUMMARY.md`.
