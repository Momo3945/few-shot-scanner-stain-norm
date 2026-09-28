# P3-07c — SDXL @1024, Overlap-Expanded Pair Count (144 pairs)

**What this tests:** the SDXL training-set-size trilogy, third and final
step. P3-06 (512px, ≤50 pairs) → P3-07 (1024px, ≤50 pairs, colour recovery
flips negative) → P3-07b (1024px, 96 non-overlapping pairs,
[`../p3_07b_sdxl_1024_full96/`](../p3_07b_sdxl_1024_full96/), colour recovery
flips back positive) → **P3-07c: does pushing pair count further via the
proposal's overlap fallback (`--overlap 0.5`, 144 pairs) close more of the
remaining gap to SD1.5's P1-10, or does correlated data plateau?**

**Status:** ✅ CLOSED, positive result (2026-09-27, jobs 58267/58465
training, 59736/59858/59738 ablation, 60645/60742/60749 full held-out
run+score+aggregate).

## Result — the training-size trend across all three SDXL@1024 runs

| | P3-07 (≤50 pairs) | P3-07b (96 pairs) | **P3-07c (144 pairs)** | SD1.5 P1-10 |
|---|---|---|---|---|
| ALL SSIM | 0.4313 | 0.4416 | **0.4457** | 0.4485 |
| ALL_excl_outliers SSIM | 0.4485 | 0.4591 | **0.4617** | 0.4695 |
| ALL recovery Δlab | −5.60 | +1.74 | **+1.85** | +2.89 |

Every slide stays positive, both SSIM and colour recovery improve further
over P3-07b — not a plateau or regression. On 3 of 5 slides (A06 +3.42,
A08 +1.86, A09 +1.47) P3-07c's own Δlab already exceeds SD1.5's P1-10,
though pooled it remains slightly behind.

**The marginal gain per pair is shrinking**: +0.11 pooled Δlab for +48
pairs (P3-07b→P3-07c), vs. P3-07b's own +7.34-unit jump for the prior +46
pairs (P3-07→P3-07b). This reinforces, rather than repeats, the P3-07b
data-starvation diagnosis — more data keeps helping, with diminishing
returns, not a hard ceiling.

**Ablation control passed** (`eval/ablation_summary/summary.csv`):
`correct` decisively beats `shuffled`/`zero` on every pooled metric —
ControlNet is genuinely used on this checkpoint, not ignored.

**⚠️ Scope note:** same status as P3-07b — this is a *supplementary*
>50-pair follow-up, explicitly **not** evidence for or against the formal
H1/RQ1 ≤50-pair few-shot claim. P3-07 (≤50 pairs) remains the number that
answers H1 as literally stated; report P3-07c alongside it, never in place
of it.

**Files:** `eval/ablation_summary/{summary.csv,per_crop.csv,paired_win_rate.csv}`
(source-conditioning control), `eval/full_heldout_summary/
{eval_summary_final.csv,summary.csv,per_crop.csv}` (main table above).

**Checkpoint:** `lora/a2h_cond_r8_sdxl_1024_overlap144/{final,best}` on the
cluster (not backed up locally — checkpoints aren't versioned in this repo).

**Full narrative:** `tickets/PHASE3-TICKETS.md` P3-07c (full ticket:
`tickets/P3-07c_sdxl_overlap_pair_expansion.md`),
`docs/results/RESULTS_SUMMARY.md`. The follow-on fusion result on this
checkpoint: [`../p3_08_sdxl_fusion/`](../p3_08_sdxl_fusion/).
