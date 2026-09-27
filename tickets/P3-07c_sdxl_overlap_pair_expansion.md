# P3-07c — SDXL @1024, overlap-expanded pair count (144 pairs)

**Status:** ✅ CLOSED, positive result (2026-09-27). Trained
`a2h_cond_r8_sdxl_1024_overlap144` (jobs 58267 smoke, 58465 real training)
on 144 overlap-generated A03/H03 1024 pairs, ran the mandatory
source-conditioning ablation (jobs 59736/59858/59738, scored job 60009),
then the full 5-slide × 3-seed held-out inference (job 60645) and
aggregation (jobs 60742/60749).

**Source:** `docs/proposal_draft(6).tex`'s H1 contingency fallback ("expand
overlapping crops from the same A03/H03 pair while reporting the minimum
viable pair count") — same citation basis as P3-07b. Direct follow-up to
`tickets/PHASE3-TICKETS.md` P3-07/P3-07b: does pushing the pair count
further via overlap close more of the remaining gap to SD1.5's P1-10, or
does correlated (overlapping) data plateau where P3-07b's genuinely
non-overlapping increase did not?

## Scope

Same discipline as P3-07b: A03/H03 only, one slide pair, no new slides,
same rank (8), same steps (4000), same resolution (1024), same direction
(A2H). **Supplementary — NOT evidence for or against the proposal's
formal ≤50-pair H1 claim** (P3-07 remains the number that answers H1 as
literally stated). Only the crop-generation mechanism changed: 144 pairs
via `extract_pairs.py --overlap 0.5` (measured locally: overlap 0.0/0.3
both cap at 96 non-overlapping candidates given the frame's ~1.5x-crop
size; 0.5 yields 144; 0.7 yields 216 — 144 chosen as a modest ~50% step
up from P3-07b's 96, not the largest available option).

## Implementation

Mirrored P3-07b's own file-for-file precedent (a hardcoded,
self-documenting slurm launcher per stage, "NOT H1 evidence" baked into
the script text itself):
- `src/data/extract_pairs.py --root data/mitos --crop 1024 --max-pairs 144
  --overlap 0.5 --seed 0 --skip-heldout --tissue-thresh 0.30` (local,
  matched the read-only measurement exactly: 144/144, no shortfall).
  Written to `pairs/train_1024_overlap144/` (288 files) +
  `pairs/train_1024_overlap144_manifest.csv` + `..._extract_config.json`,
  uploaded to the cluster.
- `src/train/train_p3_07c_lora_sdxl.py` / `slurm/train_p3_07c_lora_sdxl.slurm`
  — copies of `train_p3_07b_lora_sdxl.py`/`.slurm` with the docstring
  updated; architecture/hyperparameters byte-identical. New checkpoint
  tag `lora/a2h_cond_r8_sdxl_1024_overlap144/`, never touches
  `..._1024/` (P3-07) or `..._1024_full96/` (P3-07b).
- `slurm/score_p3_07c_full.slurm` / `slurm/aggregate_p3_07c_full.slurm` /
  `slurm/score_p3_07c_ablation.slurm` — thin clones of the P3-07b
  equivalents, paths updated only.
- No changes needed to `infer_colour_translation_sdxl.py`/`.slurm` —
  already fully generic over `CHECKPOINT_DIR`/`CROP`.

## Steps taken (with two corrections caught before submitting)

1. Smoke test (job 58267): passed cleanly, param counts identical to
   P3-07/P3-07b (11.6M LoRA + 7.6M ControlNet).
2. Full training (job 58460) hit bad node `mscluster59` (8th independent
   hit of the "device handle for GPU0" sub-signature, fail-fast caught it,
   no compute lost, added to CLAUDE.md). Resubmitted (job 58465):
   completed cleanly in ~1h49m, val_loss trending down 0.1177 → 0.1065.
3. **Correction caught before submitting:** the first proposed
   source-conditioning ablation command used the `PAIRS_DIR`+`OVERFIT_N`
   shape (correct for an *overfit* diagnostic checkpoint) — but this
   checkpoint is the real, fully-trained-on-144-pairs one. Corrected to
   the held-out-frames shape (`CROP=1024`, `LIMIT=4`, no `PAIRS_DIR`)
   that P3-07/P3-07b's own real-checkpoint ablations actually used.
4. Ablation (jobs 59736 correct / 59858 zero (resubmit after `mscluster62`,
   9th hit of the same sub-signature) / 59738 shuffled, scored job 60009)
   **passed decisively**: SSIM correct=0.318 vs shuffled=0.114 vs
   zero=0.109 (~2.8-2.9×), win-rate 16/16 crops.
5. **Correction caught before submitting:** the first proposed
   ablation-scoring command was an ad hoc `sbatch --wrap='...'` one-liner
   missing conda activation and using relative paths — would have failed
   silently or errored. Replaced with a proper cloned
   `score_p3_07c_ablation.slurm` (mirrors `score_p3_07b_ablation.slurm`)
   before submitting.
6. Full 5-slide × 3-seed held-out inference (job 60645, `--time=10:00:00`
   requested up front per P3-07b's own lesson about under-provisioning):
   completed cleanly, 1485 crops / 495 locations × 3 seeds — exact
   expected shape.
7. Scoring (job 60742) + aggregation (job 60749): clean, no errors.

## Result (2026-09-27)

| | P3-07 (≤50 pairs) | P3-07b (96 pairs) | **P3-07c (144 pairs)** | SD1.5 P1-10 |
|---|---|---|---|---|
| ALL SSIM | 0.4313 | 0.4416 | **0.4457** | 0.4485 |
| ALL_excl_outliers SSIM | 0.4485 | 0.4591 | **0.4617** | 0.4695 |
| ALL recovery Δlab | −5.60 | +1.74 | **+1.85** | +2.89 |
| ALL_excl_outliers recovery Δlab | −5.43 | +1.62 | **+1.62** | — |
| A06 Δlab | −3.48 | +2.55 | **+3.42** | +2.70 |
| A08 Δlab | −5.54 | +1.82 | **+1.86** | +1.74 |
| A09 Δlab | −6.64 | +1.31 | **+1.47** | +1.03 |
| A13 Δlab | −4.81 | +0.81 | **+0.55** | +2.31 |
| A16 Δlab | −4.89 | +2.00 | **+1.97** | +1.58 |

**Every slide stays positive — not a plateau or regression from P3-07b.**
Going 96 → 144 pairs (a further +50% via overlap, correlated rather than
fully independent crops) gave a modest further improvement on both axes:
+0.11 pooled Δlab, +0.0041 pooled SSIM, +0.0026 excl-outliers SSIM. This is
a much smaller step than 50 → 96's sign-flipping jump (as expected — that
transition fixed a genuine data-starvation failure; this one is refining
an already-working configuration with more, but increasingly redundant,
data from the same single frame set). SDXL's SSIM is now within 0.003 of
SD1.5's P1-10 (0.4457 vs 0.4485), and on 3 of 5 slides (A06, A08, A09)
P3-07c's *own* Δlab already exceeds P1-10's — SDXL is not just "recovering
positively," it is competitive with, and locally exceeds, this project's
best SD1.5 configuration on individual slides, even though it remains
slightly behind pooled.

**A06 in particular keeps improving with more data** (P3-07b +2.55 →
P3-07c +3.42, now above SD1.5's own +2.70) while A13 dipped slightly
(+0.81 → +0.55, still positive, still the weakest slide in this family
across all three overlap-pair-count runs) — consistent with this
project's standing pattern that different slides respond differently to
any single global setting, not a new finding on its own.

**Interpretation:** the data-starvation diagnosis from P3-07b is
reinforced, not merely repeated — more data (even correlated/overlapping
data from the same single slide pair) continues to help, with diminishing
but still real returns. This is not evidence that overlap-based expansion
alone will eventually match or exceed SD1.5's pooled colour recovery
(+2.89) — the marginal gain per additional ~48 pairs is shrinking
(+1.74→+1.85, +0.11 for +48 pairs, vs the +7.34-unit jump for the prior
+46 pairs) — but it has not plateaued or reversed either.

## Non-goals / what this does not establish

- **Not H1 evidence** — same status as P3-07b, one step further along the
  same sanctioned fallback. P3-07 (≤50 pairs, −5.60) remains the number
  that answers the proposal's literal H1 claim.
- **Not a claim that SDXL now beats SD1.5** — P3-07c is close on SSIM and
  behind on pooled colour recovery; "competitive, and slide-locally
  ahead on 3/5 slides" is the accurate characterization, not "wins."
- Not a rank/step/resolution sweep — only pair count/generation mechanism
  changed, matching P3-07b's own discipline.
- A further step (216 pairs, `--overlap 0.7`) would be a cheap next
  increment if diminishing-returns curve-fitting is wanted, but is not
  run here — this ticket stops at the one pre-agreed "a bit more" step.
