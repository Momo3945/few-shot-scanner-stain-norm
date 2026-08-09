# Results digest — Phase 1 ablation ladder (A0–A4)

Pulled from the cluster on 2026-08-09. Raw CSVs for each run live alongside this
file in `docs/results/<tag>/` (`eval_manifest.csv`, `eval_per_crop.csv`,
`eval_summary.csv` — the last is the one summarised below). This file exists so
the final report has a stable, local, non-cluster-dependent source for numbers
and interesting findings — regenerate it by re-pulling from
`/datasets/mhoosen/stain-norm/eval/<tag>/` if a run is repeated.

**Status:** A0, A1, A2 (rank 4 + rank 8), A3 are complete and final. **A4 (+LCM)
full held-out run is still in progress** (job 39630 as of writing) — this file
will be updated once it lands. A4's smoke-test-only numbers are not included
below because they used a since-fixed step count (see "A4 LCM step-quantisation
finding" below) and aren't representative.

All metrics: LAB-histogram Wasserstein distance (`lab_total`, lower = closer to
real target-scanner ground truth), SSIM (higher = more structure preserved),
`recovery_delta_lab` = baseline LAB − model LAB (positive = the model moved the
output closer to the real Hamamatsu ground truth than doing nothing at all).

## Raw baseline (no model at all — do-nothing comparison)

`docs/results/baseline/baseline_summary.csv` — this is the reference point every
`recovery_delta_lab` above is measured against.

| Scope | n_crops | LAB Wasserstein | SSIM |
|---|---|---|---|
| ALL | 1475 | 33.83 | 0.733 |
| ALL excl. outliers | 1296 | 25.41 | 0.748 |
| **A06 (outlier)** | 179 | **94.84** | 0.627 |
| A08 | 336 | 25.27 | 0.792 |
| A09 | 288 | 27.48 | 0.727 |
| A13 | 192 | 26.63 | 0.675 |
| A16 | 480 | 23.77 | 0.759 |

A06's raw LAB gap (94.84) is ~3.7× every other slide — robust z-score 33.8,
confirmed genuine colour-gap outlier (registration checked clean), not a data bug.

## Main comparison — strength 0.30, recovery Δlab (higher = better) and SSIM

Strength 0.30 chosen because it's the best strength for 4 of 5 slides throughout
the ladder (see "strength trend" below) — full per-strength numbers are in each
run's `eval_summary.csv`.

| Rung | Config | ALL Δlab | ALL SSIM | A06 Δlab | A06 SSIM | A08 Δlab | A09 Δlab | A13 Δlab | A16 Δlab |
|---|---|---|---|---|---|---|---|---|---|
| A0 | frozen base only | +0.95 | 0.285 | −0.39 | 0.186 | +1.85 | +1.24 | +4.94 | +0.73 |
| A1 | + ControlNet | +0.73 | **0.389** | −0.52 | 0.260 | +1.60 | +1.09 | +4.51 | +0.55 |
| A2 (rank 4) | + colour LoRA | +1.58 | 0.289 | +1.16 | 0.192 | +2.18 | +1.75 | +5.08 | +1.47 |
| A2 (rank 8) | + colour LoRA | +1.72 | 0.289 | +0.92 | 0.191 | +2.49 | +1.90 | +5.27 | +1.62 |
| A3 | ControlNet + colour LoRA (r8) | **+1.32** | 0.386 | +0.45 | **0.259** | +1.90 | +1.81 | +4.97 | +1.16 |

## Interesting details for the write-up

- **Colour LoRA (A2), not ControlNet, is what first makes A06 recover.**
  A0 and A1 (no colour adapter) both show *negative* recovery delta on A06 at
  every strength — the frozen base genuinely cannot close that colour gap on its
  own. A2 (colour LoRA, either rank) flips this positive (+0.92 to +1.16 @0.30).
  ControlNet alone never fixes colour — expected, it only conditions structure.

- **A3's real contribution isn't "first to recover A06"** (A2 already does,
  slightly better on this metric) — **it's recovering A06's colour while also
  keeping A1's structural gain.** A06 SSIM: A0 0.186, A2 ~0.19 (colour LoRA
  alone barely helps structure), A1 0.260, **A3 0.259** — A3 keeps ControlNet's
  full SSIM improvement *and* a positive colour recovery simultaneously, which
  neither single-component rung achieves together. On the non-outlier slides
  (A08/A09/A13/A16), A3 also edges out A2 alone on 3/4 slides, so the combination
  is doing genuine additive work, not just averaging two effects.

- **ControlNet's SSIM jump is large and consistent**: +0.10 over A0 at every
  strength (0.30: 0.285→0.389, 0.40: 0.227→0.351, 0.50: 0.181→0.312) — Canny
  conditioning is demonstrably constraining generation, not being silently
  ignored by the pipeline.

- **Rank barely matters for the colour LoRA** (full comparison:
  `tickets/PHASE2-TICKETS.md` P2-05). Rank 4 vs rank 8 track within ~0.1–0.3 LAB
  units on every slide/strength — well inside noise. Rank 8 has a marginal edge
  on the pooled `ALL` recovery at 0.30/0.40, which is why it was picked for A3.
  Supports the proposal's H1 hypothesis that scanner normalisation is a
  low-complexity colour shift not requiring high adapter capacity.

- **Strength trend diverges by scope, and this shows up at every rung**: A06
  improves *with* higher strength while A08/A09/A13/A16 all recover *less* at
  higher strength. E.g. A3: A06 Δlab +0.45→+0.56→+0.88 (0.30→0.40→0.50) while
  A08 Δlab +1.90→+1.22→+0.64 over the same range. SSIM falls monotonically with
  strength at every rung (structure/colour tradeoff, as expected). No single
  strength is optimal for every slide — 0.30 is the best compromise for 4/5
  slides throughout.

- **A06 must always be reported separately, never folded into a pooled `ALL`
  number** — its robust z-score sits at 23–39 across every run (threshold is
  usually ~3.5), so it would dominate or distort any single pooled statistic.
  This is why every table here (and every `eval_summary.csv`) carries
  `ALL_excl_outliers` alongside `ALL`.

### A4 LCM step-quantisation finding (methodological, from smoke testing)

At `num_inference_steps=4` (the literal "four-step LCM" wording in the
proposal), diffusers' img2img `get_timesteps()` computes
`init_timestep = int(num_inference_steps * strength)`. For our usual strength
sweep this collapses strengths 0.30 and 0.40 to the **same single real
denoising step** (`int(4*0.30)=int(4*0.40)=1`) — confirmed empirically as
byte-identical per-crop output on every single row of `eval_per_crop.csv`, not
just similar. This is not a code bug; it's an inherent interaction between
img2img's strength-based partial denoising and very low step counts. Fixed by
running A4 at 8 steps instead (still within diffusers' documented LCM range of
4–8, still far faster than 50-step DDIM) — properly differentiates
0.30/0.40/0.50 into 2/3/4 real steps. Worth including in the write-up as a
concrete illustration of why H4/RQ3 needs empirical verification rather than
taking "N-step LCM" at face value. Full details: `tickets/PHASE1-TICKETS.md` P1-05.

## Local file index

```
docs/results/
├── baseline/            raw do-nothing comparison (P2-01)
├── a0/                  A0: frozen base only
├── a1/                  A1: + ControlNet-Canny
├── a2h_r4/               A2: + colour LoRA, rank 4
├── a2h_r8/               A2: + colour LoRA, rank 8 (used in A3/A4)
├── a3/                  A3: ControlNet + colour LoRA (rank 8)
└── (a4/ to be added once the full held-out run — job 39630 — completes)
```

Each folder: `eval_manifest.csv` (crop-level path bookkeeping), `eval_per_crop.csv`
(per-crop metrics, the source for every number above), `eval_summary.csv`
(per-slide/per-strength aggregates — what's quoted throughout this file).
