# Results digest — Phase 1 ablation ladder (A0–A5, complete)

Pulled from the cluster on 2026-08-09/10. Raw CSVs for each run live alongside this
file in `docs/results/<tag>/` (`eval_manifest.csv`, `eval_per_crop.csv`,
`eval_summary.csv` — the last is the one summarised below). This file exists so
the final report has a stable, local, non-cluster-dependent source for numbers
and interesting findings — regenerate it by re-pulling from
`/datasets/mhoosen/stain-norm/eval/<tag>/` if a run is repeated.

**Status:** A0 through A5 are all complete and final — the entire Phase 1
ablation ladder is done, ready for the Phase 1 decision gate.

All metrics: LAB-histogram Wasserstein distance (`lab_total`, lower = closer to
real target-scanner ground truth), SSIM (higher = more structure preserved),
`recovery_delta_lab` = baseline LAB − model LAB (positive = the model moved the
output closer to the real Hamamatsu ground truth than doing nothing at all).

## Qualitative comparison — same crop, all six rungs side by side

Same coordinates run through every completed rung (A0/A1/A2/A3/A4/A5, strength
0.30), plus the raw Aperio input and the real registered Hamamatsu ground
truth, so the CSV numbers above can be checked against what the outputs
actually look like. Full-resolution source images: `docs/results/qualitative/<tag>/`.

**A08 (typical slide, LAB baseline 25.27):**
![A08 ablation comparison](qualitative/comparison_a08_typical.png)

Colour shift toward the Hamamatsu reference's darker, more saturated purple
nuclei is visible from A2 onward (colour LoRA active); A0/A1 stay close to the
paler Aperio input. A4 and A5 look visually close to A3 on this slide,
consistent with both scoring slightly negative recovery delta here (A4 -0.04,
A5 -1.27 @0.30) — A5's histopathology prior doesn't help (and slightly hurts)
on typical slides. Tissue structure (nuclear boundaries, stromal fibres) holds
up across all six.

**A06 (confirmed colour-gap outlier, LAB baseline 94.84), strength 0.30:**
![A06 ablation comparison](qualitative/comparison_a06_outlier.png)

This is the visual explanation for why A06's recovery deltas are small in
absolute terms even where positive: the real Hamamatsu reference is
*dramatically* more saturated than the Aperio source (a gap roughly 4× the
typical slide), and none of A0–A5 come close to closing it at strength 0.30 —
the colour shift each rung achieves is real (matches the sign of the recovery
deltas above) but tiny relative to the size of the gap.

**A06 at strength 0.50 — the headline result:**
![A06 strength 0.50 comparison](qualitative/comparison_a06_strength050.png)

A5's recovery delta here is **+16.02** — the best result anywhere in the
entire ablation ladder, more than double A4's +7.31 at the same strength (see
the A5 section below for the full comparison). Worth being honest about the
visual: the *quantitative* gain is real and verified in `eval_per_crop.csv`,
but the difference between A4 and A5 is fairly subtle to the eye in this
particular crop — LAB Wasserstein is a holistic per-crop histogram statistic,
and this tile is dominated by pale background/stroma rather than densely
stained nuclei, so a real measured shift doesn't always look dramatic in a
single image. Don't oversell the picture; trust the numbers.

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
| A3 | ControlNet + colour LoRA (r8) | **+1.32** | 0.386 | +0.45 | 0.259 | +1.90 | +1.81 | +4.97 | +1.16 |
| A4 | + LCM-LoRA (8-step) | −0.13 | **0.398** | −0.22 | **0.272** | −0.04 | +0.76 | +3.24 | −0.39 |
| A5 | + histopathology warm-start LoRA | −0.12 | 0.389 | **+2.38** | 0.265 | −1.27 | +2.38 | +3.26 | −1.51 |

A5's A06 lead over A4 grows sharply with strength — see the dedicated A5
section below for the full per-strength picture (it's the most important part
of this table, not fully visible at 0.30 alone).

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
  strength at every rung (structure/colour tradeoff, as expected).
  **Correction (2026-08-10, superseded by the P1-09 follow-up below): 0.30 is
  NOT the best strength** — this was only true within the originally-tested
  0.30-0.50 range. A dedicated follow-up (`tickets/PHASE1-TICKETS.md` P1-09)
  found strength 0.20 dominates 0.30 on nearly every axis for A3, and that A4/A5
  (LCM) behave completely differently — see the dedicated section below, "Post-gate
  follow-up." Don't cite "0.30 is optimal" in the write-up; the real picture is
  more interesting and is documented properly further down.

- **A06 must always be reported separately, never folded into a pooled `ALL`
  number** — its robust z-score sits at 23–39 across every run (threshold is
  usually ~3.5), so it would dominate or distort any single pooled statistic.
  This is why every table here (and every `eval_summary.csv`) carries
  `ALL_excl_outliers` alongside `ALL`.

### A4 / H4-RQ3: few-step LCM vs 50-step DDIM — a nuanced, not clean-pass, result

**Step-quantisation finding (from smoke testing, fixed before the full run):**
at `num_inference_steps=4` (the literal "four-step LCM" wording in the
proposal), diffusers' img2img `get_timesteps()` computes
`init_timestep = int(num_inference_steps * strength)`. For our usual strength
sweep this collapses strengths 0.30 and 0.40 to the **same single real
denoising step** (`int(4*0.30)=int(4*0.40)=1`) — confirmed empirically as
byte-identical per-crop output on every single row of the smoke test's
`eval_per_crop.csv`, not just similar. Not a code bug — an inherent interaction
between img2img's strength-based partial denoising and very low step counts.
Fixed by running the full A4 evaluation at 8 steps instead (still within
diffusers' documented LCM range of 4–8, still far faster than 50-step DDIM) —
properly differentiates 0.30/0.40/0.50 into 2/3/4 real steps.

**Full-run result (job 39630/39784, 8-step LCM vs A3's 50-step DDIM, same
colour-LoRA + ControlNet config, only the sampler changed):**

- **Structure (SSIM) is preserved, even marginally improved** — 0.386→0.398
  (ALL), 0.259→0.272 (A06), higher on every scope at strength 0.30. LCM
  acceleration is not damaging structural fidelity here.
- **Colour recovery is not a clean pass** — `recovery_delta_lab` goes
  *negative* at strength 0.30 for ALL (-0.13), A06 (-0.22), A08 (-0.04), and
  A16 (-0.39); only A09 (+0.76) and A13 (+3.24) stay clearly positive. LCM
  acceleration trades away some of the colour LoRA's gain even though
  structure holds up.
- **Strongly strength-dependent, non-monotonic**: at strength 0.50, A06 jumps
  to **+7.31** — the best A06 recovery anywhere in the entire A0–A4 ladder —
  while A08/A13/A16 fall further behind (-3.52/-1.09/-2.91) at the same
  strength. Likely the same few-step quantisation sensitivity: 8 steps ×
  strength 0.50 is still only 4 real denoising steps, more variance-prone than
  50-step DDIM's much finer resolution.
- **The proposal's SSIM ≥ 0.85 pass/fail threshold does not apply literally
  here** — no rung in the entire A0–A4 ladder ever exceeds ~0.45 SSIM under
  this project's actual metric computation, so 0.85 isn't a calibrated bar for
  these numbers. The meaningful comparison is A4 vs A3 (relative, same
  adapters, only the sampler changed), not A4 vs an absolute constant.
- **Bottom line for H4/RQ3**: structurally compatible; colour recovery is
  mixed/strength-dependent rather than a clean match at 8 steps. This reads as
  a genuine, useful *negative-leaning* result rather than grounds to invoke the
  pre-committed 20-step DDIM fallback (that fallback targets structural/artefact
  failure, which did not occur — SSIM stayed flat-to-better throughout).

Worth including in the write-up both as a concrete illustration of why H4/RQ3
needs empirical verification rather than taking "N-step LCM" at face value, and
as an example of a result that doesn't cleanly confirm the hypothesis — still a
useful, reportable finding. Full details: `tickets/PHASE1-TICKETS.md` P1-05.

### A5 / sec:hist_lora: histopathology warm-start — real, but slide-dependent

**Scope note:** A5 stacks the already-trained, already-frozen `hist_r32` and
`a2h_r8` checkpoints at inference (no new training job) — see
`tickets/PHASE1-TICKETS.md` P1-07 for how the proposal's training-order wording
was resolved. The primary comparison per the proposal is A5 vs A4 (same
colour-LoRA + ControlNet + LCM config, only the histopathology LoRA added).

**A5 vs A4, recovery Δlab, per strength:**

| scope | A4 @0.30 | A5 @0.30 | A4 @0.40 | A5 @0.40 | A4 @0.50 | A5 @0.50 |
|---|---|---|---|---|---|---|
| ALL | −0.13 | −0.12 | −1.65 | −0.94 | −1.21 | **+0.42** |
| **A06** | −0.22 | **+2.38** | +1.55 | **+7.35** | +7.31 | **+16.02** |
| A08 | −0.04 | −1.27 | −2.84 | −3.40 | −3.52 | −2.43 |
| A16 | −0.39 | −1.51 | −2.82 | −3.44 | −2.91 | −3.01 |

- **On A06, the histopathology prior is dramatically effective** — the gap
  over A4 *grows* with strength: +2.60 (@0.30) → +5.80 (@0.40) → **+8.71**
  (@0.50). A5's A06 result at strength 0.50 (+16.02) is the single best result
  anywhere in the entire A0–A5 ladder — more than double A4's already-best A06
  score. In absolute terms, raw LAB Wasserstein for A06 drops from the 94.84
  baseline to 78.82 — closing ~17% of the original colour gap, the largest
  closure of any rung.
- **SSIM is not the trade-off** — A5's A06 SSIM (0.199 @0.50) sits right next
  to A4's (~0.20) — the histopathology prior isn't buying colour recovery at
  the cost of structure.
- **On typical slides (A08, A16), A5 is consistently worse than A4** — the
  histopathology prior appears to pull colour in a direction that helps the
  extreme outlier but mildly hurts already-well-behaved slides. Pooled `ALL`
  is roughly a wash at 0.30 and modestly favours A5 at 0.40/0.50, once A06's
  large gains are averaged in with the small typical-slide losses.
- **Bottom line**: this is a genuine positive result for the specific research
  question `sec:hist_lora` asks (does a histopathology-domain prior help
  *generalisation to the hardest case*?), even though it is not a uniform
  win in the pooled-average sense. Report both framings — collapsing to a
  single "A5 beats A4" or "A5 doesn't help" verdict would misrepresent the
  slide-dependent reality. This also directly informs P3-05 (whether A5 gets
  transferred to SDXL): worth doing given how much it helps the hardest case,
  even though it isn't a universal improvement.

Full details, evidence, and the qualitative caveat about the strength-0.50
image not looking as dramatic as the numbers: `tickets/PHASE1-TICKETS.md` P1-07.

## Post-gate follow-up: extended strength sweep (A3/A4/A5) — supplementary

Run after the Phase 1 decision gate closed (doesn't reopen it), motivated by
two patterns spotted while mining the completed CSVs: A06's recovery-vs-strength
curve looked like it was still accelerating at 0.50, and A5's typical-slide
deficit vs A4 looked like it might be shrinking with strength. Both needed real
data beyond the original 0.30-0.50 range to confirm or refute — extrapolating
a 3-point trend isn't safe on its own. Full methodology, safety fix (output-dir
collision risk), and job list: `tickets/PHASE1-TICKETS.md` P1-09.

**Finding 1 — LCM step-count quantisation recurs at the 0.5/0.6 boundary.**
Confirmed byte-identical per-crop output between strength 0.50 and 0.60 on A5
(`int(8*0.50)=int(8*0.60)=4` real steps) — the same mechanism as the earlier
0.30/0.40 collision (P1-05). At 8-step LCM, only ~5-6 strength values map to
genuinely distinct step counts; "strength" is a coarse step-count proxy for
A4/A5, not the smooth continuous dial it is for A3's 50-step DDIM.

**Finding 2 — A3: strength 0.20 dominates 0.30, not a wash.**

| A3 | ALL Δlab | ALL SSIM | A06 Δlab | A08 Δlab | A09 Δlab | A13 Δlab | A16 Δlab |
|---|---|---|---|---|---|---|---|
| 0.20 | **+1.59** | **0.425** | +0.41 | **+2.45** | **+1.88** | **+5.29** | **+1.46** |
| 0.30 | +1.32 | 0.386 | **+0.45** | +1.90 | +1.81 | +4.97 | +1.16 |

0.20 wins on every metric except A06 (negligible -0.04 gap). The "0.30 is
optimal" claim earlier in this document was never tested against anything
lower — 0.20 is just the new best point found, and the true floor is still open.

**Finding 3 — A4/A5: behaviour vs strength is non-monotonic, and A4/A5 diverge
sharply from each other at the extremes.**

| Real LCM steps | strength | A4 ALL Δlab | A4 SSIM | A5 ALL Δlab | A5 SSIM | A06 Δlab (A4 / A5) |
|---|---|---|---|---|---|---|
| 1 | 0.20 | **+1.53** | **0.459** | **+1.93** | **0.454** | +0.42 / +1.19 |
| 2 | 0.30 | −0.13 | 0.389 | −0.12 | — | −0.22 / +2.38 |
| 3 | 0.40 | −1.65 | — | −0.94 | — | +1.55 / +7.35 |
| 4 | 0.50 | −1.21 | 0.312 | +0.42 | 0.299 | +7.31 / +16.02 |
| 5 | 0.70 | **+2.25** | 0.271 | −3.08 | 0.268 | **+19.83** / **+24.43** |

At 1 step: a broad, uniform win — every slide improves, SSIM is the highest
anywhere in the ladder. At 2-4 steps: a worse valley on typical slides even as
A06 climbs steadily. At 5 steps: extreme divergence — **A06 hits its best
recovery anywhere in the entire project (A5: +24.43, closing over a quarter of
its raw 94.84 LAB gap)**, but every typical slide craters (A5 @0.70: A08 −6.97,
A09 −3.37, A13 −7.58, A16 −7.72 — much worse than A4's equivalent losses at the
same step count).

**This refutes the pre-experiment hypothesis** that A5's typical-slide deficit
vs A4 would keep shrinking or flip permanently positive at higher strength — it
does narrow/flip between strength 0.30 and 0.50, but reverses hard and gets
much worse than A4 by 5 steps (0.70). A good illustration of why the
confirmatory full run mattered rather than trusting a 3-point extrapolation.

**Practical takeaway — no single best strength, it depends on the goal:**
- **General-purpose robustness across all slides**: 1-step LCM (strength
  ≈0.20-0.25) is the best operating point found anywhere in this project for
  both A4 and A5 — beats every strength in the original official ladder.
- **Maximum single-slide (hardest-case) recovery**: 5-step A5 (strength 0.70)
  is dramatically the strongest single result in the whole project, at severe
  cost everywhere else. Frame this as a "rescue mode" option for the hardest
  cases, not a general default.

Raw data: `docs/results/{a3_ext_s02,a4_ext_s02,a5_ext_s02,a4_ext_s07,a5_ext_s07,a4_ext_s67,a5_ext_s67}/`.
Reproducible analysis script: `docs/results/analyze.py`.

## Local file index

```
docs/results/
├── baseline/            raw do-nothing comparison (P2-01)
├── a0/                  A0: frozen base only
├── a1/                  A1: + ControlNet-Canny
├── a2h_r4/               A2: + colour LoRA, rank 4
├── a2h_r8/               A2: + colour LoRA, rank 8 (used in A3/A4)
├── a3/                  A3: ControlNet + colour LoRA (rank 8)
├── a4/                  A4: + LCM-LoRA (8-step)
├── a5/                  A5: + histopathology warm-start LoRA (stacked at inference)
├── a3_ext_s02/           P1-09 follow-up: A3 at strength 0.20 (full 5-slide)
├── a4_ext_s02/           P1-09 follow-up: A4 at strength 0.20 = 1 real LCM step (full)
├── a5_ext_s02/           P1-09 follow-up: A5 at strength 0.20 = 1 real LCM step (full)
├── a4_ext_s07/           P1-09 follow-up: A4 at strength 0.70 = 5 real LCM steps (full)
├── a5_ext_s07/           P1-09 follow-up: A5 at strength 0.70 = 5 real LCM steps (full)
├── a4_ext_s67/           P1-09 follow-up: A4 0.6/0.7 smoke test (A06-only; 0.6 duplicates 0.5)
├── a5_ext_s67/           P1-09 follow-up: A5 0.6/0.7 smoke test (A06-only; 0.6 duplicates 0.5)
├── analyze.py             reproducible per-crop trend analysis behind the P1-09 experiment picks
└── qualitative/          side-by-side A0-A5 comparison composites (3 example crops)
```

Each folder: `eval_manifest.csv` (crop-level path bookkeeping), `eval_per_crop.csv`
(per-crop metrics, the source for every number above), `eval_summary.csv`
(per-slide/per-strength aggregates — what's quoted throughout this file).
