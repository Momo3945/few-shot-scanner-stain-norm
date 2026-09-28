# P3-06/P3-07 H2A — SDXL Extended to the Reverse Direction, and a Colour-Direction Asymmetry Diagnostic

**What this tests:** P1-10/P1-11's source-conditioned architecture and its
SDXL transfer (P3-06/P3-07) were originally trained A→H only. P2-12's own
clinical-utility audit flagged that every one of those results is the
opposite direction a real clinical-utility claim needs (the downstream
atypia classifier is Aperio-trained, so H→A is the direction that matters).
This extends the same architecture to H2A at both SDXL resolutions —
closing that gap, and surfacing a genuinely interesting secondary finding
about backbone colour bias along the way.

**Status:** ✅ Both training runs complete and scored (2026-09-25–27).
Diagnostic chase into *why* is an honest partial answer, not a closed one.

## Colour recovery, every A2H/H2A × backbone × resolution combination

| | SSIM | Recovery Δlab |
|---|---|---|
| SD1.5 A2H (P1-10) | 0.4485 | +2.89 |
| SD1.5 H2A (P1-11) | 0.5477 | **−15.64** |
| SDXL A2H, 512px (P3-06) | 0.3920 | +1.22 |
| **SDXL H2A, 512px** | 0.3555 | **+9.47** |
| SDXL A2H, 1024px, ≤50pr (P3-07) | 0.4313 | **−5.60** |
| **SDXL H2A, 1024px, ≤50pr** | 0.3865 | **+9.20** |

**Finding: SDXL's H2A colour recovery is robustly positive regardless of
resolution or training-pair count, while its A2H recovery is fragile and
resolution/data-dependent — the opposite asymmetry from SD1.5**, whose A2H
is the comfortable direction and whose H2A collapses catastrophically.
Every H2A slide is positive at both SDXL resolutions, no exceptions
(including the A06 outlier: +12.37 to +18.85). This also answers a question
P3-07's own negative A2H result raised: does the ≤50-pair data-scarcity
mechanism (P3-07b's diagnosis) also break H2A? No — P3-07 H2A at the
identical ≤50-pair/1024px budget is strongly positive, so that mechanism is
specific to the A2H direction, not a general property of native-resolution/
low-pair-count training.

## Mechanism chase — two diagnostics, an honest partial answer

**Hypothesis:** each frozen pretrained backbone carries its own inherent
colour bias from pretraining that sits closer to one scanner's palette,
making whichever direction matches it robust and whichever fights it
fragile.

- **First attempt** (reusing existing `zero`-mode ablation outputs, no new
  compute) **failed on a methodological confound**: every one of the six
  combinations' zero-mode colour leaned toward its own *source* domain,
  never target — revealed to be an img2img source-latent-persistence
  artefact (`zero` mode still runs a real img2img pass from the noised
  source latent; only the ControlNet's *extra* conditioning channel is
  zeroed), not a colour-free backbone default. A useful correction to how
  this project's `zero` ablation should be read (it validates ControlNet-
  branch usage, not backbone colour bias) — but not an answer to the
  mechanism question.
- **Second attempt: genuinely new inference — pure text-to-image
  generation** (colour LoRA loaded alone, no ControlNet, no source image at
  all, full random noise, 24 samples/combo) removes that confound entirely.
  One representative sample per combination: [`images/`](images/).

  **Result: 5 of 6 combinations are directionally consistent** with the
  hypothesis — SD1.5 H2A (model's own default leans Hamamatsu, its *wrong*
  target) pairs with the single worst real result (−15.64); SDXL H2A@512
  (default correctly leans Aperio, its target) pairs with the single best
  real result (+9.47). SD1.5 A2H and SDXL A2H@1024 also line up.
  **SDXL H2A@1024 breaks the pattern**: its own default leans toward
  Hamamatsu (source, not target) by the largest SDXL margin measured, yet
  it produces one of the two strongest real results (+9.20) — the opposite
  of what the hypothesis predicts.

**Reads as:** backbone colour bias is real and explains SD1.5's asymmetry
cleanly — it has one *consistent* colour lean (toward Hamamatsu) regardless
of which direction's LoRA is loaded, which alone accounts for why A2H is
comfortable and H2A collapses for that specific backbone. SDXL's own
default is not similarly fixed — it flips by resolution and direction
rather than having one backbone-wide lean — so colour bias is a real
contributing factor but does not fully explain SDXL's own asymmetry;
something else (plausibly how strongly the genuinely-conditioned model,
real ControlNet input plus source structure, overrides the backbone's
uncoerced default once actual source content is supplied) must also be
doing real work in the SDXL H2A@1024 case specifically. Reported honestly
as a partial, evidence-grounded explanation, not a single clean mechanism.

**Files:** `eval/p3_06_h2a_512_ablation/`, `eval/p3_06_h2a_512_full/`
(SDXL H2A @512px), `eval/p3_07_h2a_1024_ablation/`,
`eval/p3_07_h2a_1024_full/` (SDXL H2A @1024px, ≤50 pairs) — each with
`{summary.csv / eval_summary_final.csv, per_crop.csv, paired_win_rate.csv}`.
`images/<backbone><resolution>_<direction>/t2i_seed0.png` — one
representative text-to-image sample per combination (24 seeds were run per
combination on the cluster; only one is archived here as a qualitative
illustration).

**Full narrative:** `tickets/PHASE3-TICKETS.md` P3-06b/P3-07 H2A, sections
D1 and D2; `docs/results/RESULTS_SUMMARY.md`.
