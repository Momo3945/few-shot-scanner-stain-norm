# Supervisor update — 2026-08-28 meeting

Quick catch-up for today. Read top-down; the "Delivery notes" section at
the bottom is for you, not to read aloud. This replaces the 2026-08-19
version of this file — that meeting already happened; this is prep for the
next one.

## What he asked for last time, and the direct answer

His words, from the 2026-08-20 meeting: **"never beat baselines, continue
working with SDXL, if SDXL does not beat the baselines we might have to look
at the images."** You told him at that point you were training the ResNet
classifier.

**Direct answer on SDXL: no, it doesn't beat the baselines either — tested
four different ways, not just one config.**

| SDXL configuration | SSIM | Colour recovery Δlab |
|---|---|---|
| A4-SDXL (ported best SD1.5 config, strength 0.20) | 0.527 (best SSIM of any diffusion config, still below every classical baseline 0.63–0.68) | **−5.60** (negative on every slide) |
| A5-SDXL (+ histopathology prior) | 0.526 | −5.30 |
| Source-conditioned architecture on SDXL, 512px | 0.392 | +1.22 (positive, but ~60% weaker than the SD1.5 equivalent) |
| Same, retrained at native 1024px | 0.431 | −5.60 (flips negative on every slide) |

So this is not a config-tuning gap — bigger backbone alone was tested at
four different operating points and never closes the structural gap to
classical. **That confirms his implicit hypothesis was worth checking, and
the answer is it isn't just backbone size.**

## "We might have to look at the images" — already done, in parallel, not sequentially

This is the part he hasn't seen yet, and it's the best news in this update.
Rather than wait for the SDXL verdict to fail before starting this, two
targeted fixes were made directly to *how the SD1.5 pipeline itself builds
the output image* — literally "looking at the images" in the sense of the
generation mechanism, not just scaling the model:

1. **The colour LoRA never saw the source image during training at all** —
   direction was only imposed at inference via img2img strength. Fixed by
   adding genuine training-time source conditioning (a small ControlNet-style
   branch, 12.6M params, conditioned on source RGB + Canny).
2. **Inference started from randomly-corrupted noise, not the source
   image's own trajectory.** Fixed by replacing that with DDIM inversion —
   a deterministic, source-specific starting point obtained by running the
   trained model's own noise-prediction backwards.

**Result, same 39 training pairs, same frozen SD1.5 backbone, no new data:**

| | SSIM | Colour recovery Δlab |
|---|---|---|
| Classical baselines (range) | 0.628–0.681 | +3.2 to +9.0 |
| Original best diffusion config (Phase 1 ladder) | 0.454–0.459 | +1.5 to +1.9 |
| **After both fixes (source conditioning + DDIM inversion)** | **0.496** | **+8.74** |

Colour recovery roughly **tripled** and SSIM gained ~10% relative, purely
from fixing two mechanisms — no retraining data added, no bigger model.
Still below classical, but the gap narrowed substantially, and a dedicated
check found *why* it can't close further with this approach: a pure
VAE encode→decode pass (zero denoising at all) already caps SSIM at
**0.5393** — below every classical baseline before the diffusion model does
anything. That's the real ceiling, and it points at a specific next fix
(swap the decoder — see "in progress" below), not at continuing to tune
strength/steps.

## Confirmed as of this morning — one config now clears classical outright

This is the headline of the whole update. Direct continuation of the "look
at the images" fix: instead of asking the VAE to resynthesise the whole
image, fuse the raw untouched Aperio source's fine detail back into P1-11's
output *after* generation — let diffusion supply only the colour
transformation, let the source supply the structure.

An 80-crop preliminary check (A06+A08 only) looked promising yesterday but
wasn't the full picture. **The genuine full 496-crop, 5-slide held-out run
completed this morning:**

| Scope | SSIM | Recovery Δlab |
|---|---|---|
| Classical baselines (range) | 0.628–0.681 | +3.2 to +9.0 |
| **This fusion config, ALL 5 slides** | **0.729** | **+7.81** |
| A08 / A09 / A13 / A16 (each individually) | 0.669–0.787, all clear the classical range | all positive |
| A06 (the persistent outlier) | 0.617 — narrowly short of only the weakest baseline (0.628) | +11.25 |

**This is the first result in this project's entire history — 37+
configurations tested — to clear classical stain-normalisation baselines on
SSIM, at true full held-out scale, not a subset.** Four of five slides clear
it individually. Colour recovery is a real, modest trade against P1-11's own
best (+7.81 vs. +8.74, ~89% retained) — not a case of quietly reverting
toward raw to win on structure; checked directly with a pixel-diff sanity
test, not just inferred from the SSIM number.

**One thing worth mentioning if he asks how solid this is**: getting here
took catching and discarding two real bugs along the way — a scoring
baseline-weighting bug (fixed, pushed) and, more seriously, a scaling
shortcut that accidentally leaked the real target image into the input and
produced an obviously-too-good SSIM≈0.998 that was caught, checksummed, and
thrown out before it ever reached this doc. The 0.729 above is a clean,
re-verified run. Worth stating plainly as evidence the number is trustworthy
precisely *because* the bad one didn't slip through, not despite it.

## What you told him you were doing — the ResNet classifier — has a result, and it's the one clean win

Trained on real MITOS-ATYPIA-14 atypia-severity labels, 94.68% validation
accuracy, then frozen and scored — zero retraining — against every method's
output. This is the only evaluation in the whole project where diffusion
beats classical outright, and it's arguably the most clinically relevant
metric measured.

| Method | Δ accuracy vs. raw Hamamatsu |
|---|---|
| **Diffusion (ControlNet + colour LoRA)** | **+0.0625** |
| Diffusion (ControlNet only, no colour) | +0.0577 |
| Diffusion (ControlNet + colour LoRA, lower strength) | +0.0529 |
| Reinhard (best classical) | +0.0457 |
| Raw Aperio (sanity check) | +0.0144 |
| Histogram matching | +0.0024 |
| Macenko | −0.0673 |

Top three spots are all diffusion, all beating every classical method. Worth
flagging the caveat honestly if he asks: the classifier itself is a fairly
weak instrument (even raw Aperio, its own training domain, only scores
42.79% on 3 classes), so the *ranking* is the trustworthy part, not the
absolute numbers.

## A third finding worth mentioning — classical wins on average but is far less predictable

Not from a new experiment — from re-reading the raw per-slide numbers behind
the classical-baseline comparison rather than only the pooled averages.
Restricted to the four typical (non-outlier) held-out slides:

| | Colour recovery: std. dev. across slides | SSIM: std. dev. across slides |
|---|---|---|
| Classical (Macenko / Reinhard / Histogram Matching) | 5.0–7.0 LAB units | 0.049–0.062 |
| Diffusion (best operating point) | **~1.5, zero negative slides** | **~0.030** |

Classical wins the average by more, but swings hard slide-to-slide — each
classical method has its own distinct slide it does badly on (Macenko on
A13, Reinhard on A13+A16, histogram matching on A09+A16), sometimes flipping
fully negative (worse than doing nothing). Diffusion's smaller average gain
is consistent everywhere — at its best operating point, every single
typical-slide result is positive, for all three diffusion rungs tested. If
he asks "so which one would you actually trust in production," this is the
honest answer: classical recovers more colour on average, diffusion is far
more predictable about not making things worse on an unseen slide.

## Everything else tried since the last meeting (for completeness, not to lead with)

| Ticket | What it tested | Verdict |
|---|---|---|
| P1-12 | Generic LCM (few-step) acceleration on top of the fixed pipeline | 🟡 real speed/quality tradeoff, not a substitute |
| P1-13 / P1-13b | Task-specific LCM distilled from the frozen fixed pipeline | ❌ closed negative — loses to the generic adapter |
| P1-17 | Spatially-varying denoising strength (protect edges more) | ❌ closed negative — no improvement over flat strength |
| P1-14 | Drop-in VAE decoder swap, isolated self-reconstruction test | 🟢 in progress, positive — +0.045–0.048 SSIM, full-pipeline stage running now |
| P2-07 | Round-trip reconstruction (no real target image involved at all) | ❌ confirms the structural-drift pattern independently |
| P2-08 | Downstream nucleus-detection agreement (HoVer-Net) vs. a 0.95 bar | ❌ 0.8745, fails the pre-committed threshold |
| P2-10 | Cross-hospital generalisation (CAMELYON17, 5 centres never trained on) | ❌ makes centres slightly less alike, not more |

None of these change the headline; they round it out and rule out easy
explanations (LCM scheduler artefacts, an unfair metric, an untested
hospital domain).

## The one-liner for today

> SDXL doesn't rescue this on its own — tested four ways, confirmed it's not
> just a backbone-size problem. But instead of waiting on that answer, I also
> did what you suggested and looked directly at how the images themselves are
> being generated: found and fixed two real mechanisms in the pipeline
> (training-time source conditioning, deterministic initialisation) that
> roughly tripled colour recovery with the same data and the same small
> model, and traced the remaining gap to a specific, fixable bottleneck — the
> frozen decoder, which caps structural fidelity before generation even
> starts. Building on that fix, one config now clears classical baselines on
> SSIM outright, at full 496-crop held-out scale, confirmed this morning —
> the first result in this project's history to do that. Separately, the
> ResNet classifier test I mentioned last time is done, and it's the one
> place this pipeline beats every classical baseline on a real downstream
> clinical task, also fully confirmed.

## Questions worth putting to him

- Given the VAE ceiling finding, is it reasonable to prioritise the
  decoder-swap direction (already showing early gains) over further SDXL
  work, or does he still want the SDXL question fully closed out first?
- Does the classifier result change how the thesis should frame its central
  claim — still "generative normalisation loses on structural fidelity" as
  the headline, with the classifier task as a notable secondary finding, or
  does he want it elevated?
- Is there a point of diminishing returns on the negative-result catalogue
  (LCM distillation, spatially-varying strength) worth flagging so effort
  shifts fully to the two live positive leads?
- Does the per-slide predictability finding matter for how the thesis frames
  "which method would you actually deploy" — average recovery vs. worst-case
  reliability are different questions, and this is evidence they don't point
  the same way.
- Now that the post-hoc fusion result is confirmed and clearing classical
  baselines outright, does it become the primary reported configuration for
  the thesis, ahead of P1-11 — and does that change how much further effort
  goes into P1-14's decoder swap or SDXL versus consolidating this result
  (the F1/F2 held-out comparison and the Relative Dice/HoVer-Net check are
  still open)?

## If he asks for a specific number and you blank

Fine to say "let me pull that up." The tables above cover what's most likely
to come up; the full detail lives in `docs/results/RESULTS_SUMMARY.md` and
`tickets/PHASE1-3-TICKETS.md`.

## Delivery notes (for you, not him)

- **Lead with the SDXL answer first, since it's what he asked for** — don't
  bury it. Then pivot straight to "and here's what I did instead of just
  waiting for that."
- **This update is stronger than the 08-19 one, not just longer.** Last time
  was a well-instrumented negative result. This time there's a genuine,
  reproducible positive trajectory (source conditioning + DDIM inversion)
  *and* a genuine clean win (the classifier) to set against the SDXL
  negative. Don't undersell that by defaulting back into apology mode.
- **The VAE-ceiling finding is the strongest technical point in this
  update** — it turns "diffusion loses on structure" from a flat negative
  into a diagnosed, specific, addressable bottleneck with a fix already in
  progress. Lead the technical explanation with it if he pushes on why SDXL
  didn't just fix things.
- **The fusion result is now confirmed — lead with it confidently, but keep
  the honest framing.** It's the best number in the whole project, at true
  full held-out scale, not a subset. Still worth mentioning you caught and
  discarded an invalid leaked-target version (0.998) on the way there —
  that's evidence the 0.729 is trustworthy, not a reason to hedge on it.
  What's genuinely still open: the F1/F2 comparison hasn't been repeated at
  held-out scale, and the Relative Dice/HoVer-Net structural check (the
  ticket's own optional final-config check) hasn't run yet — say that if
  asked, don't imply the whole picture is closed.
- **It's fine to not know something on the spot.** "Let me check and follow
  up" remains a completely normal thing to say.
