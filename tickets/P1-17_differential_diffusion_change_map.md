# P1-17 — Differential Diffusion Change-Map Inference (Spatially-Varying Strength)

**Status:** 🔄 IN PROGRESS (2026-08-27) — first smoke signal recorded (below),
mandatory internal-validation parameter grid not yet run.

**Smoke comparison (2026-08-27), 4 training pairs (A03), config
r4_s2_cmin0.00_cmax0.70:**

| Config | Job | Seeds | SSIM | LAB total |
|---|---|---|---|---|
| P1-10 baseline (global strength=0.70) | infer 47235 / score 47237 | 3 | 0.1616 ± 0.0057 | 22.44 ± 2.61 |
| P1-17 differential diffusion | infer 47229 / score 47231 | 1 | 0.192 | 23.69 |

SSIM improved (+0.030, outside the baseline's own seed-noise band) at a small
colour-recovery cost (LAB +1.25, i.e. slightly worse) — directionally
consistent with the ticket's hypothesis (protecting structurally-dense
regions trades a little colour recovery for structure), but this is a
4-crop, effectively single-seed sample on ONE slide (A03) — nowhere near
enough to draw a conclusion. Visual inspection of the change map on this
slide's densely-cellular crops showed it classifying most of the image as
"structure-dense" (protected) with only sparse flat/free regions, which may
be making r4/s2 more conservative than intended on this particular tissue
density. **Status update, not a result: this only justifies running the
actual parameter grid next, not adopting this config.**

**Script implemented** (`src/eval/infer_p1_10_differential_diffusion.py` +
`slurm/infer_p1_10_differential_diffusion.slurm`), smoke-tested on the
cluster (above). Next step: the mandatory internal-validation parameter
search (see Parameter Selection below) across more crops/slides than this
smoke test, never the held-out set first.

Inference-only refinement of P1-10/P1-11. No new training. Can run
independently of P1-14/P1-15/P1-16 (different mechanism — spatially-varying
denoising strength vs. VAE swap vs. post-hoc colour-residual reprojection) and
does not block or get blocked by them. Cheapest of the four follow-up tickets
to test since it needs zero new training runs and reuses the existing P1-10
checkpoint (and optionally P1-11's inversion path) directly.

**Implementation note (algorithm-critical, verified 2026-08-26):** the
underlying Differential Diffusion `map` argument uses the OPPOSITE sign
convention from this ticket's own `change_strength`/`c_min`/`c_max` naming —
HIGH map value protects a pixel (LESS change), LOW map value frees it to
regenerate (MORE change). Confirmed two ways against the official reference
(github.com/exx8/differential-diffusion): tracing `masks = map > thresholds`
through the per-step loop, and the project's own published teaser example,
where the map is WHITE exactly over the mosque structure that stays intact
in the output and BLACK over the fully-regenerated sky/water. The script
computes `change_strength` in this ticket's intuitive units (matching this
project's own `--strength` convention: 0=preserve, 1=regenerate) and inverts
it (`pipeline_map = 1 - change_strength`) immediately before the per-step
loop — get this backwards and the experiment silently protects flat colour
regions while freely regenerating edges, the opposite of the goal.

## Motivation

P1-10/P1-11's `infer_colour_translation.py`/DDIM-inversion pathway applies a
single **global** denoising strength (or, for inversion, a single global
translate-mode fraction `f`) to the entire crop. That forces one uniform
structure/colour tradeoff across the whole image: a strength high enough to
recover colour on flat tissue/background regions is also high enough to erode
fine structure (nuclear boundaries, chromatin texture) everywhere else, and
this project's own strength-sweep result already showed no single scalar
strength beats classical SSIM on both axes at once — a real tradeoff, not a
bug. P1-11 (DDIM inversion, `f=1.00`) is currently the best result in the
project (SSIM 0.4960 pooled / 0.5142 excl-A06, recovery Δlab +8.74 pooled) but
still sits below classical Macenko/Reinhard/Histogram-Matching SSIM
(0.628–0.681).

*Differential Diffusion* (Levin & Fried, 2023,
[arXiv:2306.00950](https://arxiv.org/abs/2306.00950)) replaces the single
global strength/threshold with a **per-pixel change map**: each region of the
image is assigned its own "unlock" point in the denoising trajectory, so
regions that should barely change stay pinned close to the source for most of
the trajectory while regions that should change freely evolve like an
ordinary higher-strength img2img generation — within one diffusion run, no
retraining, no new adapters. This project already computes a Canny edge map
of the source image for every crop (`extract_canny_control_image`, used as
half of the P1-10 ControlNet's 6-channel conditioning) — a direct, literally
already-computed input for the change map this technique needs.

## Scientific Question

> Does deriving a per-pixel change map from the source image's own Canny edge
> map — protecting structurally dense regions from change while letting flat
> colour regions change freely — improve SSIM over P1-10's global-strength
> img2img pathway, while retaining most of the colour recovery, on real
> held-out crops?

Desired separation (same spirit as P1-16, different mechanism):

```text
edge / structurally dense regions -> low change budget (protected)
flat / low-detail tissue regions  -> high change budget (free to recolour)
```

rather than one global strength applied uniformly.

## Scope

Inference-only. Reuses `lora/a2h_cond_r8/best` (P1-10's trained colour LoRA +
ControlNet) exactly as-is:

- no retraining, no rank/architecture/VAE/prompt/dataset changes;
- no new LoRA, no new ControlNet;
- `infer_colour_translation.py` and `infer_p1_10_ddim_inversion.py` (P1-11)
  are not modified — per this project's own precedent (P1-11/P1-13), each new
  ticket gets its own script.

Primary variant (**G1**) builds the change map on top of P1-10's existing
forward-noise img2img pathway (`infer_colour_translation.py`'s mechanism),
since Differential Diffusion's published algorithm is specified as a
modification of standard SDEdit-style forward-noising img2img, not of DDIM
inversion. Verify during implementation (re-read the paper/reference
implementation) whether the same per-pixel-threshold mechanism composes
cleanly with P1-11's inversion-derived starting latent; if not, **G2**
(differential map + DDIM inversion) becomes a stretch goal, not a
requirement — do not force an incompatible combination just to test it.

## Change-Map Construction — No Target Leakage

The change map must be derived **only** from:

```text
raw Aperio source crop A
its Canny edge map (already computed by extract_canny_control_image)
```

It must **never** use the real held-out Hamamatsu target, and must never be
computed per-crop from anything derived from scoring against the target
(e.g. do not pick the map per-crop to maximize that crop's own SSIM — the map
construction must be fixed/parametric, chosen once on internal validation,
then applied identically to held-out crops).

Raw binary Canny edges are 1-pixel-wide lines — using them unprocessed as the
change map would protect only the edge pixel itself and leave everything a
few pixels away (nucleus interior, chromatin texture) fully unprotected,
which would not meaningfully help SSIM. Convert to a smoothed structural
"protection field" before use, e.g. `dilate(canny, radius) -> Gaussian blur
(sigma) -> normalize to [0,1]`, then map to a change-strength range
`[c_min, c_max]` (edges/dense regions -> `c_min`, flat regions -> `c_max`).
`radius`, `sigma`, `c_min`, `c_max` are the tunable parameters (see below).

## Parameter Selection

Do **not** tune `radius`/`sigma`/`c_min`/`c_max` on the held-out slides
(A06/A08/A09/A13/A16). Use P1-10's internal training-domain validation split
only, matching P1-13/P1-14/P1-16's established convention. Freeze all
parameters before running the held-out evaluation.

Initial grid (keep it small, per this project's established preference —
see P1-16):

```text
radius (px)     = {2, 4, 8}
sigma            = {1, 2, 4}   (in the same units as radius, tissue-crop scale)
c_max            = {0.50, 0.70}   (matches P1-10's own best strengths)
c_min            = {0.0, 0.15}    (0.0 = edges frozen entirely; 0.15 = small residual colour drift allowed even at edges, since even nuclear regions have a real colour shift between scanners)
```

Do not launch a huge blind search — this is meant to be a small,
sanity-checked grid, not a hyperparameter sweep.

## New Script

Create `src/eval/infer_p1_10_differential_diffusion.py` (new script, per
project convention). Suggested interface, matching the existing P1-10/P1-11
scripts' CLI shape:

```text
--lora / --controlnet     (as in infer_colour_translation.py)
--root / --heldout / --out
--radius --sigma --c-min --c-max
--steps --guidance
--limit --max-crops-per-frame --seeds
```

Implementation note: diffusers has no built-in pipeline support for a
per-pixel change map on a ControlNet img2img pipeline, so (matching P1-11's
precedent) this will need a manual per-step loop over `pipe.vae`/`pipe.unet`/
`pipe.controlnet` rather than a single `pipe(...)` call, implementing the
per-pixel "unlock at threshold, otherwise re-pin toward the forward-noised
source latent at the current timestep" mechanism from the paper. Consult
arXiv:2306.00950 (and its reference implementation, if published) directly
for the exact per-step blending formula before implementing — this ticket
specifies the intended behaviour and interface, not the literal blending math.

Manifest schema: match `infer_colour_translation.py`'s strength-shaped output
(plus the `seed` column per this session's earlier fix to that script) so
`score_outputs.py` scores it unchanged.

## Mandatory Internal Controls

Compare the same validation samples across:

```text
raw source
P1-10 (best global strength)
P1-11 (DDIM inversion, f=1.00)
this method (best change-map config)
```

This prevents a change-map configuration that just approximates a low global
strength (reverting toward raw) from being misreported as a structural win —
same failure mode P1-16 guards against.

## Selection Objective

Optimize a joint colour/structure objective, not SSIM alone (same
`R_retained` ratio convention as P1-16):

```text
R_retained = Δlab_this_method / Δlab_P1-11
```

`R_retained` near 1.0 = full colour recovery retained; near 0 = reverted to
raw with no real colour transfer.

## Proposed Smoke Gate

Proceed to the full held-out run only if, on internal validation:

1. SSIM clearly improves over P1-10's best global-strength result;
2. colour recovery (`recovery_delta_lab`) stays positive;
3. at least ~70% of P1-11's colour-recovery gain is retained
   (`R_retained >= 0.70`) — set slightly below P1-16's ~80% threshold since
   this technique is expected to trade more colour for structure than a
   dedicated residual-projection method;
4. no visible seams/haloing at the protected/unprotected region boundary.

## Full Held-Out Evaluation

Canonical 496 held-out crops, 3 seeds (matching this project's established
protocol). Report per-slide (`A06, A08, A09, A13, A16, ALL, ALL excl A06`):

```text
SSIM, PSNR, MAE, windowed LAB, LAB total, dE2000, recovery Δlab, R_retained
```

## Required Comparison Table

```text
Raw
Macenko (0.628 SSIM)
Reinhard (0.681 SSIM)
Histogram Matching (0.651 SSIM)
P1-10
P1-11
P1-17 (this ticket, best config)
```

## Acceptance Criteria

```text
SSIM improves over P1-10/P1-11
AND
recovery Δlab stays positive on most/all slides
AND
R_retained >= 0.70 on the held-out set (not just internal validation)
```

Ideal outcome: SSIM moves measurably toward the classical 0.628–0.681 range
while keeping most of P1-11's colour recovery. The ticket does **not** pass
if the improvement is mostly explained by reverting toward raw Aperio (same
failure mode P1-16 explicitly guards against — check via the mandatory
internal controls above).

## Interpretation

### If it works
> Part of the remaining structural gap was caused by *forcing one global
> strength* rather than by the learned colour mapping itself — letting
> structurally dense regions change less (using information already computed
> for ControlNet conditioning, at zero extra training cost) recovers real
> SSIM without materially sacrificing colour recovery.

### If it fails
> The structure/colour tradeoff is not well explained by a spatially-varying
> *denoising strength* alone (the residual gap is more fundamental — e.g. the
> VAE bottleneck P1-14 targets, or a tradeoff that only a pixel-preserving
> architecture like P1-16 addresses). Do not chase a larger change-map
> parameter search after a clean negative result — move to P1-16 (or, if
> P1-14 has already passed, to P1-15/P1-16 in whatever order the results so
> far indicate) instead.

Either outcome is scientifically useful, and directly informative for
deciding whether P1-16's more invasive pixel-preserving fusion is worth its
implementation cost.
