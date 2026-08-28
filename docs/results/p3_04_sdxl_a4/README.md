# P3-04 — A4-SDXL (SDXL Port of the Original Ladder)

**What this tests:** direct SDXL port of A4 — colour LoRA trained on SDXL,
plus SDXL's own *pretrained* Canny ControlNet and *pretrained* LCM-LoRA
(not a freshly-trained conditioning branch — that's P3-06). Tests whether a
bigger frozen backbone fixes the structural gap on its own. Full mechanism:
Pipeline Anatomy §12.

**Status:** ❌ DONE — the mechanism is identified, and it doesn't rescue
SDXL. At the shared operating point (strength 0.20, 8-step LCM):

| | SD1.5 A4 | SDXL A4 |
|---|---|---|
| SSIM | 0.459 | **0.527** (best SSIM of any diffusion config at the time) |
| Recovery Δlab | +1.53 | **−5.60** (negative on every slide) |

A base-model-only ablation (LoRA omitted) showed the negative colour drift
is already fully present with zero LoRA involvement — frozen-SDXL-base
behaviour at this setting, not a broken/undertrained adapter. A strength
sweep found colour recovery flips positive above ~0.35 on A06 specifically,
but stays negative at every strength tested on a typical slide (A08) — no
single SDXL strength rescues the pooled comparison.

**Files:** `eval/full_heldout/eval_summary.csv` (table above),
`eval_per_crop.csv`, `eval_manifest.csv`.

**Images:** `images/sdxl_vs_sd15/` (A08 crop: raw / every classical
baseline / SD1.5 / SDXL / real Hamamatsu, side by side).

**Full narrative:** `docs/results/RESULTS_SUMMARY.md` ("Update
(2026-08-20): P3-04") and `tickets/PHASE3-TICKETS.md` P3-04.
