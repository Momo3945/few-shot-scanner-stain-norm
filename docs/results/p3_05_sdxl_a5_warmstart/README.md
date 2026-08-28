# P3-05 — A5-SDXL (Histopathology Warm-Start Transfer)

**What this tests:** transfers the histopathology warm-start LoRA to SDXL
(needed training from scratch — no SDXL histopathology LoRA existed
before), same operating point as P3-04's A4-SDXL. Tests whether the
warm-start prior helps SDXL the way it helped SD1.5's A5 (a dramatic
A06-specific win there).

**Status:** ❌ DONE — a small, real, but fundamentally different result
than SD1.5's A5:

| Scope | A4-SDXL SSIM | A5-SDXL SSIM | A4-SDXL Δlab | A5-SDXL Δlab |
|---|---|---|---|---|
| ALL | 0.5272 | 0.5263 | −5.60 | **−5.30** |
| A06 | 0.3856 | 0.3850 | −3.58 | −3.55 |

Small, consistent colour-recovery improvement on every slide (+0.24 to
+0.56 LAB units — real, not noise), but **not** the dramatic A06-specific
win SD1.5's A5 showed (+2.6 to +8.7 LAB units there) — on SDXL, A06 barely
moves. Colour recovery stays negative everywhere; SSIM unchanged. SDXL's
frozen-base colour drift is a much larger effect than a histopathology
prior can nudge.

**Files:** `eval/full_heldout/eval_summary.csv` (table above),
`eval_per_crop.csv`, `eval_manifest.csv`.

**Full narrative:** `docs/results/RESULTS_SUMMARY.md` ("Update
(2026-08-21): P3-05") and `tickets/PHASE3-TICKETS.md` P3-05.
