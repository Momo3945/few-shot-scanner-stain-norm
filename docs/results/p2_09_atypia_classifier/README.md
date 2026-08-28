# P2-09 — Downstream Atypia Classifier

**What this tests:** a ResNet18 trained on real MITOS-ATYPIA-14
atypia-severity labels (1–3 scale, per-slide train/val split, class-weighted
loss), frozen, then scored — zero retraining — against every method's
output. Tests whether normalisation helps or hurts a task a pathologist
would actually care about, not just a pixel/colour metric. Best validation
accuracy: 94.68%.

**Status:** ✅ DONE — the one place diffusion beats classical outright, and
arguably the most clinically relevant metric in the project. Δaccuracy vs.
raw Hamamatsu (A06 excluded from every method uniformly):

| Method | Δ accuracy |
|---|---|
| **A3 @0.50 (ControlNet+LoRA)** | **+0.0625** |
| A1 @0.50 (ControlNet only, no colour LoRA) | +0.0577 |
| A3 @0.30 | +0.0529 |
| **Reinhard (best classical)** | **+0.0457** |
| raw Aperio (sanity check) | +0.0144 |
| Histogram matching | +0.0024 |
| **Macenko** | **−0.0673** |

Every top diffusion rung beats every classical baseline — the opposite of
the pattern everywhere else in this project. Caveat that matters: the
classifier is a fairly weak instrument (raw Aperio scores only 42.79% on 3
classes, barely above the 33% chance floor) — the *ranking* is the reliable
part, not the absolute deltas. A1's strong showing (structural conditioning
alone, no colour adaptation) hints ControlNet's conditioning specifically —
not colour transfer — may be doing real work here.

**Files:** `eval/per_crop.csv`, `eval/summary.csv` (table above).

**Full narrative:** `docs/results/RESULTS_SUMMARY.md` ("Update
(2026-08-19): P2-09") and `tickets/PHASE2-TICKETS.md` P2-09.
