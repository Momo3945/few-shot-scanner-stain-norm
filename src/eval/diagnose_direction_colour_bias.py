#!/usr/bin/env python3
"""
diagnose_direction_colour_bias.py -- investigates WHY A2H/H2A colour recovery
is asymmetric and backbone-dependent (tickets/PHASE3-TICKETS.md P3-06b/P3-07
H2A's closing finding: SDXL's H2A recovery is robustly positive regardless of
resolution/pair-count while its A2H recovery is fragile/negative; SD1.5 shows
the opposite asymmetry).

Hypothesis under test: each frozen pretrained backbone (SD1.5, SDXL) carries
its own inherent generative colour bias (from pretraining, independent of any
scanner-specific fine-tuning), and that bias sits closer to one scanner's real
colour profile than the other. If so, whichever direction's TARGET domain is
already close to the backbone's own default would need less correction to
"work" and would be more robust; the direction demanding a push away from the
backbone's default would fight that prior and be more fragile.

Method: reuse the mandatory `zero`-mode source-conditioning ablation outputs
that already exist for every trained checkpoint in this project (ControlNet's
conditioning forced to zero, so the output reflects the frozen backbone +
trained colour LoRA's own default generation, NOT genuine source guidance --
exactly isolates "what does this backbone+LoRA want to generate on its own").
No new training/inference needed -- these images already exist from each
checkpoint's own mandatory ablation gate. Computes each zero-mode output set's
mean L*a*b* (OpenCV convention, matching metrics.py's own
`cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)`) and compares it against the REAL
domain-wide mean L*a*b* of Aperio and Hamamatsu (computed once from all 50
pairs/train crops each side, as a fixed, run-independent domain colour
reference) -- reports which real domain's colour each zero-mode set sits
closer to, and by how much.

Usage
-----
    python diagnose_direction_colour_bias.py --root /datasets/mhoosen/stain-norm --mode zero

`--mode text2img` compares the follow-up pure text2img samples from
`diagnose_text2img_colour_default.py` instead (no img2img source-latent
confound -- see that script's docstring for why this second mode exists):

    python diagnose_direction_colour_bias.py --root /datasets/mhoosen/stain-norm --mode text2img
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np


def mean_lab_of_paths(paths: list[str]) -> np.ndarray:
    """Mean L*a*b* (OpenCV convention) across every pixel of every image."""
    totals = np.zeros(3, dtype=np.float64)
    n_pixels = 0
    for p in paths:
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64)
        totals += lab.sum(axis=0)
        n_pixels += lab.shape[0]
    return totals / n_pixels


def main():
    ap = argparse.ArgumentParser(description="Diagnose A2H/H2A colour-recovery asymmetry.")
    ap.add_argument("--root", default="/datasets/mhoosen/stain-norm")
    ap.add_argument("--mode", choices=["zero", "text2img"], default="zero",
                    help="zero: reuse existing zero-mode ablation outputs (img2img source-latent "
                         "confound, see docstring). text2img: the follow-up pure text2img samples "
                         "from diagnose_text2img_colour_default.py (no source-latent confound).")
    args = ap.parse_args()
    root = Path(args.root)

    # ---- fixed, run-independent real domain colour references ----
    aperio_paths = sorted(glob.glob(str(root / "pairs/train/*_aperio.png")))
    hamamatsu_paths = sorted(glob.glob(str(root / "pairs/train/*_hamamatsu.png")))
    print(f"Real Aperio reference: {len(aperio_paths)} crops")
    print(f"Real Hamamatsu reference: {len(hamamatsu_paths)} crops")
    lab_aperio = mean_lab_of_paths(aperio_paths)
    lab_hamamatsu = mean_lab_of_paths(hamamatsu_paths)
    print(f"  Real Aperio mean L*a*b*    = {lab_aperio}")
    print(f"  Real Hamamatsu mean L*a*b* = {lab_hamamatsu}")
    dom_dist = np.linalg.norm(lab_aperio - lab_hamamatsu)
    print(f"  (real domain-to-domain LAB distance: {dom_dist:.2f} -- the scale "
          f"against which the comparisons below should be read)\n")

    if args.mode == "zero":
        # ---- every zero-mode ablation this project has generated, per
        # backbone/resolution/direction combination (no new compute -- these
        # already exist) ----
        combos = [
            ("SD1.5",       "A2H", "eval/p1_10_ablation_zero", True),
            ("SD1.5",       "H2A", "eval/p1_10_h2a_ablation_zero", True),
            ("SDXL@512",    "A2H", "eval/p3_06_ablation_zero", True),
            ("SDXL@512",    "H2A", "eval/p3_06_h2a_ablation_zero_2000", True),
            ("SDXL@1024",   "A2H", "eval/p3_07_ablation_zero", True),
            ("SDXL@1024",   "H2A", "eval/p3_07_h2a_ablation_zero_2000", True),
        ]
    else:
        # ---- follow-up pure text2img samples (diagnose_text2img_colour_
        # default.py output dirs -- no "outputs" subfolder, written flat) ----
        combos = [
            ("SD1.5",       "A2H", "eval/diagnose_t2i/sd15_a2h", False),
            ("SD1.5",       "H2A", "eval/diagnose_t2i/sd15_h2a", False),
            ("SDXL@512",    "A2H", "eval/diagnose_t2i/sdxl512_a2h", False),
            ("SDXL@512",    "H2A", "eval/diagnose_t2i/sdxl512_h2a", False),
            ("SDXL@1024",   "A2H", "eval/diagnose_t2i/sdxl1024_a2h", False),
            ("SDXL@1024",   "H2A", "eval/diagnose_t2i/sdxl1024_h2a", False),
        ]

    print(f"{'Backbone':<10} {'Dir':<4} {'n':>4}  {'mean L*a*b*':<28} "
          f"{'dist->Aperio':>13} {'dist->Hamamatsu':>16}  closer to")
    for backbone, direction, rel, has_outputs_subdir in combos:
        out_dir = root / rel / "outputs" if has_outputs_subdir else root / rel
        paths = sorted(glob.glob(str(out_dir / "*.png")))
        if not paths:
            print(f"{backbone:<10} {direction:<4}  -- NO FILES FOUND at {out_dir} --")
            continue
        lab_out = mean_lab_of_paths(paths)
        d_aperio = float(np.linalg.norm(lab_out - lab_aperio))
        d_hamamatsu = float(np.linalg.norm(lab_out - lab_hamamatsu))
        closer = "APERIO" if d_aperio < d_hamamatsu else "HAMAMATSU"
        margin = abs(d_aperio - d_hamamatsu)
        lab_str = f"[{lab_out[0]:.1f}, {lab_out[1]:.1f}, {lab_out[2]:.1f}]"
        print(f"{backbone:<10} {direction:<4} {len(paths):>4}  {lab_str:<28} "
              f"{d_aperio:>13.2f} {d_hamamatsu:>16.2f}  {closer} (by {margin:.2f})")

    mode_note = ("(zero-mode: still an img2img pass from the real source latent -- "
                  "a 'closer to SOURCE' result here reflects source-latent persistence, "
                  "not a colour-free backbone default; see the script docstring)"
                  if args.mode == "zero" else
                  "(text2img: no source image/latent at all -- this IS the colour-free "
                  "backbone+LoRA default)")
    print(f"\nMode: {args.mode} {mode_note}")
    print("Interpretation guide: TARGET domain for A2H is Hamamatsu, for H2A is "
          "Aperio. If a row's 'closer to' column matches its own TARGET, the "
          "frozen backbone+LoRA's own default already leans toward the correct "
          "colour -- consistent with that direction being the robust one for "
          "this backbone. A mismatch means the default leans toward the WRONG "
          "domain, consistent with that direction needing to fight the "
          "backbone's own prior -- consistent with fragility/negative recovery.")


if __name__ == "__main__":
    main()
