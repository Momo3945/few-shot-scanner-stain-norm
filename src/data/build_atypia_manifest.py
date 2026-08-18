#!/usr/bin/env python3
"""
build_atypia_manifest.py -- join MITOS-ATYPIA-14 atypia-score labels to their
x20 frame images (P2-09, Clinical Utility).

Walks an Aperio root for `<slide-root>/atypia/x20/<slide>_<frame>_cna_score_decision.csv`
label files (one integer 1-3 per file, no header -- verified directly against real
files, both the training and testing archives) and joins each to its sibling
`<slide-root>/frames/x20/<slide>_<frame>.tiff` frame image, writing one row per
labelled frame to `atypia_manifest.csv` (slide, frame_id, image_path, atypia_score).
Crops happen downstream at train/eval time from these frame paths -- this script
only builds the frame-level label index, same division of labour as
extract_pairs.py's train_manifest.csv (which lists geometry, not pixels).

IMPORTANT directory-layout gotcha (verified directly, not assumed from CLAUDE.md's
existing A03 note): the training archive has an INCONSISTENT layout across slides.
A03 is doubled (`A03/A03/atypia/x20/...`, from an earlier separate extraction) but
every other training slide (A04, A05, A07, A10-A12, A14, A15, A17, A18) is NOT
doubled (`A10/atypia/x20/...` directly). The testing/held-out archive is never
doubled (matches CLAUDE.md). This script does not assume either layout -- it
locates each `atypia/x20/*_cna_score_decision.csv` via a recursive glob and
resolves frames/x20 relative to THAT file's own parent chain
(csv.parent.parent.parent == the slide root, whichever depth it's actually at),
so both layouts are handled correctly without special-casing.

Also verified directly: 3 of the 300 training decision files are genuinely empty
(no adjudicated score at all -- A10_01C, A14_00A, A17_02B; their _cna_score_all.csv
siblings are empty too, not just the decision file) -- these frames are skipped,
not treated as parse errors.

Usage
-----
    python build_atypia_manifest.py \
        --aperio-root /datasets/mhoosen/stain-norm/mitos_atypia_train_aperio \
        --out pairs/atypia_train_manifest.csv

    python build_atypia_manifest.py \
        --aperio-root /datasets/mhoosen/stain-norm/mitos_heldout/mitos_atypia_2014_testing_aperio \
        --out pairs/atypia_test_manifest.csv

Dependencies: none beyond the standard library.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

DECISION_SUFFIX = "_cna_score_decision.csv"
VALID_SCORES = {1, 2, 3}


def parse_args():
    ap = argparse.ArgumentParser(
        description="Join MITOS-ATYPIA-14 atypia-score labels to their x20 frame images.")
    ap.add_argument("--aperio-root", required=True,
                    help="Root containing <slide[-root]>/atypia/x20/*_cna_score_decision.csv "
                         "and .../frames/x20/*.tiff, at any nesting depth.")
    ap.add_argument("--out", required=True, help="Output atypia_manifest.csv path.")
    return ap.parse_args()


def main():
    args = parse_args()
    root = Path(args.aperio_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    decision_paths = sorted(root.glob(f"**/atypia/x20/*{DECISION_SUFFIX}"))
    if not decision_paths:
        raise SystemExit(f"No *{DECISION_SUFFIX} files found under {root}")

    n_written = n_empty = n_bad_score = n_missing_frame = 0
    rows = []
    for dp in decision_paths:
        stem = dp.name[: -len(DECISION_SUFFIX)]
        raw = dp.read_text().strip()
        if not raw:
            n_empty += 1
            continue
        try:
            score = int(raw)
        except ValueError:
            print(f"  WARNING: unparseable score {raw!r} in {dp} -- skipping.")
            n_bad_score += 1
            continue
        if score not in VALID_SCORES:
            print(f"  WARNING: score {score} out of range 1-3 in {dp} -- skipping.")
            n_bad_score += 1
            continue

        # x20 -> atypia -> slide root, whichever depth this file was actually found at
        slide_root = dp.parent.parent.parent
        frame_path = slide_root / "frames" / "x20" / f"{stem}.tiff"
        if not frame_path.exists():
            print(f"  WARNING: no frame image at {frame_path} for {dp} -- skipping.")
            n_missing_frame += 1
            continue

        slide, frame_id = stem.split("_", 1)
        rows.append((slide, frame_id, str(frame_path.relative_to(root)).replace("\\", "/"), score))
        n_written += 1

    rows.sort()
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["slide", "frame_id", "image_path", "atypia_score"])
        w.writerows(rows)

    print(f"Wrote {n_written} labelled frames to {out_path}")
    print(f"  skipped: {n_empty} empty labels, {n_bad_score} bad/out-of-range scores, "
          f"{n_missing_frame} missing frame images")
    if rows:
        from collections import Counter
        dist = Counter(r[3] for r in rows)
        print(f"  score distribution: {dict(sorted(dist.items()))}")


if __name__ == "__main__":
    main()
