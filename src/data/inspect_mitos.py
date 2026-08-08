#!/usr/bin/env python3
"""
inspect_mitos.py -- MITOS-ATYPIA-14 dataset structure inspector.

Purpose
-------
Before writing any crop-extraction code, discover the *actual* on-disk layout
of your MITOS-ATYPIA-14 download, so the extractor is built against reality
rather than assumptions. This answers the five questions the crop pipeline
needs:

  1. How are Aperio (A) vs Hamamatsu (H) frames named and organised?
  2. What magnifications exist, and where do the x20 frames live?
  3. What are the pixel dimensions of x20 frames per scanner?
     (this reveals the resolution mismatch that forces registration)
  4. Are there correspondence / coordinate files mapping A<->H frames?
  5. Are A03/H03 and the five held-out pairs (A06/A08/A09/A13/A16) all present?

It reads only image *headers* (dimensions), never full pixel data, so it stays
fast and light even on large multi-page TIFFs.

Usage
-----
    python inspect_mitos.py --root /path/to/MITOS-ATYPIA-14
    python inspect_mitos.py --root /path/to/MITOS-ATYPIA-14 --manifest manifest.json
    python inspect_mitos.py --root /path/to/MITOS-ATYPIA-14 --dims-sample 2

Dependencies
------------
    Pillow          (required for dimension reads; falls back gracefully)
    tifffile        (optional; used only if Pillow cannot open a TIFF)

Copy the full stdout back to me, plus manifest.json if you generate it.
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

# ----------------------------------------------------------------------
# Optional imaging backends -- both are optional; the script still reports
# structure without them, just without dimensions.
# ----------------------------------------------------------------------
try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None  # MITOS frames are large; disable the bomb guard
    _HAVE_PIL = True
except Exception:  # noqa: BLE001
    _HAVE_PIL = False

try:
    import tifffile  # noqa: F401
    _HAVE_TIFF = True
except Exception:  # noqa: BLE001
    _HAVE_TIFF = False


IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
TABLE_EXTS = {".csv", ".txt", ".xml", ".tsv"}

# Expected slides per the research proposal, so the report can flag gaps.
TRAINING_SLIDES = ["03", "04", "05", "07", "10", "11", "12", "14", "15", "17", "18"]
HELDOUT_SLIDES = ["06", "08", "09", "13", "16"]

SCANNER_NAME = {"A": "Aperio", "H": "Hamamatsu"}

# Slide token like A03 / H16 appearing at the start of a filename stem or a
# directory name. Anchored so it does not match spuriously mid-token.
SLIDE_RE = re.compile(r"^([AH])(\d{2})")
# Magnification tokens seen in MITOS-style trees: x20, x40, x10, x2.5, 20x, etc.
MAG_RE = re.compile(r"^x?(\d+(?:\.\d+)?)x?$", re.IGNORECASE)
PLAUSIBLE_MAGS = {"2.5", "5", "10", "20", "40"}


def human_int(n):
    return f"{n:,}"


def detect_slide(stem, path_parts):
    """Return (scanner_letter, slide_number) or None.

    Checks the filename stem first, then each parent directory name, so it works
    whether the dataset labels frames by filename (A03_...) or by folder (A03/).
    """
    m = SLIDE_RE.match(stem)
    if m:
        return m.group(1).upper(), m.group(2)
    for part in path_parts:
        m = SLIDE_RE.match(part)
        if m:
            return m.group(1).upper(), m.group(2)
    return None


def detect_mag(path_parts, stem):
    """Return a magnification token (e.g. 'x20') if one is present in the path."""
    for part in list(path_parts) + [stem]:
        m = MAG_RE.match(part)
        if m and m.group(1) in PLAUSIBLE_MAGS:
            return "x" + m.group(1)
    return None


def get_dims(path):
    """Return (width, height, mode_or_dtype) reading only the header, or None."""
    if _HAVE_PIL:
        try:
            with Image.open(path) as im:
                return im.width, im.height, im.mode
        except Exception:  # noqa: BLE001
            pass
    if _HAVE_TIFF and path.lower().endswith((".tif", ".tiff")):
        try:
            import tifffile
            with tifffile.TiffFile(path) as tf:
                shp = tf.pages[0].shape
                dt = str(tf.pages[0].dtype)
                if len(shp) == 3:
                    h, w, _ = shp
                elif len(shp) == 2:
                    h, w = shp
                else:
                    return None
                return w, h, dt
        except Exception:  # noqa: BLE001
            pass
    return None


def print_tree(root, max_depth=3, max_entries=25):
    """Compact depth-limited directory tree."""
    root = os.path.abspath(root)
    print("\nDIRECTORY TREE (depth <= %d, <= %d entries/dir)" % (max_depth, max_entries))
    print("-" * 68)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath[len(root):].count(os.sep)
        if depth > max_depth:
            dirnames[:] = []
            continue
        dirnames.sort()
        filenames.sort()
        indent = "  " * depth
        label = os.path.basename(dirpath) or dirpath
        n_img = sum(1 for f in filenames if os.path.splitext(f)[1].lower() in IMAGE_EXTS)
        n_oth = len(filenames) - n_img
        print(f"{indent}{label}/  [{human_int(n_img)} images, {human_int(n_oth)} other]")
        shown = filenames[:max_entries]
        for f in shown:
            print(f"{indent}  - {f}")
        if len(filenames) > max_entries:
            print(f"{indent}  ... (+{len(filenames) - max_entries} more)")


def main():
    ap = argparse.ArgumentParser(description="Inspect a MITOS-ATYPIA-14 download.")
    ap.add_argument("--root", required=True, help="Path to the MITOS-ATYPIA-14 root folder.")
    ap.add_argument("--manifest", default=None, help="Optional path to write a JSON manifest.")
    ap.add_argument("--dims-sample", type=int, default=1,
                    help="Images to open per (slide, magnification) group for dimensions.")
    ap.add_argument("--tree-depth", type=int, default=3)
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"ERROR: not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    print("=" * 68)
    print("MITOS-ATYPIA-14 INSPECTION")
    print("=" * 68)
    print(f"Root: {root}")
    print(f"Pillow available:   {_HAVE_PIL}")
    print(f"tifffile available: {_HAVE_TIFF}")
    if not (_HAVE_PIL or _HAVE_TIFF):
        print("NOTE: no imaging backend -> dimensions will be skipped. "
              "Install Pillow for dimension reads: pip install Pillow")

    # -- Pass 1: walk everything (cheap; no image decode) --------------------
    n_dirs = 0
    image_paths = []
    table_paths = []
    unmatched_samples = []
    # group[(scanner, slide, mag)] -> list of paths
    group = defaultdict(list)
    mags_seen = set()

    for dirpath, dirnames, filenames in os.walk(root):
        n_dirs += 1
        rel_parts = os.path.relpath(dirpath, root).split(os.sep)
        for f in filenames:
            ext = os.path.splitext(f)[1].lower()
            full = os.path.join(dirpath, f)
            if ext in TABLE_EXTS:
                table_paths.append(full)
                continue
            if ext not in IMAGE_EXTS:
                continue
            image_paths.append(full)
            stem = os.path.splitext(f)[0]
            slide = detect_slide(stem, rel_parts)
            mag = detect_mag(rel_parts, stem)
            if mag:
                mags_seen.add(mag)
            if slide:
                group[(slide[0], slide[1], mag or "?")].append(full)
            else:
                if len(unmatched_samples) < 15:
                    unmatched_samples.append(os.path.relpath(full, root))

    print("\nTOTALS")
    print("-" * 68)
    print(f"Directories : {human_int(n_dirs)}")
    print(f"Image files : {human_int(len(image_paths))}")
    print(f"Table files : {human_int(len(table_paths))}  (.csv/.txt/.xml/.tsv)")
    print(f"Magnifications seen in paths: "
          f"{', '.join(sorted(mags_seen)) if mags_seen else '(none detected in path names)'}")

    # -- Tree ----------------------------------------------------------------
    print_tree(root, max_depth=args.tree_depth)

    # -- Per-slide x magnification crosstab ----------------------------------
    slides_found = sorted({(s, n) for (s, n, _m) in group.keys()})
    print("\nFRAME COUNTS  (scanner-slide x magnification)")
    print("-" * 68)
    all_mags = sorted({m for (_s, _n, m) in group.keys()})
    header = "slide".ljust(10) + "".join(m.ljust(10) for m in all_mags) + "total"
    print(header)
    per_slide_mag = defaultdict(dict)
    for (s, n, m), paths in group.items():
        per_slide_mag[(s, n)][m] = len(paths)
    for (s, n) in slides_found:
        row = f"{s}{n}".ljust(10)
        total = 0
        for m in all_mags:
            c = per_slide_mag[(s, n)].get(m, 0)
            total += c
            row += (str(c) if c else "-").ljust(10)
        row += str(total)
        print(row)

    # -- Sample dimensions per (slide, mag) group ----------------------------
    print("\nSAMPLE IMAGE DIMENSIONS  (width x height, mode/dtype)")
    print("-" * 68)
    print("Watch for Aperio vs Hamamatsu size differences at the same nominal "
          "magnification -- that mismatch is why held-out eval needs registration.")
    dims_manifest = {}
    for (s, n, m) in sorted(group.keys()):
        paths = group[(s, n, m)]
        sample = paths[: max(1, args.dims_sample)]
        for p in sample:
            d = get_dims(p)
            key = f"{s}{n} {m}"
            if d:
                w, h, mode = d
                print(f"  {key.ljust(14)} {w} x {h}   {mode}   ({os.path.basename(p)})")
                dims_manifest.setdefault(f"{s}{n}", {})[m] = [w, h, str(mode)]
            else:
                print(f"  {key.ljust(14)} (could not read dimensions: {os.path.basename(p)})")

    # -- Table / correspondence files ----------------------------------------
    print("\nTABLE / CORRESPONDENCE FILES  (first 2 lines of up to 8)")
    print("-" * 68)
    if not table_paths:
        print("  (none found -- check whether frame-correspondence CSVs were "
              "included in the download; the extractor's pairing logic depends on this)")
    for tp in sorted(table_paths)[:8]:
        print(f"  {os.path.relpath(tp, root)}")
        try:
            with open(tp, "r", errors="replace") as fh:
                for _ in range(2):
                    line = fh.readline().rstrip("\n")
                    if not line:
                        break
                    print(f"      | {line[:120]}")
        except Exception as e:  # noqa: BLE001
            print(f"      (unreadable: {e})")
    if len(table_paths) > 8:
        print(f"  ... (+{len(table_paths) - 8} more table files)")

    # -- Unmatched naming ----------------------------------------------------
    if unmatched_samples:
        print("\nIMAGE FILES NOT MATCHING THE A##/H## PATTERN  (sample)")
        print("-" * 68)
        print("If these are real frames, the slide/scanner naming differs from the "
              "assumed A03/H03 convention and the extractor must adapt:")
        for u in unmatched_samples:
            print(f"  - {u}")

    # -- Presence check vs proposal -----------------------------------------
    print("\nPRESENCE CHECK vs PROPOSAL")
    print("-" * 68)
    have = {(s, n) for (s, n) in slides_found}

    def check(role, letters, nums):
        present, missing = [], []
        for lt in letters:
            for nm in nums:
                (present if (lt, nm) in have else missing).append(f"{lt}{nm}")
        print(f"  {role}:")
        print(f"    present ({len(present)}): {', '.join(present) if present else '(none)'}")
        print(f"    missing ({len(missing)}): {', '.join(missing) if missing else '(none)'}")
        return present, missing

    tr_present, tr_missing = check("Training slides (A/H)", "AH", TRAINING_SLIDES)
    ho_present, ho_missing = check("Held-out slides (A/H)", "AH", HELDOUT_SLIDES)

    # -- JSON manifest -------------------------------------------------------
    if args.manifest:
        manifest = {
            "root": root,
            "backends": {"pillow": _HAVE_PIL, "tifffile": _HAVE_TIFF},
            "counts": {
                "dirs": n_dirs,
                "images": len(image_paths),
                "tables": len(table_paths),
            },
            "magnifications": sorted(mags_seen),
            "slides": {
                f"{s}{n}": {
                    "scanner": SCANNER_NAME.get(s, s),
                    "mags": per_slide_mag.get((s, n), {}),
                    "sample_dims": dims_manifest.get(f"{s}{n}", {}),
                }
                for (s, n) in slides_found
            },
            "tables": [os.path.relpath(t, root) for t in sorted(table_paths)],
            "unmatched_samples": unmatched_samples,
            "presence_check": {
                "training_present": tr_present,
                "training_missing": tr_missing,
                "heldout_present": ho_present,
                "heldout_missing": ho_missing,
            },
        }
        with open(args.manifest, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"\nWrote manifest: {os.path.abspath(args.manifest)}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
