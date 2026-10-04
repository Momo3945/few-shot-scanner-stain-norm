#!/usr/bin/env python3
"""
extract_atypia_x40_features.py -- P2-15 step 1: frozen UNI2-h embeddings for the
x40 subfields (<slide>/frames/x40/<slide>_<frame><a-d>.tiff) together with their
six-criteria rater scores (<slide>/atypia/x40/<slide>_<frame><a-d>_cna_criteria.csv).
See tickets/PHASE2-TICKETS.md P2-15.

Each x40 subfield is tiled at native 224px (no downscale: nuclear detail is the
point), tissue-filtered (same 0.30 rule), capped at --max-tiles by a
deterministic stride, embedded, and mean-pooled to ONE vector per subfield.
Criteria value = mean over the (up to 3) rater scores; NaN when the file has
names only (no scores). Output .npz keys: emb [n,1536], slide, frame, sub,
crit [n,6] (order = CRITERIA), n_tiles; plus a .json sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

CRITERIA = ["nuclei_size", "anisonucleosis", "nucleoli_size",
            "chromatin_density", "membrane_thickness", "nuclei_contour"]


def parse_args():
    ap = argparse.ArgumentParser(description="P2-15: x40 subfield embeddings + criteria scores.")
    ap.add_argument("--root", required=True, help="Dir containing <slide>/frames/x40 and <slide>/atypia/x40.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", default="uni2h")
    ap.add_argument("--tile", type=int, default=224)
    ap.add_argument("--tissue-thresh", type=float, default=0.30)
    ap.add_argument("--max-tiles", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="Max subfields (smoke test).")
    ap.add_argument("--allow-cpu", action="store_true")
    return ap.parse_args()


def read_criteria(path):
    """-> list of 6 floats (NaN where unavailable)."""
    import numpy as np
    vals = {}
    for line in Path(path).read_text().strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        scores = [float(x) for x in p[1:] if x != ""]
        vals[p[0]] = float(np.mean(scores)) if scores else float("nan")
    return [vals.get(c, float("nan")) for c in CRITERIA]


def main():
    args = parse_args()
    import numpy as np
    import torch
    from pathology_encoder import load_encoder, encode, set_deterministic
    from registration import read_rgb

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise SystemExit("ABORT: no GPU on this node -- resubmit (or --allow-cpu).")
    set_deterministic()

    root = Path(args.root)
    items = []
    for tiff in sorted(root.glob("A*/frames/x40/*.tiff")):
        m = re.match(r"(A\d+)_(\d+[A-D])([a-d])$", tiff.stem)
        if not m:
            continue
        slide, frame, sub = m.groups()
        crit = tiff.parents[2] / "atypia" / "x40" / f"{tiff.stem}_cna_criteria.csv"
        items.append((slide, frame, sub, tiff, crit if crit.exists() else None))
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise SystemExit(f"No x40 subfields found under {root}")

    def tissue_fraction(rgb):
        a = rgb.astype(np.int16); mx = a.max(2); mn = a.min(2)
        return float(((mx < 235) & ((mx - mn) > 12)).mean())

    def offsets(length, t):
        if length <= t:
            return [0]
        o = list(range(0, length - t + 1, t))
        if o[-1] != length - t:
            o.append(length - t)
        return sorted(set(o))

    model, info = load_encoder(args.encoder, device)
    print(f"Loaded encoder {info}; {len(items)} subfields")

    embs, meta = [], {"slide": [], "frame": [], "sub": [], "crit": [], "n_tiles": []}
    t0 = time.time()
    for i, (slide, frame, sub, tiff, crit) in enumerate(items):
        rgb = read_rgb(tiff)
        H, W = rgb.shape[:2]
        tiles = []
        for y in offsets(H, args.tile):
            for x in offsets(W, args.tile):
                t = rgb[y:y + args.tile, x:x + args.tile]
                if t.shape[0] == args.tile and t.shape[1] == args.tile and tissue_fraction(t) >= args.tissue_thresh:
                    tiles.append(t)
        if len(tiles) > args.max_tiles:
            idx = np.linspace(0, len(tiles) - 1, args.max_tiles).round().astype(int)
            tiles = [tiles[j] for j in idx]
        if not tiles:
            continue
        embs.append(encode(model, tiles, device, batch_size=args.batch_size, size=args.tile).mean(axis=0))
        meta["slide"].append(slide); meta["frame"].append(frame); meta["sub"].append(sub)
        meta["crit"].append(read_criteria(crit) if crit else [float("nan")] * len(CRITERIA))
        meta["n_tiles"].append(len(tiles))
        if (i + 1) % 50 == 0 or i + 1 == len(items):
            print(f"  [{i + 1}/{len(items)}] {slide}_{frame}{sub}: {len(tiles)} tiles ({time.time() - t0:.0f}s)")

    emb = np.stack(embs)
    if not np.isfinite(emb).all():
        raise SystemExit("ABORT: non-finite embeddings.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, emb=emb, slide=np.array(meta["slide"]), frame=np.array(meta["frame"]),
             sub=np.array(meta["sub"]), crit=np.array(meta["crit"], dtype=np.float32),
             n_tiles=np.array(meta["n_tiles"]))
    side = {**info, "root": str(args.root), "tile": args.tile, "max_tiles": args.max_tiles,
            "tissue_thresh": args.tissue_thresh, "n_subfields": int(emb.shape[0]),
            "n_with_criteria": int(np.isfinite(np.array(meta["crit"])).all(axis=1).sum()),
            "criteria_order": CRITERIA, "embedding_sha256": hashlib.sha256(emb.tobytes()).hexdigest(),
            "seconds": round(time.time() - t0, 1)}
    out.with_suffix(".json").write_text(json.dumps(side, indent=2))
    print(f"\nWrote {emb.shape[0]} subfield embeddings -> {out}  (with criteria: {side['n_with_criteria']})")


if __name__ == "__main__":
    main()
