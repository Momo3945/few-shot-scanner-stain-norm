#!/usr/bin/env python3
"""
extract_atypia_features.py -- P2-14 step 1: frozen-encoder tile embeddings for
the atypia probe. See tickets/PHASE2-TICKETS.md P2-14.

For every labelled frame in --manifest: tile with the project's standard 512px
grid + 0.30 tissue filter (same convention as train_atypia_classifier.py, but
ALL tissue tiles by default, not just 4), embed each tile with the frozen
encoder (pathology_encoder.py), and cache embeddings + metadata to one .npz and
a .json sidecar (provenance: manifest SHA-256, encoder/library versions,
tile counts). Run once for the train manifest and once for the held-out test
manifest (held-out Aperio features are used ONLY for the final one-time
sanity check, never for fitting or selection).

Usage
-----
    python extract_atypia_features.py --manifest pairs/atypia_train_manifest.csv \
        --root /datasets/mhoosen/stain-norm/mitos_atypia_train_aperio \
        --out /datasets/mhoosen/stain-norm/classifier/features/uni2h_train.npz
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))


def parse_args():
    ap = argparse.ArgumentParser(description="P2-14: frozen-encoder tile embeddings for the atypia probe.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True, help="Output .npz (a .json sidecar is written next to it).")
    ap.add_argument("--encoder", default="uni2h")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--tissue-thresh", type=float, default=0.30)
    ap.add_argument("--max-crops-per-frame", type=int, default=0, help="0 = all tissue tiles.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit-frames", type=int, default=0, help="0 = all (smoke tests use a small number).")
    ap.add_argument("--allow-cpu", action="store_true")
    return ap.parse_args()


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


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

    with open(args.manifest, newline="") as fh:
        rows = [dict(r, atypia_score=int(r["atypia_score"])) for r in csv.DictReader(fh)]
    rows.sort(key=lambda r: (r["slide"], r["frame_id"]))
    if args.limit_frames:
        rows = rows[: args.limit_frames]
    for r in rows:
        if r["atypia_score"] not in (1, 2, 3):
            raise SystemExit(f"ABORT: label {r['atypia_score']} out of range for {r['slide']}_{r['frame_id']}")

    def tissue_fraction(rgb):
        a = rgb.astype(np.int16); mx = a.max(2); mn = a.min(2)
        return float(((mx < 235) & ((mx - mn) > 12)).mean())

    def grid_offsets(length, crop):
        if length <= crop:
            return [0]
        offs = list(range(0, length - crop + 1, crop))
        if offs[-1] != length - crop:
            offs.append(length - crop)
        return sorted(set(offs))

    model, info = load_encoder(args.encoder, device)
    print(f"Loaded encoder {info}")

    emb_chunks, meta = [], {"slide": [], "frame": [], "x": [], "y": [], "label": []}
    t0 = time.time()
    for i, r in enumerate(rows):
        rgb = read_rgb(Path(args.root) / r["image_path"])
        H, W = rgb.shape[:2]
        tiles, coords = [], []
        for y in grid_offsets(H, args.crop):
            for x in grid_offsets(W, args.crop):
                if args.max_crops_per_frame and len(tiles) >= args.max_crops_per_frame:
                    break
                tile = rgb[y:y + args.crop, x:x + args.crop]
                if tissue_fraction(tile) < args.tissue_thresh:
                    continue
                tiles.append(tile); coords.append((x, y))
        if tiles:
            emb_chunks.append(encode(model, tiles, device, batch_size=args.batch_size))
            for (x, y) in coords:
                meta["slide"].append(r["slide"]); meta["frame"].append(r["frame_id"])
                meta["x"].append(x); meta["y"].append(y); meta["label"].append(r["atypia_score"])
        print(f"  [{i + 1}/{len(rows)}] {r['slide']}_{r['frame_id']}: {len(tiles)} tiles "
              f"({time.time() - t0:.0f}s)")

    if not emb_chunks:
        raise SystemExit("No tissue tiles embedded -- check --root / --tissue-thresh.")
    emb = np.concatenate(emb_chunks, axis=0)
    if not np.isfinite(emb).all():
        raise SystemExit("ABORT: non-finite values in embeddings.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, emb=emb, slide=np.array(meta["slide"]), frame=np.array(meta["frame"]),
             x=np.array(meta["x"]), y=np.array(meta["y"]), label=np.array(meta["label"]))
    sidecar = {
        **info, "manifest": str(args.manifest), "manifest_sha256": sha256_file(args.manifest),
        "crop": args.crop, "tissue_thresh": args.tissue_thresh,
        "max_crops_per_frame": args.max_crops_per_frame, "n_frames": len(rows),
        "n_tiles": int(emb.shape[0]), "embedding_sha256": hashlib.sha256(emb.tobytes()).hexdigest(),
        "seconds": round(time.time() - t0, 1),
    }
    out.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
    print(f"\nWrote {emb.shape[0]} tile embeddings ({emb.shape[1]}-d) from {len(rows)} frames -> {out}")
    print(f"embedding_sha256={sidecar['embedding_sha256']}")


if __name__ == "__main__":
    main()
