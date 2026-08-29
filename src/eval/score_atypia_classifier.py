#!/usr/bin/env python3
"""
score_atypia_classifier.py -- score the trained atypia classifier against every
already-computed method's crop outputs (P2-09, Clinical Utility).

Per the proposal: "recovery delta = performance(Proposed) - performance(Raw
Hamamatsu)". This script runs the classifier ONCE (inference only, no retraining)
over every method's existing eval_manifest.csv outputs, joins predictions back to
the true atypia score via the testing-root manifest built by
build_atypia_manifest.py, and reports accuracy/macro-F1/macro-AUC per method.

P2-12 hardening (tickets/P2-12_atypia_classifier_evaluation_hardening.md) rewrote
this script around several fixes discovered while specifying that ticket:

1. Translation-direction validation (S:3.1). The classifier is trained ONLY on
   Aperio -- the clinical-utility claim requires every scored method to be H2A
   (Hamamatsu input, normalised toward Aperio). P2-12 S:0 found EVERY method ever
   scored by the old version of this script was actually A2H (verified empirically,
   job 47468) -- the opposite direction, silently. This script now refuses to score
   a tag whose direction can't be resolved or doesn't match --expect-direction,
   unless --allow-direction-mismatch is passed, in which case the method is scored
   but its recovery_delta is suppressed (labelled non-clinical), never presented as
   the headline number.
2. Strictly paired recovery_delta (S:3.2). Every method's accuracy, and
   raw_hamamatsu's accuracy for that comparison, is computed on the exact same
   sample-key intersection -- not two independently-pooled populations.
3. Frame-level primary metric (S:3.3). The true label lives at the frame level;
   crop-level accuracy is now explicitly secondary/diagnostic (still in
   per_crop.csv). Primary metrics are in per_frame.csv / summary.csv rows with
   level=frame, using mean(crop_probability_vectors) -> argmax per frame.
3.5 Seed averaging (S:3.7). When a manifest has a "seed" column, crops sharing
   (slide,frame,x,y) across seeds are averaged at the PROBABILITY level into one
   crop-identity record before pairing/frame-aggregation -- not counted as N
   independent clinical samples.
4. One common outlier policy (S:3.4). Excluded slides are derived once, from
   raw_hamamatsu's own frame-level per-slide accuracy, and reused for every
   method's ALL_excl_outliers scope -- not re-derived per method.
5. Batched GPU inference (S:3.5). classify_batch() now moves one batch to device
   at a time; --batch-size actually bounds peak GPU memory.
6. Manifest-assumption + integrity checks (S:3.7/3.8). source_mode uniformity is
   verified (not assumed); duplicate (method,crop,seed) records are detected and
   deduplicated, logged; an integrity report is written per method.
7. Paired bootstrap CI for recovery_delta (S:3.9).
8. Extended outputs (S:3.10): per_crop.csv (secondary/diagnostic), per_frame.csv
   (primary), summary.csv (both levels, explicit `level` column), integrity_report.csv.

Each eval_manifest.csv is treated as one or more "methods": one method per
distinct strength value when a `strength` column exists and isn't "na" (the older
infer_colour_lora.py/infer_baseline.py schema), a single method per tag when a
`strength` column is absent (the newer infer_colour_source_ddim_inversion.py-family
schema, which has no notion of a strength sweep). Two additional synthetic methods
are derived without any new inference, from ONE EXPLICIT source tag's manifest
(--raw-hamamatsu-tag, default the first --tags entry) -- deliberately independent
of which tags are being scored as methods:
  - raw_hamamatsu: the source tag's reference_path column, deduplicated. Requires
    the source tag be A2H-direction -- reference_path is the Aperio crop for
    H2A-direction tags, not Hamamatsu (P2-12 bugfix, job 47648: an H2A run's
    raw_hamamatsu accidentally matched raw_aperio to 5 decimal places before this
    was caught and fixed). Validated at runtime, not assumed.
  - raw_aperio (sanity check, not proposal-required): the ORIGINAL raw Aperio frame
    re-cropped at the manifest's own (x, y, crop). Exempt from the direction gate --
    it's ground truth, not a translation output. Already direction-agnostic (always
    reads real aperio_path), unaffected by the raw_hamamatsu bug above.

Usage
-----
    python score_atypia_classifier.py \
        --checkpoint /datasets/mhoosen/stain-norm/classifier/atypia_r18/best.pt \
        --testing-manifest pairs/atypia_test_manifest.csv \
        --testing-root /datasets/mhoosen/stain-norm/mitos_heldout/mitos_atypia_2014_testing_aperio \
        --eval-root /datasets/mhoosen/stain-norm/eval \
        --tags h2a_a0 h2a_a1 \
        --raw-hamamatsu-tag a0 \
        --out eval/atypia_classifier_scores

    (--raw-hamamatsu-tag is required whenever --tags are H2A-direction, since none
    of them can supply genuine Hamamatsu pixels themselves -- point it at any
    existing A2H-direction tag, e.g. the canonical 'a0' run.)

Direction resolution per tag: reads <eval-root>/<tag>/run_metadata.json's
"direction" field if present (written by infer_colour_lora.py, infer_baseline.py,
infer_colour_source_ddim_inversion.py as of P2-12). If absent, falls back to
--tag-direction TAG=DIRECTION overrides. If still unresolved, the script fails
loudly naming the tag, rather than guessing.

Dependencies: torch, torchvision, numpy, opencv-python-headless, scikit-learn,
tifffile. Requires metrics.py and registration.py on the path (same folder).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

DEFAULT_TAGS = ["a0", "a1", "a2h_r4", "a2h_r8", "a3", "a4", "a5",
                "macenko", "reinhard", "histogram_matching"]


# ======================================================================
# Pure logic (no torch/cv2/sklearn) -- importable and unit-testable without
# a GPU or the cluster. Keep genuinely pure: no file I/O beyond what's passed in.
# ======================================================================

def parse_tag_direction_overrides(pairs):
    """['a0=A2H', 'h2a_a1=H2A'] -> {'a0': 'A2H', 'h2a_a1': 'H2A'}. Raises on bad input."""
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise ValueError(f"--tag-direction expects TAG=DIRECTION, got: {p!r}")
        tag, direction = p.split("=", 1)
        direction = direction.strip().upper()
        if direction not in ("A2H", "H2A"):
            raise ValueError(f"--tag-direction direction must be A2H or H2A, got: {p!r}")
        out[tag.strip()] = direction
    return out


def resolve_direction(tag_dir: Path, tag: str, overrides: dict):
    """Returns (direction_or_None, source_str) where source_str is 'run_metadata.json',
    '--tag-direction', or 'unresolved'."""
    meta_path = tag_dir / "run_metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            meta = {}
        d = meta.get("direction")
        if d in ("A2H", "H2A"):
            return d, "run_metadata.json"
    if tag in overrides:
        return overrides[tag], "--tag-direction"
    return None, "unresolved"


def detect_schema(rows):
    """(has_strength, has_seed) from the manifest's own header (first row's keys)."""
    if not rows:
        return False, False
    keys = rows[0].keys()
    return ("strength" in keys), ("seed" in keys)


def check_source_mode_uniform(rows, tag):
    """Raises SystemExit if a `source_mode` column exists and carries more than one
    distinct value -- P2-12 S:3.7: never silently pool incompatible modes together."""
    if not rows or "source_mode" not in rows[0]:
        return None
    modes = {r["source_mode"] for r in rows}
    if len(modes) > 1:
        raise SystemExit(
            f"ABORT: tag '{tag}' eval_manifest.csv mixes source_mode values {sorted(modes)} "
            f"in one manifest -- refusing to silently pool incompatible modes "
            f"(P2-12 S:3.7). Split this manifest by source_mode before scoring, or "
            f"pass a tag path that already isolates one mode.")
    return modes.pop()


def build_method_items(tag, rows, has_strength, has_seed):
    """rows -> {method_label: [item, ...]}, item = dict(slide,frame,x,y,seed,output_path).
    Also returns n_duplicates (rows sharing (label,slide,frame,x,y,seed) -- kept once).
    Does not touch the filesystem."""
    methods = defaultdict(list)
    seen = set()
    n_duplicates = 0
    for r in rows:
        if has_strength:
            strength = r["strength"]
            label = tag if strength == "na" else f"{tag}_s{strength}"
        else:
            label = tag
        seed = r["seed"] if has_seed else None
        key = (label, r["slide"], r["frame"], r["x"], r["y"], seed)
        if key in seen:
            n_duplicates += 1
            continue
        seen.add(key)
        methods[label].append({
            "slide": r["slide"], "frame": r["frame"], "x": r["x"], "y": r["y"],
            "seed": seed, "output_path": r["output_path"],
        })
    return dict(methods), n_duplicates


def ckey(item_or_row):
    """Canonical cross-method sample key: (slide, frame, x, y)."""
    return (item_or_row["slide"], item_or_row["frame"], item_or_row["x"], item_or_row["y"])


def average_probs_by_ckey(crop_records):
    """crop_records: list of (ckey_tuple, prob_vector[3]). Seeds sharing the same
    ckey are averaged (P2-12 S:3.7) -- returns {ckey: averaged_prob_vector} plus
    n_input (raw records in) for the caller's own accounting."""
    import numpy as np
    groups = defaultdict(list)
    for k, probs in crop_records:
        groups[k].append(probs)
    return {k: np.mean(np.stack(v), axis=0) for k, v in groups.items()}, len(crop_records)


def pair_keys(method_keys: set, raw_keys: set):
    """Returns (paired, method_only, raw_only) -- all sets of ckey tuples."""
    paired = method_keys & raw_keys
    return paired, method_keys - raw_keys, raw_keys - method_keys


def frame_groups(keys, key_to_frame):
    """keys -> {(slide,frame): [key, ...]} using key_to_frame(key) -> (slide,frame)."""
    out = defaultdict(list)
    for k in keys:
        out[key_to_frame(k)].append(k)
    return out


def validate_raw_hamamatsu_source_direction(direction, source_tag):
    """P2-12 bug fix: reference_path only represents genuine Hamamatsu pixels when the
    source tag's own direction is A2H -- for H2A, reference_path is the Aperio crop
    (infer_colour_lora.py's ref_frame=a_rgb branch), which silently made raw_hamamatsu
    identical to raw_aperio the first time an H2A run was scored (job 47648, caught by
    the two accuracies matching to 5 decimal places). Raises loudly instead of repeating
    that silently."""
    if direction != "A2H":
        raise SystemExit(
            f"ABORT: raw_hamamatsu cannot be derived from tag '{source_tag}' -- its "
            f"direction is {direction!r}, not A2H. For H2A-direction tags, reference_path "
            f"is the Aperio crop, not Hamamatsu (see tickets/"
            f"P2-12_atypia_classifier_evaluation_hardening.md). Pass --raw-hamamatsu-tag "
            f"pointing at an existing A2H-direction tag (e.g. 'a0') instead.")


def paired_bootstrap_ci(method_correct: dict, raw_correct: dict, frame_ids, n_boot=2000, seed=0):
    """method_correct/raw_correct: {frame_id: 0/1}, both covering every id in frame_ids.
    Returns (mean_delta, lo, hi) over a paired bootstrap resample of frame_ids."""
    frame_ids = list(frame_ids)
    if not frame_ids:
        return None, None, None
    rng = random.Random(seed)
    n = len(frame_ids)
    deltas = []
    for _ in range(n_boot):
        sample = [frame_ids[rng.randrange(n)] for _ in range(n)]
        m_acc = sum(method_correct[f] for f in sample) / n
        r_acc = sum(raw_correct[f] for f in sample) / n
        deltas.append(m_acc - r_acc)
    deltas.sort()
    lo = deltas[int(0.025 * n_boot)]
    hi = deltas[min(int(0.975 * n_boot), n_boot - 1)]
    mean_delta = sum(deltas) / n_boot
    return mean_delta, lo, hi


# ======================================================================
# I/O + orchestration (needs torch/cv2/sklearn, deferred into main())
# ======================================================================

def load_testing_labels(path):
    with open(path, newline="") as fh:
        return {(r["slide"], r["frame_id"]): int(r["atypia_score"]) for r in csv.DictReader(fh)}


def parse_args():
    ap = argparse.ArgumentParser(description="Score the atypia classifier against every method's outputs (P2-12 hardened).")
    ap.add_argument("--checkpoint", required=True, help="Trained classifier checkpoint (best.pt).")
    ap.add_argument("--testing-manifest", required=True,
                    help="atypia_manifest.csv built from the testing/held-out root (ground truth).")
    ap.add_argument("--testing-root", required=True,
                    help="Aperio root the testing manifest's image_path is relative to "
                         "(only used for --with-raw-aperio's re-crop).")
    ap.add_argument("--eval-root", required=True, help="Dir containing eval/<tag>/eval_manifest.csv per method.")
    ap.add_argument("--tags", nargs="+", default=DEFAULT_TAGS)
    ap.add_argument("--out", required=True, help="Output dir for per_crop.csv / per_frame.csv / summary.csv.")
    ap.add_argument("--resize", type=int, default=224)
    ap.add_argument("--crop", type=int, default=512, help="Crop size used for --with-raw-aperio's re-crop.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--outlier-z", type=float, default=3.5)
    ap.add_argument("--no-raw-aperio", action="store_true",
                    help="Skip the raw_aperio sanity-check method (on by default).")
    ap.add_argument("--raw-hamamatsu-tag", default=None,
                    help="Existing A2H-direction tag (e.g. 'a0') whose reference/ folder "
                         "holds genuine registered-Hamamatsu crops, used as the raw_hamamatsu "
                         "ground-truth baseline. Required whenever the run includes H2A tags -- "
                         "reference_path means Aperio, not Hamamatsu, for H2A (P2-12 bug). "
                         "Defaults to the first --tags entry if omitted, but that fallback is "
                         "only safe when the first tag is itself A2H -- validated at runtime "
                         "either way, never silently trusted.")
    ap.add_argument("--expect-direction", choices=["A2H", "H2A"], default="H2A",
                    help="Direction every non-synthetic tag must resolve to (P2-12 S:3.1). "
                         "H2A is what the clinical-utility claim actually needs, since the "
                         "classifier is Aperio-trained -- default H2A, not A2H, deliberately.")
    ap.add_argument("--tag-direction", nargs="+", default=[],
                    help="TAG=DIRECTION overrides when a tag's eval dir has no "
                         "run_metadata.json (e.g. historical pre-P2-12 runs). "
                         "Example: --tag-direction a0=A2H a1=A2H")
    ap.add_argument("--allow-direction-mismatch", action="store_true",
                    help="Score tags whose resolved direction != --expect-direction anyway. "
                         "Their recovery_delta is suppressed (marked non-clinical), never "
                         "presented as the headline number -- use only for the deliberate "
                         "'does colour restyling preserve enough structure' robustness "
                         "question P2-12 S:3.1 describes, not as a way past the gate.")
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--bootstrap-seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    tag_overrides = parse_tag_direction_overrides(args.tag_direction)

    import numpy as np
    import torch
    import cv2
    from PIL import Image
    from torchvision.models import resnet18
    from sklearn.metrics import f1_score, roc_auc_score

    from metrics import flag_outliers
    from registration import read_rgb

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels = load_testing_labels(args.testing_manifest)
    if not labels:
        raise SystemExit(f"Empty testing manifest: {args.testing_manifest}")

    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 3)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device); model.eval()
    print(f"Loaded checkpoint {args.checkpoint} (step {ckpt.get('step')}, "
          f"val_acc {ckpt.get('val_acc')})")

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def preprocess(rgb):
        img = Image.fromarray(rgb).resize((args.resize, args.resize), Image.LANCZOS)
        arr = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return (arr - mean) / std

    @torch.no_grad()
    def classify_batch(rgb_list):
        # P2-12 S:3.5: preprocess/stack/transfer per BATCH, not the whole method at
        # once -- peak GPU memory now scales with --batch-size, not len(rgb_list).
        out = []
        for i in range(0, len(rgb_list), args.batch_size):
            chunk = rgb_list[i:i + args.batch_size]
            x = torch.stack([preprocess(r) for r in chunk]).to(device)
            probs = torch.softmax(model(x), dim=1).cpu()
            out.append(probs)
            del x
        return torch.cat(out).numpy()

    def read_png(path):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"could not read {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------------
    # Build per-method item lists + resolve direction per tag (P2-12 S:3.1)
    # ------------------------------------------------------------------
    eval_root = Path(args.eval_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_methods = {}       # label -> [item, ...]
    method_tag = {}        # label -> originating tag (for direction lookup)
    total_duplicates = {}  # label -> n_duplicates dropped

    for tag in args.tags:
        tag_dir = eval_root / tag
        man_path = tag_dir / "eval_manifest.csv"
        if not man_path.exists():
            print(f"  WARNING: no manifest at {man_path} -- skipping tag {tag}.")
            continue
        with open(man_path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            print(f"  WARNING: empty manifest at {man_path} -- skipping tag {tag}.")
            continue

        check_source_mode_uniform(rows, tag)  # raises loudly on a real mix

        has_strength, has_seed = detect_schema(rows)
        tag_methods, n_dup = build_method_items(tag, rows, has_strength, has_seed)
        for label, items in tag_methods.items():
            all_methods[label] = items
            method_tag[label] = tag
            total_duplicates[label] = n_dup  # same manifest -> same dup count per label from this tag

    if not all_methods:
        raise SystemExit(f"No usable eval_manifest.csv found under {eval_root} for tags {args.tags}")

    # ------------------------------------------------------------------
    # raw_hamamatsu / raw_aperio ground truth -- derived from ONE explicit source
    # tag's manifest, independent of which --tags are actually being scored as
    # methods (P2-12 bugfix: this used to implicitly reuse the first --tags entry,
    # which silently mislabelled Aperio as Hamamatsu whenever that entry was
    # H2A-direction, since reference_path means Aperio for H2A -- caught when
    # raw_hamamatsu and raw_aperio scored identically to 5 decimal places, job
    # 47648). raw_hamamatsu specifically REQUIRES an A2H-direction source tag,
    # validated below, not assumed.
    # ------------------------------------------------------------------
    raw_ham_source_tag = args.raw_hamamatsu_tag or args.tags[0]
    raw_ham_source_dir = eval_root / raw_ham_source_tag
    raw_ham_source_direction, raw_ham_source_via = resolve_direction(
        raw_ham_source_dir, raw_ham_source_tag, tag_overrides)
    if raw_ham_source_direction is None:
        raise SystemExit(
            f"ABORT: cannot determine direction for raw_hamamatsu source tag "
            f"'{raw_ham_source_tag}' -- no run_metadata.json under {raw_ham_source_dir} and "
            f"no --tag-direction override given. Pass --tag-direction "
            f"{raw_ham_source_tag}=A2H if you know it, or pass --raw-hamamatsu-tag pointing "
            f"at a different, known-A2H tag.")
    validate_raw_hamamatsu_source_direction(raw_ham_source_direction, raw_ham_source_tag)
    print(f"  raw_hamamatsu source: tag={raw_ham_source_tag} -> {raw_ham_source_direction} "
          f"(via {raw_ham_source_via})")

    raw_ham_man_path = raw_ham_source_dir / "eval_manifest.csv"
    if not raw_ham_man_path.exists():
        raise SystemExit(f"ABORT: raw_hamamatsu source manifest not found at {raw_ham_man_path}.")
    with open(raw_ham_man_path, newline="") as fh:
        raw_ham_rows = list(csv.DictReader(fh))

    raw_hamamatsu_items, raw_aperio_items = [], []
    seen_ref = set()
    for r in raw_ham_rows:
        key = (r["slide"], r["frame"], r["x"], r["y"])
        if key in seen_ref:
            continue
        seen_ref.add(key)
        raw_hamamatsu_items.append({"slide": r["slide"], "frame": r["frame"],
                                    "x": r["x"], "y": r["y"], "seed": None,
                                    "_ref_path": raw_ham_source_dir / r["reference_path"]})
        if not args.no_raw_aperio:
            aperio_path = Path(args.testing_root) / r["aperio_path"]
            x, y, crop = int(r["x"]), int(r["y"]), args.crop
            raw_aperio_items.append({"slide": r["slide"], "frame": r["frame"],
                                     "x": r["x"], "y": r["y"], "seed": None,
                                     "_aperio_path": aperio_path, "_x": x, "_y": y, "_c": crop})

    if not raw_hamamatsu_items:
        raise SystemExit(f"raw_hamamatsu could not be derived -- '{raw_ham_source_tag}' produced "
                          f"no rows (P2-12 S:3.8 integrity guard: the clinical baseline must exist).")

    # Direction gate (P2-12 S:3.1) -- fail loudly before any classification happens,
    # not partway through a long GPU job.
    method_is_clinical = {}
    for label, tag in method_tag.items():
        direction, source = resolve_direction(eval_root / tag, tag, tag_overrides)
        if direction is None:
            raise SystemExit(
                f"ABORT: cannot determine translation direction for tag '{tag}' "
                f"(method '{label}'). No run_metadata.json under {eval_root / tag} and no "
                f"--tag-direction override given. This scorer refuses to guess (P2-12 S:0/S:3.1) "
                f"-- pass --tag-direction {tag}=A2H (or H2A) if you know it, or regenerate the "
                f"manifest with a current infer_colour_lora.py/infer_baseline.py/"
                f"infer_colour_source_ddim_inversion.py, which now write it.")
        mismatch = direction != args.expect_direction
        if mismatch and not args.allow_direction_mismatch:
            raise SystemExit(
                f"ABORT: tag '{tag}' (method '{label}') resolved direction={direction} "
                f"(via {source}), but --expect-direction={args.expect_direction}. Refusing to "
                f"silently score a mismatched-direction manifold as a clinical-utility result "
                f"(P2-12 S:0/S:3.1). Pass --allow-direction-mismatch to score it anyway as an "
                f"explicitly non-clinical robustness check.")
        method_is_clinical[label] = not mismatch
        print(f"  direction: tag={tag} method={label} -> {direction} (via {source})"
              + ("" if not mismatch else "  ** NON-CLINICAL (direction mismatch, scored under --allow-direction-mismatch) **"))

    # ------------------------------------------------------------------
    # Classify every crop (all methods + raw_hamamatsu + raw_aperio)
    # ------------------------------------------------------------------
    per_crop_path = out_dir / "per_crop.csv"
    per_crop_fh = open(per_crop_path, "w", newline="")
    pc_w = csv.writer(per_crop_fh)
    pc_w.writerow(["method", "slide", "frame", "x", "y", "seed",
                   "true_score", "pred_score", "correct", "prob_1", "prob_2", "prob_3"])

    def classify_items(method_label, items, loader_fn):
        """items already have output paths resolved via loader_fn(item) -> rgb.
        Returns crop_identity: {ckey: averaged_prob_vector}, plus n_unlabelled."""
        rgb_list, item_keys, seeds = [], [], []
        n_unlabelled = 0
        for it in items:
            if (it["slide"], it["frame"]) not in labels:
                n_unlabelled += 1
                continue
            item_keys.append(ckey(it))
            seeds.append(it["seed"])
            rgb_list.append(loader_fn(it))
        if n_unlabelled:
            print(f"  {method_label}: {n_unlabelled} crops skipped (no atypia label for their frame).")
        if not rgb_list:
            return {}, n_unlabelled
        probs = classify_batch(rgb_list)
        preds = probs.argmax(1) + 1
        crop_records = []
        for k, seed, p, pred in zip(item_keys, seeds, probs, preds):
            slide, frame = k[0], k[1]
            true = labels[(slide, frame)]
            pc_w.writerow([method_label, slide, frame, k[2], k[3], seed if seed is not None else "",
                          true, int(pred), int(true == pred),
                          round(float(p[0]), 5), round(float(p[1]), 5), round(float(p[2]), 5)])
            crop_records.append((k, p))
        crop_identity, _ = average_probs_by_ckey(crop_records)
        return crop_identity, n_unlabelled

    # raw_hamamatsu / raw_aperio first -- everything else pairs against raw_hamamatsu
    raw_ham_crops, _ = classify_items("raw_hamamatsu", raw_hamamatsu_items,
                                       lambda it: read_png(it["_ref_path"]))
    raw_aperio_crops = {}
    if not args.no_raw_aperio:
        raw_aperio_crops, _ = classify_items(
            "raw_aperio", raw_aperio_items,
            lambda it: read_rgb(it["_aperio_path"])[it["_y"]:it["_y"] + it["_c"], it["_x"]:it["_x"] + it["_c"]])

    if not raw_ham_crops:
        raise SystemExit("ABORT: raw_hamamatsu produced zero labelled crops -- cannot compute any "
                          "recovery_delta (P2-12 S:3.8 integrity guard).")

    raw_ham_keys = set(raw_ham_crops.keys())

    def key_to_frame(k):
        return (k[0], k[1])

    raw_ham_frames = frame_groups(raw_ham_keys, key_to_frame)

    # Common outlier policy (P2-12 S:3.4): derive ONCE from raw_hamamatsu's own
    # frame-level per-slide accuracy, reuse for every method.
    def frame_pred(probs_by_ckey_subset):
        return int(np.mean(np.stack(probs_by_ckey_subset), axis=0).argmax()) + 1

    raw_ham_slide_acc = defaultdict(lambda: [0, 0])  # slide -> [correct, total]
    for (slide, frame), keys in raw_ham_frames.items():
        true = labels[(slide, frame)]
        pred = frame_pred([raw_ham_crops[k] for k in keys])
        raw_ham_slide_acc[slide][0] += int(true == pred)
        raw_ham_slide_acc[slide][1] += 1
    raw_ham_slide_acc_ratio = {s: c / t for s, (c, t) in raw_ham_slide_acc.items()}
    outlier_flags = flag_outliers(raw_ham_slide_acc_ratio, z_thresh=args.outlier_z)
    excluded_slides = {s for s, f in outlier_flags.items() if f["outlier"]}
    print(f"\nCommon outlier policy (from raw_hamamatsu, frame-level, z>{args.outlier_z}): "
          f"excluded_slides={sorted(excluded_slides) or 'none'}")

    # ------------------------------------------------------------------
    # Score every real method, paired against raw_hamamatsu
    # ------------------------------------------------------------------
    per_frame_path = out_dir / "per_frame.csv"
    per_frame_fh = open(per_frame_path, "w", newline="")
    pf_w = csv.writer(per_frame_fh)
    pf_w.writerow(["method", "slide", "frame", "true_score", "pred_score", "correct",
                   "prob_1", "prob_2", "prob_3", "n_crops"])

    integrity_path = out_dir / "integrity_report.csv"
    integrity_fh = open(integrity_path, "w", newline="")
    ir_w = csv.writer(integrity_fh)
    ir_w.writerow(["method", "source_direction", "is_clinical", "paired_frames",
                   "paired_crops", "method_only_crops", "raw_only_crops", "duplicates_dropped"])

    summary_rows = []  # each: dict of columns (written out at the end)

    def pool_frame_level(method_frames, method_crops, keep_slides):
        t_all, p_all, pr_all = [], [], []
        for (slide, frame), keys in method_frames.items():
            if slide not in keep_slides:
                continue
            true = labels[(slide, frame)]
            probs_avg = np.mean(np.stack([method_crops[k] for k in keys]), axis=0)
            pred = int(probs_avg.argmax()) + 1
            t_all.append(true); p_all.append(pred); pr_all.append(probs_avg)
        if not t_all:
            return None
        t_all, p_all, pr_all = np.array(t_all), np.array(p_all), np.array(pr_all)
        acc = float((t_all == p_all).mean())
        f1 = f1_score(t_all, p_all, average="macro", zero_division=0)
        try:
            auc = roc_auc_score(t_all, pr_all, multi_class="ovr", average="macro", labels=[1, 2, 3])
        except ValueError:
            auc = None
        return {"n": len(t_all), "accuracy": acc, "macro_f1": f1, "macro_auc": auc}

    def add_summary(method, level, scope, stats, recovery_delta="", ci_lo="", ci_hi="",
                    paired_raw_acc="", excluded=""):
        summary_rows.append({
            "method": method, "level": level, "scope": scope,
            "n": stats["n"] if stats else 0,
            "accuracy": round(stats["accuracy"], 5) if stats else "",
            "macro_f1": round(stats["macro_f1"], 5) if stats else "",
            "macro_auc": round(stats["macro_auc"], 5) if (stats and stats["macro_auc"] is not None) else "",
            "paired_raw_hamamatsu_accuracy": paired_raw_acc,
            "recovery_delta": recovery_delta, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "excluded_slides": excluded,
        })

    # raw_hamamatsu's own summary rows (frame + crop level, no recovery_delta vs itself)
    ham_all = pool_frame_level(raw_ham_frames, raw_ham_crops, keep_slides=set(raw_ham_slide_acc))
    add_summary("raw_hamamatsu", "frame", "ALL", ham_all)
    if excluded_slides:
        ham_excl = pool_frame_level(raw_ham_frames, raw_ham_crops,
                                    keep_slides=set(raw_ham_slide_acc) - excluded_slides)
        add_summary("raw_hamamatsu", "frame", "ALL_excl_outliers", ham_excl, excluded=sorted(excluded_slides))
    for (slide, frame), keys in sorted(raw_ham_frames.items()):
        true = labels[(slide, frame)]
        pred = frame_pred([raw_ham_crops[k] for k in keys])
        pf_w.writerow(["raw_hamamatsu", slide, frame, true, pred, int(true == pred), "", "", "", len(keys)])

    all_labels_to_score = sorted(all_methods) + (["raw_aperio"] if raw_aperio_crops else [])

    for label in all_labels_to_score:
        if label == "raw_aperio":
            crops = raw_aperio_crops
            is_clinical = False  # sanity check only, never a headline recovery_delta
            direction_str = "ground_truth"
        else:
            items = all_methods[label]
            tag = method_tag[label]
            tag_dir = eval_root / tag
            crops, _ = classify_items(label, items, lambda it, td=tag_dir: read_png(td / it["output_path"]))
            is_clinical = method_is_clinical[label]
            direction_str, _ = resolve_direction(tag_dir, tag, tag_overrides)

        method_keys = set(crops.keys())
        paired, method_only, raw_only = pair_keys(method_keys, raw_ham_keys)
        method_frames = frame_groups(paired, key_to_frame)  # only paired keys enter frame aggregation
        paired_frame_ids = set(method_frames.keys())

        overlap_frac = len(paired) / max(len(method_keys), 1)
        if overlap_frac < 0.90:
            print(f"  WARNING: {label} paired crop overlap with raw_hamamatsu is only "
                  f"{overlap_frac:.1%} ({len(paired)}/{len(method_keys)}) -- larger mismatch "
                  f"than expected same-architecture variance; verify this isn't a bug "
                  f"(P2-12 S:3.2).")

        n_paired_crops = len(paired)
        n_dup = total_duplicates.get(label, 0)
        ir_w.writerow([label, direction_str, is_clinical, len(paired_frame_ids),
                      n_paired_crops, len(method_only), len(raw_only), n_dup])
        print(f"method={label} paired_frames={len(paired_frame_ids)} "
              f"missing_vs_raw={len(method_only)+len(raw_only)} duplicates={n_dup} "
              f"source_direction={direction_str}")

        if not paired_frame_ids:
            print(f"  {label}: no paired frames with raw_hamamatsu -- skipping.")
            continue

        # method_frames was built from `paired` keys only (P2-12 S:3.2/3.3) -- reuse
        # the SAME frame->keys grouping to index BOTH `crops` and `raw_ham_crops`, so
        # raw_hamamatsu's per-frame average uses exactly the crops that overlap with
        # this method, not its own full (possibly larger) crop set for that frame.
        keep_slides_all = {slide for slide, _ in paired_frame_ids}
        m_all = pool_frame_level(method_frames, crops, keep_slides=keep_slides_all)

        if label == "raw_aperio" and m_all and ham_all and m_all["accuracy"] == ham_all["accuracy"]:
            print(f"  WARNING: raw_aperio accuracy ({m_all['accuracy']:.5f}) is IDENTICAL to "
                  f"raw_hamamatsu's -- this is the exact signature of the P2-12 direction bug "
                  f"(raw_hamamatsu accidentally reading Aperio pixels). Verify --raw-hamamatsu-tag "
                  f"'{raw_ham_source_tag}' is genuinely A2H-direction and its reference/ folder "
                  f"holds real Hamamatsu crops before trusting any recovery_delta in this run.")

        recovery_delta = ci_lo = ci_hi = paired_raw_acc = ""
        if is_clinical and m_all:
            raw_all_paired = pool_frame_level(method_frames, raw_ham_crops, keep_slides=keep_slides_all)
            if raw_all_paired:
                paired_raw_acc = round(raw_all_paired["accuracy"], 5)
                recovery_delta = round(m_all["accuracy"] - raw_all_paired["accuracy"], 5)
                method_correct = {f: int(labels[f] == int(np.mean(
                    np.stack([crops[k] for k in method_frames[f]]), axis=0).argmax()) + 1)
                    for f in paired_frame_ids}
                raw_correct = {f: int(labels[f] == int(np.mean(
                    np.stack([raw_ham_crops[k] for k in method_frames[f]]), axis=0).argmax()) + 1)
                    for f in paired_frame_ids}
                _, ci_lo, ci_hi = paired_bootstrap_ci(method_correct, raw_correct, paired_frame_ids,
                                                      n_boot=args.n_bootstrap, seed=args.bootstrap_seed)
                ci_lo = round(ci_lo, 5) if ci_lo is not None else ""
                ci_hi = round(ci_hi, 5) if ci_hi is not None else ""
        elif not is_clinical:
            recovery_delta = "N/A (non-clinical direction)"

        add_summary(label, "frame", "ALL", m_all, recovery_delta, ci_lo, ci_hi, paired_raw_acc)

        if excluded_slides:
            keep_excl = keep_slides_all - excluded_slides
            m_excl = pool_frame_level(method_frames, crops, keep_slides=keep_excl)
            if m_excl:
                excl_delta = ""
                if is_clinical:
                    raw_excl = pool_frame_level(method_frames, raw_ham_crops, keep_slides=keep_excl)
                    if raw_excl:
                        excl_delta = round(m_excl["accuracy"] - raw_excl["accuracy"], 5)
                add_summary(label, "frame", "ALL_excl_outliers", m_excl, excl_delta,
                           excluded=sorted(excluded_slides))

        for (slide, frame), keys in sorted(method_frames.items()):
            true = labels[(slide, frame)]
            probs_avg = np.mean(np.stack([crops[k] for k in keys]), axis=0)
            pred = int(probs_avg.argmax()) + 1
            pf_w.writerow([label, slide, frame, true, pred, int(true == pred),
                          round(float(probs_avg[0]), 5), round(float(probs_avg[1]), 5),
                          round(float(probs_avg[2]), 5), len(keys)])

        # secondary/diagnostic crop-level summary (paired set, no seed-double-count
        # since crops here are already the post-seed-averaging crop-identity records)
        t_all = np.array([labels[k[0], k[1]] for k in paired])
        p_all = np.array([int(crops[k].argmax()) + 1 for k in paired])
        crop_acc = float((t_all == p_all).mean()) if len(t_all) else None
        add_summary(label, "crop", "ALL", {"n": len(t_all), "accuracy": crop_acc,
                                           "macro_f1": f1_score(t_all, p_all, average="macro", zero_division=0) if len(t_all) else None,
                                           "macro_auc": None} if len(t_all) else None)

    per_crop_fh.close(); per_frame_fh.close(); integrity_fh.close()

    summary_path = out_dir / "summary.csv"
    fieldnames = ["method", "level", "scope", "n", "accuracy", "macro_f1", "macro_auc",
                 "paired_raw_hamamatsu_accuracy", "recovery_delta", "ci_lo", "ci_hi", "excluded_slides"]
    with open(summary_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)

    print(f"\nPer-crop  : {per_crop_path}  (secondary/diagnostic)")
    print(f"Per-frame : {per_frame_path}  (PRIMARY clinical result)")
    print(f"Summary   : {summary_path}")
    print(f"Integrity : {integrity_path}")


if __name__ == "__main__":
    main()
