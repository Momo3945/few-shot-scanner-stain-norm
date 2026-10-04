#!/usr/bin/env python3
"""
fit_atypia_probe.py -- P2-14 step 2: fit + validate a regularised linear head on
frozen-encoder frame embeddings (extract_atypia_features.py output). CPU only,
deterministic (L2 logistic regression, lbfgs, fixed seeds for the bootstrap).
See tickets/PHASE2-TICKETS.md P2-14.

Protocol (decided before running; no post-hoc loosening):
  1. Frame embedding = mean of that frame's tile embeddings.
  2. NESTED leave-one-slide-out CV on the TRAINING slides only: the outer loop
     holds out one slide; the inner loop (leave-one-slide-out over the remaining
     slides) picks the L2 strength C by pooled balanced accuracy. Out-of-fold
     predictions are pooled across the outer folds.
  3. Metrics on pooled out-of-fold predictions: accuracy (proposal metric),
     balanced accuracy, macro-F1, quadratic weighted kappa, per-class recall,
     predicted-class shares, vs the majority-class baseline. Intervals come from
     a bootstrap over SLIDES (not frames).
  4. Validity gate (gate.json): balanced acc > 0.45, QWK > 0.2, no predicted
     class > 80%, slide-bootstrap lower bound of balanced acc > 1/3 (chance for
     3 classes), and score-1 recall > 0 when slide A12 is held out (otherwise the
     head has only learned slide identity: 19 of 23 training score-1 frames are
     in A12).
  5. Final head: C chosen by LOSO over all training slides, fit on all training
     frames, saved with provenance. Held-out Aperio features (--heldout-features)
     are scored ONCE at the end as a sanity check only (collapse check), never
     used for fitting or selection.

Usage
-----
    python fit_atypia_probe.py \
        --features /datasets/mhoosen/stain-norm/classifier/features/uni2h_train.npz \
        --heldout-features /datasets/mhoosen/stain-norm/classifier/features/uni2h_test.npz \
        --out /datasets/mhoosen/stain-norm/classifier/atypia_probe_uni2h
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import defaultdict
from pathlib import Path

import numpy as np

CLASSES = [1, 2, 3]
C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
GATE = {"balanced_acc_min": 0.45, "qwk_min": 0.2, "max_class_share": 0.80, "chance_balanced_acc": 1 / 3}


# ----------------------------- pure helpers -----------------------------

def frame_table(npz):
    """tile-level npz -> frame-level (slides, frames, labels, X[n_frames, dim])."""
    slide, frame, label, emb = npz["slide"], npz["frame"], npz["label"], npz["emb"]
    groups = defaultdict(list)
    for i, (s, f) in enumerate(zip(slide, frame)):
        groups[(str(s), str(f))].append(i)
    keys = sorted(groups)
    X = np.stack([emb[groups[k]].mean(axis=0) for k in keys])
    y = np.array([int(label[groups[k][0]]) for k in keys])
    return np.array([k[0] for k in keys]), np.array([k[1] for k in keys]), y, X


def metrics(y, pred):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score, recall_score
    y, pred = np.asarray(y), np.asarray(pred)
    out = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0)),
        "qwk": float(cohen_kappa_score(y, pred, labels=CLASSES, weights="quadratic")) if len(set(y)) > 1 else float("nan"),
    }
    present = [c for c in CLASSES if (y == c).any()]
    rec = recall_score(y, pred, labels=present, average=None, zero_division=0)
    out["recall"] = {int(c): float(r) for c, r in zip(present, rec)}
    out["pred_share"] = {int(c): float((pred == c).mean()) for c in CLASSES}
    maj = max(set(y.tolist()), key=y.tolist().count)
    out["majority_class"] = int(maj)
    out["majority_accuracy"] = float((y == maj).mean())
    return out


def fit_head(Xtr, ytr, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, solver="lbfgs")
    clf.fit(sc.transform(Xtr), ytr)
    return sc, clf


def predict(sc, clf, X):
    return clf.predict(sc.transform(X))


def loso_oof(X, y, slides, C):
    """leave-one-slide-out out-of-fold predictions for one C."""
    pred = np.zeros(len(y), dtype=int)
    for s in sorted(set(slides)):
        te = slides == s
        sc, clf = fit_head(X[~te], y[~te], C)
        pred[te] = predict(sc, clf, X[te])
    return pred


def select_C(X, y, slides):
    """C maximising pooled LOSO balanced accuracy on (X, y, slides); ties -> smaller C."""
    from sklearn.metrics import balanced_accuracy_score
    best, best_score = None, -1.0
    for C in C_GRID:
        score = balanced_accuracy_score(y, loso_oof(X, y, slides, C))
        if score > best_score + 1e-12:
            best, best_score = C, score
    return best, best_score


def slide_bootstrap(y, pred, slides, n_boot=2000, seed=0):
    """CI over slides: resample slides with replacement, recompute metrics."""
    from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(slides)))
    idx_by = {s: np.where(slides == s)[0] for s in uniq}
    ba, qk, ac = [], [], []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[s] for s in pick])
        yy, pp = y[idx], pred[idx]
        ac.append((yy == pp).mean())
        ba.append(balanced_accuracy_score(yy, pp))
        if len(set(yy)) > 1:
            qk.append(cohen_kappa_score(yy, pp, labels=CLASSES, weights="quadratic"))
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"accuracy_ci": q(ac), "balanced_accuracy_ci": q(ba), "qwk_ci": q(qk)}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# --------------------------------- main ---------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="P2-14: fit + validate the atypia probe head.")
    ap.add_argument("--features", required=True, help="Training-split features .npz.")
    ap.add_argument("--heldout-features", default=None,
                    help="Held-out Aperio features .npz; scored once as a sanity check, never used to fit/select.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tr = np.load(args.features)
    slides, frames, y, X = frame_table(tr)
    print(f"Training frames: {len(y)} from {len(set(slides))} slides; "
          f"labels {dict(zip(*np.unique(y, return_counts=True)))}")

    # ---- nested leave-one-slide-out CV ----
    oof = np.zeros(len(y), dtype=int)
    fold_rows, chosen_C = [], {}
    for s in sorted(set(slides)):
        te = slides == s
        C, inner = select_C(X[~te], y[~te], slides[~te])
        sc, clf = fit_head(X[~te], y[~te], C)
        oof[te] = predict(sc, clf, X[te])
        chosen_C[s] = C
        present_train = sorted(set(y[~te].tolist()))
        fold_rows.append({"held_out_slide": s, "n_frames": int(te.sum()), "chosen_C": C,
                          "inner_balanced_acc": round(inner, 4),
                          "classes_in_heldout": sorted(set(y[te].tolist())),
                          "classes_in_train": present_train,
                          "fold_accuracy": round(float((oof[te] == y[te]).mean()), 4)})
        print(f"  fold {s}: C={C:g} acc={fold_rows[-1]['fold_accuracy']:.3f} "
              f"heldout_classes={fold_rows[-1]['classes_in_heldout']}")

    cv = metrics(y, oof)
    cv["slide_bootstrap"] = slide_bootstrap(y, oof, slides, args.n_bootstrap, args.seed)
    a12 = slides == "A12"
    a12_recall1 = float((oof[a12 & (y == 1)] == 1).mean()) if (a12 & (y == 1)).any() else None

    gate = {
        "balanced_acc_gt_0.45": cv["balanced_accuracy"] > GATE["balanced_acc_min"],
        "qwk_gt_0.2": bool(cv["qwk"] > GATE["qwk_min"]),
        "no_class_share_gt_0.80": max(cv["pred_share"].values()) <= GATE["max_class_share"],
        "slide_bootstrap_ba_lo_gt_chance": cv["slide_bootstrap"]["balanced_accuracy_ci"][0] > GATE["chance_balanced_acc"],
        "a12_score1_recall_gt_0": (a12_recall1 is not None and a12_recall1 > 0),
    }
    gate["PASS"] = all(gate.values())
    print("\nPooled out-of-fold CV:", json.dumps({k: v for k, v in cv.items() if k != 'slide_bootstrap'}, indent=1))
    print("Slide-bootstrap CIs:", cv["slide_bootstrap"])
    print("A12 score-1 recall:", a12_recall1)
    print("GATE:", json.dumps(gate, indent=1))

    # ---- final head ----
    C_final, loso_ba = select_C(X, y, slides)
    sc, clf = fit_head(X, y, C_final)
    np.savez(out / "head.npz", coef=clf.coef_, intercept=clf.intercept_, classes=clf.classes_,
             scaler_mean=sc.mean_, scaler_scale=sc.scale_, C=np.array(C_final))
    print(f"\nFinal head: C={C_final:g} (LOSO balanced acc {loso_ba:.3f}) -> {out / 'head.npz'}")

    # ---- one-time held-out sanity check ----
    held = None
    if args.heldout_features:
        hs, hf, hy, hX = frame_table(np.load(args.heldout_features))
        hp = predict(sc, clf, hX)
        held = metrics(hy, hp)
        held["per_frame"] = [{"slide": str(a), "frame": str(b), "true": int(c), "pred": int(d)}
                             for a, b, c, d in zip(hs, hf, hy, hp)]
        held["note"] = "raw-Aperio sanity check only; never used for fitting or selection"
        held["collapse_check_max_class_share_le_0.80"] = max(held["pred_share"].values()) <= GATE["max_class_share"]
        print("Held-out raw-Aperio sanity:", json.dumps({k: v for k, v in held.items() if k != 'per_frame'}, indent=1))

    result = {
        "cv": cv, "folds": fold_rows, "a12_score1_recall": a12_recall1, "gate": gate,
        "final_C": C_final, "heldout_sanity": held,
        "provenance": {
            "features": str(args.features), "features_sha256": sha256_file(args.features),
            "heldout_features": str(args.heldout_features) if args.heldout_features else None,
            "head_sha256": sha256_file(out / "head.npz"), "c_grid": C_GRID, "gate_thresholds": GATE,
            "python": platform.python_version(), "numpy": np.__version__,
            "n_bootstrap": args.n_bootstrap, "seed": args.seed,
        },
    }
    try:
        import sklearn
        result["provenance"]["sklearn"] = sklearn.__version__
    except ImportError:
        pass
    (out / "probe_results.json").write_text(json.dumps(result, indent=2))
    print(f"Wrote {out / 'probe_results.json'}")


if __name__ == "__main__":
    main()
