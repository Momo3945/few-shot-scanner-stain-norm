#!/usr/bin/env python3
"""
fit_atypia_v2.py -- P2-15 step 2: three candidate frame-level atypia models on
frozen UNI2-h features, all under nested leave-one-slide-out CV on the TRAINING
slides only, with pre-registered gates. See tickets/PHASE2-TICKETS.md P2-15.

Candidates (hyper-parameters fixed in advance; only the ones marked "nested" are
chosen by inner CV):
  M1  ordinal head (Frank-Hall: P(y>1), P(y>2) as two L2 logistic regressions,
      class-balanced) on mean-pooled x20 tile features. C chosen by nested CV.
  M2  gated-attention pooling over x20 tile features -> ordinal logits.
      FIXED: hidden 64, dropout 0.3, weight decay 1e-2, lr 1e-3, 100 full-batch
      epochs, class-balanced loss, 5 seeds (probabilities averaged).
  M3  two-stage: ridge from x40 subfield features to 4 criteria (nuclei_size,
      anisonucleosis, nucleoli_size, nuclei_contour), alpha chosen by nested CV;
      frame score from the 4 predicted criteria (out-of-fold within the training
      slides, so the score head never sees in-sample criteria predictions) via the
      same ordinal head.
Score gate (same as P2-14): balanced acc > 0.45, QWK > 0.2, no class > 80% of
predictions, slide-bootstrap lower bound of balanced acc > 1/3, score-1 recall > 0
on A12. Criteria gate (M3 regressor only): mean over the 4 criteria of the
SLIDE-CENTRED Spearman correlation (predicted vs true, both minus their slide
mean) > 0.15 with slide-bootstrap lower bound > 0 -- cannot be passed by
recognising the slide.
Held-out slides are NOT touched here (scored separately, once, only for a model
that passes the CV gates).
"""

from __future__ import annotations

import argparse
import json
import platform
from collections import defaultdict
from pathlib import Path

import numpy as np

import fit_atypia_probe as P

KEEP_CRIT = [0, 1, 2, 5]
CRIT_NAMES = ["nuclei_size", "anisonucleosis", "nucleoli_size", "nuclei_contour"]
C_GRID = P.C_GRID
ALPHA_GRID = [1e1, 1e2, 1e3, 1e4, 1e5]
CRIT_GATE = 0.15


# ------------------------------ ordinal head ------------------------------

def fh_fit(X, y, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    Z = sc.transform(X)
    heads = [LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(Z, (y > k).astype(int))
             for k in (1, 2)]
    return sc, heads


def fh_predict(model, X):
    sc, heads = model
    Z = sc.transform(X)
    p1, p2 = heads[0].predict_proba(Z)[:, 1], heads[1].predict_proba(Z)[:, 1]
    return np.where(p2 > 0.5, 3, np.where(p1 > 0.5, 2, 1))


def select_C_fh(X, y, slides):
    from sklearn.metrics import balanced_accuracy_score
    best, best_s = None, -1.0
    for C in C_GRID:
        pred = np.zeros(len(y), dtype=int)
        for s in sorted(set(slides)):
            te = slides == s
            pred[te] = fh_predict(fh_fit(X[~te], y[~te], C), X[te])
        sc = balanced_accuracy_score(y, pred)
        if sc > best_s + 1e-12:
            best, best_s = C, sc
    return best


# --------------------------------- M2: MIL ---------------------------------

def pad_bags(bags, d):
    n = max(len(b) for b in bags)
    X = np.zeros((len(bags), n, d), dtype=np.float32)
    M = np.zeros((len(bags), n), dtype=bool)
    for i, b in enumerate(bags):
        X[i, :len(b)] = b
        M[i, :len(b)] = True
    return X, M


def mil_fit_predict(train_bags, ytr, test_bags, seeds=(0, 1, 2, 3, 4), hidden=64, drop=0.3,
                    wd=1e-2, lr=1e-3, epochs=100):
    """Returns predicted scores (1/2/3) for test_bags, probabilities averaged over seeds."""
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler

    d = train_bags[0].shape[1]
    sc = StandardScaler().fit(np.concatenate(train_bags))
    tr = [sc.transform(b).astype(np.float32) for b in train_bags]
    te = [sc.transform(b).astype(np.float32) for b in test_bags]
    Xtr, Mtr = map(torch.from_numpy, pad_bags(tr, d))
    Xte, Mte = map(torch.from_numpy, pad_bags(te, d))
    ytr_t = torch.tensor(ytr)
    t = torch.stack([(ytr_t > 1).float(), (ytr_t > 2).float()], dim=1)
    cnt = np.bincount(ytr, minlength=4)[1:].astype(float)
    cw = torch.tensor([1.0 / max(cnt[c - 1], 1) for c in ytr], dtype=torch.float32)
    cw = cw / cw.mean()

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.proj = nn.Sequential(nn.Dropout(drop), nn.Linear(d, hidden), nn.ReLU())
            s.V, s.U, s.w = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden), nn.Linear(hidden, 1)
            s.out = nn.Sequential(nn.Dropout(drop), nn.Linear(hidden, 2))

        def forward(s, X, M):
            z = s.proj(X)
            a = s.w(torch.tanh(s.V(z)) * torch.sigmoid(s.U(z))).squeeze(-1)
            a = a.masked_fill(~M, float("-inf")).softmax(1).unsqueeze(-1)
            return s.out((a * z).sum(1))

    probs = []
    for seed in seeds:
        torch.manual_seed(seed)
        net = Net()
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
        for _ in range(epochs):
            net.train()
            opt.zero_grad()
            loss = (nn.functional.binary_cross_entropy_with_logits(net(Xtr, Mtr), t, reduction="none")
                    .mean(1) * cw).mean()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            probs.append(torch.sigmoid(net(Xte, Mte)).numpy())
    p = np.mean(probs, axis=0)
    return np.where(p[:, 1] > 0.5, 3, np.where(p[:, 0] > 0.5, 2, 1))


# --------------------------- M3: criteria regressor ---------------------------

def centred_spearman(pred, true, slides):
    """Spearman of (pred - slide mean) vs (true - slide mean), pooled."""
    from scipy.stats import spearmanr
    p, t = pred.astype(float).copy(), true.astype(float).copy()
    for s in set(slides):
        m = slides == s
        p[m] -= p[m].mean(); t[m] -= t[m].mean()
    if np.allclose(p, 0) or np.allclose(t, 0):
        return 0.0
    return float(spearmanr(p, t)[0])


def ridge_fit_predict(Xtr, Ytr, Xte, alpha):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    return Ridge(alpha=alpha).fit(sc.transform(Xtr), Ytr).predict(sc.transform(Xte))


def ridge_loso_oof(X, Y, valid, slides, alpha):
    """LOSO OOF predictions for ALL rows (training only on valid rows of other slides)."""
    oof = np.zeros((len(X), Y.shape[1]))
    for s in sorted(set(slides)):
        te = slides == s
        tr = (~te) & valid
        oof[te] = ridge_fit_predict(X[tr], Y[tr], X[te], alpha)
    return oof


def select_alpha(X, Y, valid, slides):
    best, best_s = None, -9.0
    for a in ALPHA_GRID:
        oof = ridge_loso_oof(X, Y, valid, slides, a)
        sc = np.mean([centred_spearman(oof[valid, j], Y[valid, j], slides[valid]) for j in range(Y.shape[1])])
        if sc > best_s + 1e-12:
            best, best_s = a, sc
    return best


def frame_mean(preds, sub_keys, frame_keys):
    d = defaultdict(list)
    for p, k in zip(preds, sub_keys):
        d[k].append(p)
    return np.stack([np.mean(d[k], axis=0) if k in d else np.full(preds.shape[1], np.nan) for k in frame_keys])


# ----------------------------------- main -----------------------------------

def gate_for(cv, boot, a12_recall):
    g = {
        "balanced_acc_gt_0.45": cv["balanced_accuracy"] > P.GATE["balanced_acc_min"],
        "qwk_gt_0.2": bool(cv["qwk"] > P.GATE["qwk_min"]),
        "no_class_share_gt_0.80": max(cv["pred_share"].values()) <= P.GATE["max_class_share"],
        "slide_bootstrap_ba_lo_gt_chance": boot["balanced_accuracy_ci"][0] > P.GATE["chance_balanced_acc"],
        "a12_score1_recall_gt_0": a12_recall is not None and a12_recall > 0,
    }
    g["PASS"] = all(g.values())
    return g


def parse_args():
    ap = argparse.ArgumentParser(description="P2-15: frame-level candidate models + gates.")
    ap.add_argument("--x20-features", required=True)
    ap.add_argument("--x40-features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", default="M1,M2,M3")
    return ap.parse_args()


def main():
    args = parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    npz = np.load(args.x20_features)
    slides, frames, y, Xmean = P.frame_table(npz)
    keys = [(str(s), str(f)) for s, f in zip(slides, frames)]
    # tile bags per frame, same order as keys
    bag_idx = defaultdict(list)
    for i, (s, f) in enumerate(zip(npz["slide"], npz["frame"])):
        bag_idx[(str(s), str(f))].append(i)
    bags = [npz["emb"][bag_idx[k]] for k in keys]
    print(f"{len(y)} frames, {len(set(slides))} slides")

    sub = np.load(args.x40_features)
    sub_slide, sub_frame = sub["slide"].astype(str), sub["frame"].astype(str)
    Ysub = sub["crit"][:, KEEP_CRIT].astype(float)
    valid = np.isfinite(Ysub).all(axis=1)
    sub_keys = list(zip(sub_slide, sub_frame))
    print(f"{len(sub_slide)} subfields, {int(valid.sum())} with the 4 criteria")

    results, wanted = {}, set(args.models.split(","))
    oof = {m: np.zeros(len(y), dtype=int) for m in ("M1", "M2", "M3")}
    crit_oof = np.zeros_like(Ysub)

    for s in sorted(set(slides)):
        te = slides == s
        tr = ~te
        if "M1" in wanted:
            C = select_C_fh(Xmean[tr], y[tr], slides[tr])
            oof["M1"][te] = fh_predict(fh_fit(Xmean[tr], y[tr], C), Xmean[te])
        if "M2" in wanted:
            tr_bags = [b for b, m in zip(bags, tr) if m]
            te_bags = [b for b, m in zip(bags, te) if m]
            oof["M2"][te] = mil_fit_predict(tr_bags, y[tr], te_bags)
        if "M3" in wanted:
            sub_te = sub_slide == s
            sub_tr = ~sub_te
            alpha = select_alpha(sub["emb"][sub_tr], Ysub[sub_tr], valid[sub_tr], sub_slide[sub_tr])
            crit_oof[sub_te] = ridge_fit_predict(sub["emb"][sub_tr & valid], Ysub[sub_tr & valid],
                                                 sub["emb"][sub_te], alpha)
            # training-frame criteria features must be OOF wrt the training slides
            inner = ridge_loso_oof(sub["emb"][sub_tr], Ysub[sub_tr], valid[sub_tr], sub_slide[sub_tr], alpha)
            tr_keys = [k for k, m in zip(keys, tr) if m]
            F_tr = frame_mean(inner, [k for k, m in zip(sub_keys, sub_tr) if m], tr_keys)
            te_keys = [k for k, m in zip(keys, te) if m]
            F_te = frame_mean(crit_oof[sub_te], [k for k, m in zip(sub_keys, sub_te) if m], te_keys)
            ok_tr, ok_te = np.isfinite(F_tr).all(1), np.isfinite(F_te).all(1)
            C = select_C_fh(F_tr[ok_tr], y[tr][ok_tr], slides[tr][ok_tr])
            model = fh_fit(F_tr[ok_tr], y[tr][ok_tr], C)
            pred = np.full(int(te.sum()), 2)
            pred[ok_te] = fh_predict(model, F_te[ok_te])
            oof["M3"][te] = pred
        print(f"  fold {s} done")

    a12 = slides == "A12"
    for m in ("M1", "M2", "M3"):
        if m not in wanted:
            continue
        cv = P.metrics(y, oof[m])
        boot = P.slide_bootstrap(y, oof[m], slides, args.n_bootstrap, args.seed)
        a12r = float((oof[m][a12 & (y == 1)] == 1).mean()) if (a12 & (y == 1)).any() else None
        results[m] = {"cv": cv, "slide_bootstrap": boot, "a12_score1_recall": a12r, "gate": gate_for(cv, boot, a12r)}
        print(f"\n{m}: acc {cv['accuracy']:.3f} (majority {cv['majority_accuracy']:.3f}) "
              f"bal-acc {cv['balanced_accuracy']:.3f} QWK {cv['qwk']:.3f} "
              f"pred shares {cv['pred_share']}  GATE PASS={results[m]['gate']['PASS']}")
        print("   ", results[m]["gate"])

    if "M3" in wanted:
        rng = np.random.default_rng(args.seed)
        per = {n: centred_spearman(crit_oof[valid, j], Ysub[valid, j], sub_slide[valid])
               for j, n in enumerate(CRIT_NAMES)}
        mean_rho = float(np.mean(list(per.values())))
        uniq = np.array(sorted(set(sub_slide[valid])))
        boots = []
        for _ in range(args.n_bootstrap):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([np.where((sub_slide == s) & valid)[0] for s in pick])
            # relabel duplicated slides so slide-centring treats each draw separately
            lab = np.concatenate([np.full(((sub_slide == s) & valid).sum(), i) for i, s in enumerate(pick)])
            boots.append(np.mean([centred_spearman(crit_oof[idx, j], Ysub[idx, j], lab)
                                  for j in range(len(CRIT_NAMES))]))
        ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        results["criteria_gate"] = {"per_criterion_within_slide_spearman": per, "mean": mean_rho,
                                    "slide_bootstrap_ci": ci,
                                    "PASS": bool(mean_rho > CRIT_GATE and ci[0] > 0)}
        print("\nCriteria gate (within-slide Spearman):", json.dumps(results["criteria_gate"], indent=1))

    results["provenance"] = {
        "x20_features": args.x20_features, "x40_features": args.x40_features,
        "x20_sha256": P.sha256_file(args.x20_features), "x40_sha256": P.sha256_file(args.x40_features),
        "python": platform.python_version(), "numpy": np.__version__, "n_bootstrap": args.n_bootstrap,
        "seed": args.seed, "mil_fixed": {"hidden": 64, "dropout": 0.3, "wd": 1e-2, "lr": 1e-3,
                                         "epochs": 100, "seeds": [0, 1, 2, 3, 4]},
        "c_grid": C_GRID, "alpha_grid": ALPHA_GRID, "gate_thresholds": P.GATE, "criteria_gate_min": CRIT_GATE,
        "n_candidates": 3,
    }
    (out / "v2_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out / 'v2_results.json'}")


if __name__ == "__main__":
    main()
