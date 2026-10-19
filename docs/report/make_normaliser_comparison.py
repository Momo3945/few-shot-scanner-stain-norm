"""Normaliser comparison for the paper: A08 (typical) and A06 (outlier), crop (0,0) of frame 00A, A->H.

Columns: Aperio source | Macenko | Reinhard | Histogram matching | ParamNet | StainGAN | ours (P1-11 DDIM
inversion + P1-16 F3 fusion, seed 0) | real Hamamatsu (registered). Per-tile LAB Wasserstein and SSIM vs the
target come from each method's scored eval_per_crop.csv (cluster copies in cluster_raw/compare/
per_crop_scores.csv; ours from docs/results/p1_16_source_fusion), not recomputed here.

Inputs fetched from /datasets/mhoosen/stain-norm/eval/ into docs/report/cluster_raw/compare/ and
cluster_raw/p1_18/ (source + target); every method's reference crop was checked against the target (max
|diff| <= 1 level). Run from repo root: python docs/report/make_normaliser_comparison.py"""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("docs/report/cluster_raw")
OUT = Path("docs/report/figures")
CROPS = [("A08_00A_x0_y0", "A08 (typical)"), ("A06_00A_x0_y0", "A06 (outlier)")]
METHODS = [("macenko", "Macenko"), ("reinhard", "Reinhard"), ("histogram_matching", "Hist. matching"),
           ("paramnet", "ParamNet"), ("staingan", "StainGAN")]

rgb = lambda p: np.asarray(Image.open(p).convert("RGB"))
sc = pd.read_csv(R / "compare/per_crop_scores.csv", header=None, names=["method", "slide", "lab", "ssim"])
sc = sc.set_index(["method", "slide"])
f3 = pd.read_csv("docs/results/p1_16_source_fusion/eval/f3_full_heldout/eval_per_crop.csv")
f3 = f3[(f3.frame == "00A") & (f3.x == 0) & (f3.y == 0)].set_index("slide")

fig, axs = plt.subplots(len(CROPS), 8, figsize=(14.0, 4.1))
for ri, (t, row_label) in enumerate(CROPS):
    s = t[:3]
    tiles = [("Aperio source", rgb(R / f"p1_18/p1_10_ddim_inversion/identity_full_heldout/reference/{t}.png"), "")]
    for key, name in METHODS:
        r = sc.loc[(key, s)]
        tiles.append((name, rgb(R / f"compare/{key}/outputs/{t}.png"), f"LAB {r.lab:.1f} | SSIM {r.ssim:.2f}"))
    r = f3.loc[s]
    tiles.append(("Ours (inversion + F3)",
                  rgb(R / f"compare/p1_16_fusion/f3_s8_b0.50_heldout_full_corrected/outputs/{t}_seed0_f3_s8_b0.50.png"),
                  f"LAB {r.lab_total:.1f} | SSIM {r.ssim:.2f}"))
    tiles.append(("Real Hamamatsu", rgb(R / f"p1_18/p1_15_alt_decoder/p1_18_d1_pn03_full_r2/reference/{t}.png"), "target"))
    for ci, (name, img, note) in enumerate(tiles):
        ax = axs[ri, ci]
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{name}\n{note}" if note else f"{name}\n ", fontsize=8.5, pad=3)
        if ci == 0:
            ax.set_ylabel(row_label, fontsize=9)
plt.tight_layout(pad=0.3, h_pad=0.8, w_pad=0.3)
fig.savefig(OUT / "normaliser_comparison.jpg", dpi=170); plt.close(fig)
print("wrote", OUT / "normaliser_comparison.jpg")
