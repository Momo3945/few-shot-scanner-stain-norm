"""VAE self-reconstruction SSIM per held-out slide, against the ideal (SSIM = 1.0).

A lossless autoencoder would reproduce its input exactly (SSIM = 1.0, dotted line). No VAE comes
close: stock SD1.5 ~0.58, sd-vae-ft-mse ~0.62, our H&E decoder fine-tune (P1-18) ~0.65 on Aperio.
That gap is the "VAE ceiling" that caps every diffusion result in this project
(tickets/P1-18_vae_decoder_finetune.md).

Reads the P1-18 Stage 1 held-out scores (496 canonical held-out crops, deterministic .mode() encode,
same code path as P1-14; its stock/ft_mse rows reproduce P1-14 exactly). Two panels: Aperio and
Hamamatsu self-reconstruction. Add arms by extending LINES.
Run from repo root: python docs/report/make_vae_ceiling_figure.py"""
from pathlib import Path
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path("docs/results/p1_18_vae_finetune/stage1_heldout/per_crop.csv")
OUT = Path("docs/report/figures")
SLIDES = ["A06", "A08", "A09", "A13", "A16"]

# (vae key in per_crop.csv, label, colour, marker)
LINES = [
    ("stock", "Stock SD1.5 VAE", "#0072B2", "o"),
    ("ft_mse", "sd-vae-ft-mse", "#D55E00", "s"),
    ("d1_pn03_best", "H&E decoder fine-tune (P1-18)", "#009E73", "^"),
]

df = pd.read_csv(SRC)
x = list(range(len(SLIDES) + 1))
xlabels = SLIDES + ["ALL"]

fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.0), sharey=True)
for ax, dom, title in zip(axs, ("aperio", "hamamatsu"), ("Aperio crops", "Hamamatsu crops")):
    ax.axhline(1.0, color="#444444", ls=":", lw=2.0, zorder=1)
    ax.text(-0.2, 0.985, "lossless autoencoder (SSIM = 1.0)", va="top", ha="left", fontsize=8, color="#444444")
    ends = []
    for key, label, colour, marker in LINES:
        d = df[(df.vae == key) & (df.domain == dom)]
        ys = [d[d.slide == s].ssim.mean() for s in SLIDES] + [d.ssim.mean()]
        ax.plot(x, ys, color=colour, marker=marker, lw=2, ms=6, mfc=colour, mec="white", mew=1.2,
                label=label, zorder=3)
        ends.append(ys[-1])
    # End labels, pushed apart so close values (e.g. 0.779 / 0.757) do not overlap
    MIN_GAP, prev = 0.03, None
    for i in sorted(range(len(ends)), key=lambda i: ends[i]):
        y = ends[i] if prev is None else max(ends[i], prev + MIN_GAP)
        ax.text(len(SLIDES) + 0.15, y, f"{ends[i]:.3f}", va="center", ha="left", fontsize=8, color="#222222")
        prev = y
    ax.axvline(len(SLIDES) - 0.5, color="#bbbbbb", lw=0.8, zorder=0)
    for g in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):  # manual grid: no solid line under the dotted ideal at 1.0
        ax.axhline(g, color="#e5e5e5", lw=0.6, zorder=0)
    ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylim(0.35, 1.05); ax.set_xlim(-0.3, len(SLIDES) + 0.9)
    ax.set_xlabel("Held-out slide (A06 = colour-gap outlier)", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=9, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axs[0].set_ylabel("SSIM(input, reconstruction)", fontsize=8)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=len(LINES), fontsize=8, frameon=False)
fig.suptitle("VAE self-reconstruction ceiling: no encode-decode round trip reaches SSIM 1.0",
             fontsize=10, x=0.01, ha="left")

plt.tight_layout(pad=0.5, rect=(0, 0.07, 1, 0.95))
for ext, kw in (("png", {"dpi": 160}), ("pdf", {})):
    fig.savefig(OUT / f"vae_ceiling.{ext}", **kw)
print("wrote", OUT / "vae_ceiling.png")
