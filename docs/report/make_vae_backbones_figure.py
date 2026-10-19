"""Cross-backbone VAE self-reconstruction ceiling (the slide-A3 comparison, updated with the P1-18 decoder).

SSIM(x, D(E(x))) with a deterministic mode encode. SD1.5 rows: 496 canonical held-out Aperio crops
(docs/results/p1_18_vae_finetune/stage1_heldout/summary.json, same code path as P1-14). SDXL and the SD3.5 /
PixCell 16-channel VAE: 8 A06 crops only (P3-08 and PXC-02 checks), so they are drawn lighter and labelled.
Run from repo root: python docs/report/make_vae_backbones_figure.py"""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("docs/report/figures")
s = json.load(open("docs/results/p1_18_vae_finetune/stage1_heldout/summary.json"))["summary"]
ap = lambda k: s[k]["aperio"]["ssim"]
# (label, value, n_crops, accent)
ROWS = [
    ("SDXL VAE (fp32)", 0.4924, 8, False),
    ("SD 1.5 stock VAE", ap("stock"), 496, False),
    ("SD 1.5 sd-vae-ft-mse", ap("ft_mse"), 496, False),
    ("SD 1.5 fine-tuned decoder (ours)", ap("d1_pn03_best"), 496, True),
    ("SD 3.5 / PixCell 16-ch VAE", 0.7732, 8, False),
]

fig, ax = plt.subplots(figsize=(3.5, 2.05))
for i, (lab, v, n, acc) in enumerate(ROWS):
    col = "#1f6fd1" if acc else "#8a8f98"
    ax.barh(i, v, height=0.62, color=col, alpha=1.0 if n == 496 else 0.45, zorder=2)
    ax.text(v + 0.012, i, f"{v:.3f}", va="center", fontsize=6.5, color="#222222")
ax.axvline(1.0, color="#444444", ls=":", lw=1.6, zorder=1)
ax.text(0.985, len(ROWS) - 0.35, "lossless", ha="right", va="bottom", fontsize=6, color="#444444")
ax.set_yticks(range(len(ROWS)))
ax.set_yticklabels([r[0] + ("" if r[2] == 496 else " [8 crops]") for r in ROWS], fontsize=6.5)
ax.set_xlim(0, 1.04); ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.tick_params(axis="x", labelsize=6)
ax.set_xlabel("Self-reconstruction SSIM (Aperio crops)", fontsize=6.5)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.set_ylim(-0.6, len(ROWS) - 0.1)
plt.tight_layout(pad=0.3)
fig.savefig(OUT / "vae_backbones.pdf"); fig.savefig(OUT / "vae_backbones.png", dpi=220); plt.close(fig)
print("wrote", OUT / "vae_backbones.pdf", {r[0]: round(r[1], 4) for r in ROWS})
