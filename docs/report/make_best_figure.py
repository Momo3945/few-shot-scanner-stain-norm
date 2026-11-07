"""Best-case panel for the SD1.5 final pipeline (P1-11 + F3 fusion, A2H).
Crops chosen by per-crop recovery delta-LAB (top crop per slide, SSIM >= 0.6 and no SSIM loss vs raw).
Run from repo root after fetching cluster_raw/best/ and cluster_raw/csv/f3_merged.csv."""
from pathlib import Path
import pandas as pd
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
Image.MAX_IMAGE_PIXELS = None
B = Path("docs/report/cluster_raw/best")
m = pd.read_csv("docs/report/cluster_raw/csv/f3_merged.csv")
TIFF = "data/mitos/mitos_atypia_2014_testing_aperio/{s}/frames/x20/{s}_{f}.tiff"
crops = [("A09","00B",0,0),("A16","09A",1027,0),("A08","00A",0,0),("A06","01C",1027,0)]
# Labels sit outside the tiles (row label = slide/frame, caption under each tile = its numbers) at
# readable sizes; overlaid 7 pt boxes were unreadable once the figure is scaled into the page.
# Wide layout: one column per crop, rows = source / ours / target, so the figure fills the page width.
fig, axs = plt.subplots(3, len(crops), figsize=(2.6*len(crops), 8.4), layout="constrained")
ttl = ["Raw Aperio (source)", "Ours (inversion + F3)", "Real Hamamatsu (target)"]
for c,(s,f,x,y) in enumerate(crops):
    cid = f"{s}_{f}_x{x}_y{y}"
    row = m[(m.slide==s)&(m.frame==f)&(m.x==x)&(m.y==y)].iloc[0]
    src = Image.open(TIFF.format(s=s,f=f)).convert("RGB").crop((x,y,x+512,y+512))
    ims = [src, Image.open(B/f"f3_{cid}.png"), Image.open(B/f"ref_{cid}.png")]
    caps = [f"LAB {row.b_lab:.1f}", f"LAB {row.lab_total:.1f} ($\\Delta$ = {row.dlab:+.1f}), SSIM {row.ssim:.2f}", ""]
    for r in range(3):
        ax = axs[r,c]; ax.imshow(ims[r]); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlabel(caps[r], fontsize=9.5, labelpad=3)
        if c == 0: ax.set_ylabel(ttl[r], fontsize=10)
    axs[0,c].set_title(f"{s} frame {f}", fontsize=11)
fig.savefig("docs/report/figures/best_cases_sd15.jpg", dpi=150)
