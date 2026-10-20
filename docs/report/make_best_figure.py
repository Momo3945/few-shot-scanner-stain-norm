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
fig, axs = plt.subplots(len(crops), 3, figsize=(7.2, 2.5*len(crops)))
for r,(s,f,x,y) in enumerate(crops):
    cid = f"{s}_{f}_x{x}_y{y}"
    row = m[(m.slide==s)&(m.frame==f)&(m.x==x)&(m.y==y)].iloc[0]
    src = Image.open(TIFF.format(s=s,f=f)).convert("RGB").crop((x,y,x+512,y+512))
    ims = [src, Image.open(B/f"f3_{cid}.png"), Image.open(B/f"ref_{cid}.png")]
    ttl = ["Raw Aperio (source)", "SD1.5 pipeline (P1-11 + F3)", "Real Hamamatsu (target)"]
    for c in range(3):
        ax = axs[r,c]; ax.imshow(ims[c]); ax.axis("off")
        if r == 0: ax.set_title(ttl[c], fontsize=8)
    axs[r,0].text(0.01,0.02,f"{s} {f}  LAB {row.b_lab:.1f}", transform=axs[r,0].transAxes, fontsize=7, color="w", bbox=dict(fc="k",alpha=.55,pad=1.5))
    axs[r,1].text(0.01,0.02,f"LAB {row.lab_total:.1f} ($\Delta$={row.dlab:+.1f})  SSIM {row.ssim:.2f}", transform=axs[r,1].transAxes, fontsize=7, color="w", bbox=dict(fc="k",alpha=.55,pad=1.5))
plt.tight_layout(pad=0.3); fig.savefig("docs/report/figures/best_cases_sd15.jpg", dpi=130)
