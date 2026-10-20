"""Panels from images fetched off the cluster (docs/report/cluster_raw/).
Raw Aperio source tile is cropped locally from data/mitos at the same (x,y)=(0,0), 1024px.
Run from repo root: python docs/report/make_cluster_figures.py"""
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW = Path("docs/report/cluster_raw"); OUT = Path("docs/report/figures")
TIFF = "data/mitos/mitos_atypia_2014_testing_aperio/{s}/frames/x20/{s}_00A.tiff"

def aperio(slide):
    Image.MAX_IMAGE_PIXELS = None
    return Image.open(TIFF.format(s=slide)).convert("RGB").crop((0, 0, 1024, 1024))

def panel(slide, extra, name):
    c = f"{slide}_00A_x0_y0"
    tiles = [("Raw Aperio", aperio(slide)),
             ("SDXL 1024, 50 pairs (P3-07)", Image.open(RAW/f"sdxl1024_50pair_{c}.png")),
             ("SDXL 1024, 144 pairs (P3-07c)", Image.open(RAW/f"sdxl1024_144pair_{c}.png")),
             ("SDXL + F3 fusion (P3-08)", Image.open(RAW/f"sdxl_f3_fused_{c}.png"))]
    tiles += extra
    tiles.append(("Real Hamamatsu", Image.open(RAW/f"real_hamamatsu_{c}.png")))
    fig, axs = plt.subplots(1, len(tiles), figsize=(2.6*len(tiles), 3.0))
    for ax, (t, im) in zip(axs, tiles):
        ax.imshow(im.convert("RGB").resize((512, 512))); ax.set_title(t, fontsize=7); ax.axis("off")
    plt.tight_layout(pad=0.3); fig.savefig(OUT/name, dpi=130); plt.close(fig)

panel("A08", [], "sdxl_panel_a08.jpg")
panel("A06", [("SD3.5 probe (PR-02, s=0.30)", Image.open(RAW/"sd35_A06_00A_x0_y0.png"))], "sdxl_sd35_panel_a06.jpg")

# One labelled row: cam_raw_patch is the same centre-0 patch P2-10 used, so its SD1.5 output sits alongside
# (previously a separate unlabelled row stacked in LaTeX, which clipped this row's titles).
CAM = Path("docs/results/p2_10_camelyon17/images/camelyon")
tiles = [("Raw (512px)", Image.open(RAW/"cam_raw_patch.png")),
         ("SD1.5 operating point (P2-10)", Image.open(CAM/"centre0_normalised.png")),
         ("SDXL P3-07c unfused", Image.open(RAW/"cam_sdxl_unfused.png")),
         ("SDXL + F3 fused", Image.open(RAW/"cam_sdxl_fused.png"))]
fig, axs = plt.subplots(1, 4, figsize=(9.6, 2.65), layout="constrained")
for ax, (t, im) in zip(axs, tiles):
    ax.imshow(im.convert("RGB").resize((512, 512))); ax.set_title(t, fontsize=8); ax.axis("off")
fig.savefig(OUT/"camelyon_sdxl.jpg", dpi=130)
