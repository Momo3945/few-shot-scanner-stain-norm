"""Panels from images fetched off the cluster (docs/report/cluster_raw/).
Raw Aperio source tile is cropped locally from data/mitos at the same (x,y)=(0,0), 1024px.
Run from repo root: python docs/report/make_cluster_figures.py"""
from pathlib import Path
import textwrap
from PIL import Image, ImageDraw
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW = Path("docs/report/cluster_raw"); OUT = Path("docs/report/figures")
TIFF = "data/mitos/mitos_atypia_2014_testing_aperio/{s}/frames/x20/{s}_00A.tiff"

def aperio(slide):
    Image.MAX_IMAGE_PIXELS = None
    return Image.open(TIFF.format(s=slide)).convert("RGB").crop((0, 0, 1024, 1024))

# Short one-line titles for the print-width panels, so the figure stays shallow.
SHORT = {"SDXL 1024, 50 pairs": "SDXL, 50 pairs", "SDXL 1024, 144 pairs": "SDXL, 144 pairs",
         "SDXL 144 pairs + F3 fusion": "144 pairs + F3", "SD3.5 probe (s=0.30)": "SD3.5 (s = 0.30)"}

def panel(slide, extra, name, width=None):
    """width (inches): draw at the printed width so 7 pt titles print at 7 pt; None keeps the old size."""
    c = f"{slide}_00A_x0_y0"
    tiles = [("Raw Aperio", aperio(slide)),
             ("SDXL 1024, 50 pairs", Image.open(RAW/f"sdxl1024_50pair_{c}.png")),
             ("SDXL 1024, 144 pairs", Image.open(RAW/f"sdxl1024_144pair_{c}.png")),
             ("SDXL 144 pairs + F3 fusion", Image.open(RAW/f"sdxl_f3_fused_{c}.png"))]
    tiles += extra
    tiles.append(("Real Hamamatsu", Image.open(RAW/f"real_hamamatsu_{c}.png")))
    if width is None:
        fig, axs = plt.subplots(1, len(tiles), figsize=(2.6*len(tiles), 3.0))
    else:  # constrained layout leaves room for the wrapped two-line titles
        fig, axs = plt.subplots(1, len(tiles), figsize=(width, width / len(tiles) + 0.16), layout="constrained")
    for ax, (t, im) in zip(axs, tiles):
        t = t if width is None else SHORT.get(t, t)
        ax.imshow(im.convert("RGB").resize((512, 512))); ax.set_title(t, fontsize=7); ax.axis("off")
    if width is None:
        plt.tight_layout(pad=0.3)
    fig.savefig(OUT/name, dpi=130 if width is None else 300); plt.close(fig)

panel("A08", [], "sdxl_panel_a08.jpg", width=7.16)  # main-text Fig. 6, drawn at print width
panel("A06", [("SD3.5 probe (s=0.30)", Image.open(RAW/"sd35_A06_00A_x0_y0.png"))], "sdxl_sd35_panel_a06.jpg", width=7.16)

# One labelled row: cam_raw_patch is the same centre-0 patch P2-10 used, so its SD1.5 output sits alongside
# (previously a separate unlabelled row stacked in LaTeX, which clipped this row's titles).
CAM = Path("docs/results/p2_10_camelyon17/images/camelyon")
tiles = [("Raw (512px)", Image.open(RAW/"cam_raw_patch.png")),
         ("SD1.5 ladder setting (A4)", Image.open(CAM/"centre0_normalised.png")),
         ("SDXL 144 pairs, unfused", Image.open(RAW/"cam_sdxl_unfused.png")),
         ("SDXL + F3 fused", Image.open(RAW/"cam_sdxl_fused.png"))]
fig, axs = plt.subplots(1, 4, figsize=(7.16, 2.05), layout="constrained")
for ax, (t, im) in zip(axs, tiles):
    ax.imshow(im.convert("RGB").resize((512, 512))); ax.set_title(t, fontsize=7.5); ax.axis("off")
fig.savefig(OUT/"camelyon_sdxl.jpg", dpi=300)
