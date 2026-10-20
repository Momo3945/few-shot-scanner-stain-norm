"""Extra report figures from cluster-fetched images (cluster_raw/more/) and local panels.
Run from repo root: python docs/report/make_extra_figures.py"""
from pathlib import Path
import numpy as np, scipy.io as sio
from PIL import Image
from scipy.ndimage import binary_erosion
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

Image.MAX_IMAGE_PIXELS = None
M = Path("docs/report/cluster_raw/more"); C = Path("docs/report/cluster_raw"); OUT = Path("docs/report/figures")
R = Path("docs/results")
TIFF = "data/mitos/mitos_atypia_2014_testing_aperio/A06/frames/x20/A06_00A.tiff"
rgb = lambda p: Image.open(p).convert("RGB")


def row(axs, tiles, fs=8):
    for ax, (t, im) in zip(axs, tiles):
        ax.imshow(im); ax.set_title(t, fontsize=fs); ax.axis("off")


def save(fig, name, dpi=130):
    if fig.get_layout_engine() is None:  # multi-row figures use layout="constrained" instead
        plt.tight_layout(pad=0.3)
    fig.savefig(OUT / name, dpi=dpi); plt.close(fig); print(name)


# 1. VAE reconstruction panels (already 3-panel strips: original / stock / ft-mse)
fig, axs = plt.subplots(2, 1, figsize=(7.2, 4.9))
for ax, f, t in zip(axs, ["vae_panel_a06.png", "vae_panel_a08.png"], ["A06", "A08"]):
    ax.imshow(rgb(M / f)); ax.axis("off"); ax.set_title(t + " crop (0,0)", fontsize=8)
save(fig, "vae_recon.jpg")

# 2. Round trip A06 (A -> H -> A), 512px
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), layout="constrained")  # tight_layout clipped the hat titles
row(axs, [("Original Aperio", rgb(M / "rt_original.png")), (r"Forward $\hat H$ (A$\to$H)", rgb(M / "rt_hhat.png")),
          (r"Round trip $\hat A$ (H$\to$A)", rgb(M / "rt_ahat.png"))])
save(fig, "roundtrip.jpg")

# 3. P1-13 LCM arms, A06 (0,0), 512px; target = top-left 512 of the 1024 registered Hamamatsu
src = Image.open(TIFF).convert("RGB").crop((0, 0, 512, 512))
tgt = rgb(C / "real_hamamatsu_A06_00A_x0_y0.png").crop((0, 0, 512, 512))
fig, axs = plt.subplots(1, 4, figsize=(9.6, 2.7))
row(axs, [("Raw Aperio", src), ("Generic LCM-LoRA", rgb(M / "p113_generic.png")),
          ("Task-specific LCM (P1-13)", rgb(M / "p113_taskspecific.png")), ("Real Hamamatsu", tgt)])
save(fig, "p1_13_lcm.jpg")

# 4. P1-17 differential diffusion smoke crop (training crop A03_00A_c024)
chg = Image.open(M / "dd_changemap.png").convert("L")
fig, axs = plt.subplots(1, 5, figsize=(11, 2.5))
row(axs, [("Source (Aperio)", rgb(M / "dd_source.png")), ("Change map", chg), ("P1-17 output", rgb(M / "dd_output.png")),
          ("P1-10 baseline", rgb(M / "dd_p110_baseline.png")), ("Real Hamamatsu", rgb(M / "dd_reference.png"))], fs=7)
axs[1].images[0].set_cmap("gray")
save(fig, "p1_17_diffdiff.jpg")

# 5. SDXL H2A: source = registered real Hamamatsu, target = raw Aperio
ham = rgb(C / "real_hamamatsu_A06_00A_x0_y0.png"); aper = rgb(M / "h2a_reference.png")
fig, axs = plt.subplots(2, 3, figsize=(7.2, 5.2), layout="constrained")
row(axs[0], [("Real Hamamatsu (source), 512px FOV", ham.crop((0, 0, 512, 512))), ("SDXL 512 H2A (P3-06)", rgb(M / "h2a_sdxl512.png")),
             ("Raw Aperio (target)", aper.crop((0, 0, 512, 512)))], fs=7)
row(axs[1], [("Real Hamamatsu (source), 1024px FOV", ham), ("SDXL 1024 H2A (P3-07)", rgb(M / "h2a_sdxl1024.png")), ("Raw Aperio (target)", aper)], fs=7)
save(fig, "sdxl_h2a.jpg")

# 6. Lizard: HoVer-Net on original vs normalised, boundary overlay on a 400px patch
g = sio.loadmat(M / "liz_gt.mat")["inst_map"]
mo, mn = np.load(M / "liz_mask_orig.npy"), np.load(M / "liz_mask_norm.npy")
io, inn = np.asarray(rgb(M / "liz_orig.png")), np.asarray(rgb(M / "liz_norm.png"))
y0, x0, s = 350, 300, 400
def ov(img, mask, col):
    o = img.copy(); m = mask.astype(bool); b = m & ~binary_erosion(m)
    o[b] = col; return o[y0:y0 + s, x0:x0 + s]
fig, axs = plt.subplots(1, 4, figsize=(9.6, 2.7))
row(axs, [("Original image", io[y0:y0 + s, x0:x0 + s]), ("Ground truth (Lizard)", ov(io, g > 0, (0, 255, 0))),
          ("HoVer-Net on original (A)", ov(io, mo, (255, 255, 0))), ("HoVer-Net on normalised (B)", ov(inn, mn, (0, 255, 255)))], fs=7)
save(fig, "lizard_overlay.jpg")

# 7. Text-to-image backbone colour defaults (6 images)
names = ["sd15_a2h", "sd15_h2a", "sdxl512_a2h", "sdxl512_h2a", "sdxl1024_a2h", "sdxl1024_h2a"]
fig, axs = plt.subplots(2, 3, figsize=(7.2, 5.2), layout="constrained")
for ax, n in zip(axs.T.flatten(), names):
    ax.imshow(rgb(R / f"p3_06_p3_07_h2a_extension/images/{n}/t2i_seed0.png").resize((384, 384))); ax.set_title(n, fontsize=7); ax.axis("off")
save(fig, "t2i_backbones.jpg")

# 8. Downscaled local panels: strength sweeps and A06 strength 0.50
for k, p in {"sweep_a3_06": "qualitative/strength_sweep_a3_06.png", "sweep_a4_06": "qualitative/strength_sweep_a4_06.png",
             "ladder_a06_s050": "qualitative/comparison_a06_strength050.png",
             "best_worst": "../assets/sample_outputs/comparison_best.png"}.items():
    im = rgb(R / p); im.thumbnail((1600, 1600)); im.save(OUT / f"{k}.jpg", quality=85); print(k, im.size)

# 9. A5 strength sweeps, rebuilt from the per-strength tiles (the cluster-rendered strips clip their left labels)
SW = R / "qualitative/strength_sweep"
for sl, tag in [("06", "A06 (outlier)"), ("08", "A08 (typical)")]:
    tiles = [("Aperio (input)", rgb(SW / f"source_{sl}.png"))]
    tiles += [(f"s = {s / 100:.2f} ({int(8 * s / 100)} step{'s' if int(8 * s / 100) > 1 else ''})", rgb(SW / f"a5_{sl}_{s:03d}.png"))
              for s in (20, 30, 40, 50, 70)]
    tiles.append(("Hamamatsu (ground truth)", rgb(SW / f"reference_{sl}.png")))
    fig, axs = plt.subplots(1, 7, figsize=(11, 1.95), layout="constrained")
    row(axs, tiles, fs=8)
    fig.suptitle(f"A5 strength sweep, {tag}, 8-step LCM", fontsize=8, x=0.005, ha="left")
    save(fig, f"sweep_a5_{sl}.jpg")

# 10. A08 at A4 (strength 0.20): SD1.5 vs SDXL with the classical baselines, rebuilt from the tiles
# (the cluster-rendered strip's title overlaps its column labels)
SX = R / "p3_04_sdxl_a4/images/sdxl_vs_sd15"
fig, axs = plt.subplots(1, 7, figsize=(11, 1.9), layout="constrained")
row(axs, [("Aperio (input)", rgb(SX / "A08_aperio_input.png")), ("A4, SD1.5", rgb(SX / "A08_sd15_output.png")),
          ("A4, SDXL", rgb(SX / "A08_sdxl_output.png")), ("Macenko", rgb(SX / "A08_macenko_output.png")),
          ("Reinhard", rgb(SX / "A08_reinhard_output.png")), ("Histogram matching", rgb(SX / "A08_histmatch_output.png")),
          ("Hamamatsu (ground truth)", rgb(SX / "A08_hamamatsu_ground_truth.png"))], fs=8)
save(fig, "sdxl_vs_sd15.jpg")

# 11. P1-11 H2A fraction fix (A08/A16), re-labelled. The cluster panel is a 2x4 grid of 512px tiles (rows at
# y=0 and y=522, 10px white gap) with tiny labels burned into a 209x27 black box at each tile's top-left, so
# every tile is cut to the same 484x484 window below that box and titled here instead.
P = rgb(R / "p1_11_ddim_inversion/qualitative_h2a_fraction_fix.png")
COLS = ["Raw Hamamatsu (source)", "Real Aperio (target)", "H2A output, f = 1.00", "H2A output, f = 0.25"]
fig, axs = plt.subplots(2, 4, figsize=(9.6, 5.0), layout="constrained")
for r, (y0, slide) in enumerate([(0, "A08"), (522, "A16")]):
    for c in range(4):
        x0 = 512 * c
        axs[r, c].imshow(P.crop((x0 + 14, y0 + 28, x0 + 498, y0 + 512)))
        axs[r, c].set_title(COLS[c], fontsize=10); axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
        for s in axs[r, c].spines.values():
            s.set_visible(False)
    axs[r, 0].set_ylabel(slide, fontsize=11)
save(fig, "p1_11_h2a.jpg", dpi=150)
