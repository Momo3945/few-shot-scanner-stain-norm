"""P1-18 G1.2 qualitative panels for the report: A06 and A08 crop (0,0), seed 0.

Figure 1 (p1_18_g12_recon.jpg): self-reconstruction -- original | stock VAE | sd-vae-ft-mse | P1-18 decoder.
Figure 2 (p1_18_g12_translated.jpg): A->H translation of the SAME P1-11 latent through each decoder --
  Aperio source | real Hamamatsu | stock (P1-11) | ft_mse (P1-15) | P1-18 decoder.
Each slide gets a full-crop row and a zoom row (the 128 px window with the most texture in the source,
outlined on the full crops; zoom shown with nearest-neighbour upsampling so no detail is invented).
Per-tile numbers come from the scored per-crop CSVs, not recomputed here.

Inputs: docs/report/cluster_raw/p1_18/ (fetched from /datasets/mhoosen/stain-norm/eval/; the P1-14 tiles and
the identity-run crops were checked pixel-identical) + docs/results/p1_18_vae_finetune/ per-crop scores.
Run from repo root: python docs/report/make_p1_18_panels.py"""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

R = Path("docs/report/cluster_raw/p1_18")
RES = Path("docs/results/p1_18_vae_finetune")
OUT = Path("docs/report/figures")
CROPS = ["A06_00A_x0_y0", "A08_00A_x0_y0"]
Z = 128  # zoom window (px)

rgb = lambda p: np.asarray(Image.open(R / p).convert("RGB"))


def zoom_window(img, size=Z, stride=32):
    """Top-left of the size x size window with the highest greyscale std (most texture)."""
    g = img.astype(float).mean(2)
    best, pos = -1.0, (0, 0)
    for y in range(0, g.shape[0] - size + 1, stride):
        for x in range(0, g.shape[1] - size + 1, stride):
            s = g[y:y + size, x:x + size].std()
            if s > best:
                best, pos = s, (y, x)
    return pos


def draw(rows, cols, tiles, labels, name, suptitle):
    """tiles[slide] = list of images per column; labels[slide] = list of per-tile caption lines."""
    fig, axs = plt.subplots(2 * len(CROPS), cols, figsize=(2.1 * cols, 4.35 * len(CROPS)))
    for si, t in enumerate(CROPS):
        y0, x0 = zoom_window(tiles[t][0])
        for ci in range(cols):
            img = tiles[t][ci]
            ax = axs[2 * si, ci]
            ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
            ax.add_patch(Rectangle((x0, y0), Z, Z, fill=False, ec="white", lw=1.0))
            ax.set_title(f"{rows[ci]}\n{labels[t][ci]}", fontsize=7, pad=3)
            if ci == 0:
                ax.set_ylabel(f"{t[:3]} crop (0,0)", fontsize=8)
            az = axs[2 * si + 1, ci]
            az.imshow(img[y0:y0 + Z, x0:x0 + Z], interpolation="nearest")
            az.set_xticks([]); az.set_yticks([])
            if ci == 0:
                az.set_ylabel(f"{t[:3]} zoom {Z}px", fontsize=8)
    fig.suptitle(suptitle, fontsize=8.5, x=0.01, ha="left")
    plt.tight_layout(pad=0.4, h_pad=1.4, rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / name, dpi=150); plt.close(fig); print("wrote", OUT / name)


# ---- per-crop scores -------------------------------------------------------------------------------
rec = pd.read_csv(RES / "stage1_heldout/per_crop.csv")
rec = rec[rec.domain == "aperio"].set_index(["vae", "crop_id"])
g13 = pd.read_csv(RES / "g13/per_crop.csv").set_index(["source_mode", "seed", "crop_id"])
p15 = pd.read_csv(RES / "g13/p1_15_ref/per_crop.csv").set_index(["source_mode", "seed", "crop_id"])

# ---- Figure 1: self-reconstruction ---------------------------------------------------------------
tiles, labels = {}, {}
for t in CROPS:
    s = t[:3]
    strip = rgb(f"p1_14_stage_b_qualitative/qualitative/{s}_{t}_panel_orig_stock_ftmse.png")
    orig = rgb(f"p1_10_ddim_inversion/identity_full_heldout/reference/{t}.png")
    assert np.array_equal(orig, strip[:, :512]), f"{t}: P1-14 original tile is not the same crop"
    stock = rgb(f"p1_10_ddim_inversion/identity_full_heldout/outputs/{t}_seed0_vae_only.png")
    ftm = strip[:, 1024:1536]
    new = rgb(f"p1_18_g14/vae_only_d1_pn03/outputs/{t}_vae_only.png")
    tiles[t] = [orig, stock, ftm, new]
    off = lambda im: (im.astype(float) - orig.astype(float)).mean((0, 1))
    lab = ["input"]
    for v, im in (("stock", stock), ("ft_mse", ftm), ("d1_pn03_best", new)):
        r = rec.loc[(v, t)]
        lab.append(f"SSIM {r.ssim:.3f} | blue {off(im)[2]:+.1f}")
    labels[t] = lab
draw(["Original (Aperio)", "Stock SD1.5 VAE", "sd-vae-ft-mse", "P1-18 decoder"], 4, tiles, labels,
     "p1_18_g12_recon.jpg",
     "VAE self-reconstruction (encoder frozen in all three). SSIM vs input from the held-out benchmark;\n"
     "'blue' = mean blue-channel shift vs input (levels). White box = zoomed window.")

# ---- Figure 2: translated output (same latent, three decoders) ------------------------------------
tiles, labels = {}, {}
for t in CROPS:
    src = rgb(f"p1_10_ddim_inversion/identity_full_heldout/reference/{t}.png")
    tgt = rgb(f"p1_15_alt_decoder/p1_18_d1_pn03_full_r2/reference/{t}.png")
    stock = rgb(f"p1_15_alt_decoder/p1_18_d1_pn03_full_r2/outputs/{t}_seed0_correct_stock_decoder.png")
    ftm = rgb(f"p1_15_alt_decoder/full_correct/outputs/{t}_seed0_correct_alt_decoder.png")
    new = rgb(f"p1_15_alt_decoder/p1_18_d1_pn03_full_r2/outputs/{t}_seed0_correct_alt_decoder.png")
    tiles[t] = [src, tgt, stock, ftm, new]
    a = g13.loc[("correct_stock_decoder", 0, t)]
    b = p15.loc[("correct_alt_decoder", 0, t)]
    c = g13.loc[("correct_alt_decoder", 0, t)]
    labels[t] = ["model input", "target (registered)"] + [f"SSIM {r.ssim:.3f} | wLAB {r.wlab_mean:.1f}" for r in (a, b, c)]
draw(["Aperio source", "Real Hamamatsu", "Stock decoder (P1-11)", "ft_mse decoder (P1-15)", "P1-18 decoder"], 5,
     tiles, labels, "p1_18_g12_translated.jpg",
     "A->H translation, seed 0: one P1-11 latent per crop, decoded three ways.\n"
     "SSIM / windowed LAB vs real Hamamatsu (lower wLAB = closer colour). White box = zoomed window.")
