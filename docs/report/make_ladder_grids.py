"""2x4 grids of the A0-A5 ladder (larger tiles than the original 1x8 strips).
Run from repo root: python docs/report/make_ladder_grids.py"""
from pathlib import Path
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

Q = Path("docs/results/qualitative"); OUT = Path("docs/report/figures")
TITLES = ["Raw Aperio (source)", "A0 frozen img2img", "A1 + ControlNet", "A2 + colour LoRA",
          "A3 ControlNet + LoRA", "A4 + LCM-LoRA", "A5 + hist. warm-start", "Real Hamamatsu (target)"]
FILES = ["source", "a0", "a1", "a2", "a3", "a4", "a5", "reference"]

for folder, name in [("a08_typical", "ladder_grid_a08"), ("a06_outlier", "ladder_grid_a06"),
                     ("a06_outlier_strength050", "ladder_grid_a06_s050")]:
    # constrained layout reserves room for each row's titles (tight_layout let row-2 titles overlap row-1 tiles)
    fig, axs = plt.subplots(2, 4, figsize=(7.2, 4.1), layout="constrained")
    for ax, f, t in zip(axs.flatten(), FILES, TITLES):
        ax.imshow(Image.open(Q / folder / f"{f}.png").convert("RGB")); ax.set_title(t, fontsize=7); ax.axis("off")
    fig.savefig(OUT / f"{name}.jpg", dpi=150); plt.close(fig); print(name)
