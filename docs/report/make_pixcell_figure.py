"""PixCell embedding-only generation vs source and registered target (PXC-02 Part B), A06_00A (0,0), 256px.
Target = top-left 256px of the registered Hamamatsu reference (same Aperio grid)."""
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
Image.MAX_IMAGE_PIXELS = None
src = Image.open("data/mitos/mitos_atypia_2014_testing_aperio/A06/frames/x20/A06_00A.tiff").convert("RGB").crop((0,0,256,256))
gen = Image.open("docs/report/cluster_raw/pixcell_A06_00A_x0_y0.png").convert("RGB")
tgt = Image.open("docs/report/cluster_raw/real_hamamatsu_A06_00A_x0_y0.png").convert("RGB").crop((0,0,256,256))
print(gen.size)
fig, axs = plt.subplots(1,3,figsize=(7.2,2.8))
for ax,(t,im) in zip(axs,[("Raw Aperio (source)",src),("PixCell, UNI2-h embedding only",gen.resize((256,256))),("Real Hamamatsu (target)",tgt)]):
    ax.imshow(im); ax.set_title(t,fontsize=8); ax.axis("off")
plt.tight_layout(pad=0.3); fig.savefig("docs/report/figures/pixcell_floor.jpg",dpi=130)
