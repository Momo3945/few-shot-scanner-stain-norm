"""Build report figures: downscaled copies of existing panels + a summary chart.
Run from repo root: python docs/report/make_figures.py"""
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("docs/results"); OUT = Path("docs/report/figures")
PANELS = {
    "ladder_a08": R/"qualitative/comparison_a08_typical.png",
    "ladder_a06": R/"qualitative/comparison_a06_outlier.png",
    "p1_10_vs_baselines": R/"qualitative/comparison_a08_p1_10_vs_phase1_vs_baselines.png",
    # p1_11_h2a is rebuilt with readable labels in make_extra_figures.py
    "p1_15_decoder": R/"p1_15_alt_decoder/qualitative_panel.png",
    "p1_16_a06": R/"p1_16_source_fusion/images/comparison_p1_16_f3_process_a06.png",
    "p1_16_a08": R/"p1_16_source_fusion/images/comparison_p1_16_f3_process_a08.png",
    # sdxl_vs_sd15 is rebuilt from its tiles in make_extra_figures.py
    "cam_raw": R/"p2_10_camelyon17/images/camelyon/centre0_raw.png",
    "cam_norm": R/"p2_10_camelyon17/images/camelyon/centre0_normalised.png",
}
for k, p in PANELS.items():
    im = Image.open(p).convert("RGB"); im.thumbnail((1600, 1600))
    im.save(OUT/f"{k}.jpg", quality=85); print(k, im.size)

# Summary chart: recovery delta-LAB (ALL slides, positive = better), A2H, values from tickets/results READMEs.
rows = [("A0 frozen img2img",0.95),("A1 +ControlNet",0.73),("A2 r8 colour LoRA",1.72),("A3 CN+LoRA (s=.2)",1.59),
        ("A4 +LCM, 1 step",1.53),("A5 +hist LoRA, 1 step",1.93),("Macenko",9.04),("Reinhard",5.81),("HistMatch",3.23),
        ("P1-10 src-cond.",2.89),("P1-11 DDIM inv.",8.74),("P1-16 F3 fusion",7.81),
        ("SDXL A4",-5.60),("SDXL P3-07c",1.85),("SDXL P3-08 fusion",2.47)]
fig, ax = plt.subplots(figsize=(7,4.6))
cols = ["#888" if n.startswith(("Mac","Rein","Hist")) else ("#c0504d" if v<0 else "#2a6f97") for n,v in rows]
ax.barh([n for n,_ in rows][::-1],[v for _,v in rows][::-1],color=cols[::-1])
ax.axvline(0,color="k",lw=.8); ax.set_xlabel(r"Recovery $\Delta_{LAB}$ (all slides, A$\to$H; higher is better)")
ax.text(0.99,0.02,"grey = classical baselines",transform=ax.transAxes,ha="right",fontsize=8)
plt.tight_layout(); fig.savefig(OUT/"summary_delta_lab.pdf"); print("chart ok")
