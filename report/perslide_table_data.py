"""Per-slide appendix table data (Table app-perslide) and the two checks quoted beside it: typical-slide recovery s.d. and the dE2000-SSIM rank correlation. Run: python report/perslide_table_data.py (reads docs/results, pairs/baseline_metrics, and the SDXL+F3 summary on the cluster, read-only)."""
import io
import subprocess
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

R = "D:/Research/Mitos/docs/results/"
SL = ["A06", "A08", "A09", "A13", "A16"]
TY = SL[1:]
base = pd.read_csv("D:/Research/Mitos/pairs/baseline_metrics/baseline_summary.csv").set_index("scope")
cnt = pd.read_csv(R + "classical_baselines/macenko/eval_summary.csv").set_index("scope").n_crops


def summ(path, strength=None):
    d = pd.read_csv(path, dtype={"strength": str})
    if strength is not None and "strength" in d:
        d = d[d.strength == strength]
    return d.set_index("scope")


def seeds(paths):
    ds = [summ(p) for p in paths]
    out = ds[0][["lab_total", "ssim", "psnr", "mae", "de2000_mean"]].copy()
    for c in out.columns:
        out[c] = sum(d[c] for d in ds) / len(ds)
    return out


rows = {}
w = lambda c: float((base.loc[SL, c] * cnt[SL]).sum() / cnt[SL].sum())
raw = base.loc[SL, ["lab_total", "ssim", "psnr", "mae"]].copy()
raw.loc["ALL"] = [w("lab_total"), w("ssim"), w("psnr"), w("mae")]
rows["Raw Aperio"] = raw
for k, n in [("macenko", "Macenko"), ("reinhard", "Reinhard"), ("histogram_matching", "Histogram matching")]:
    rows[n] = summ(R + f"classical_baselines/{k}/eval_summary.csv")
for k, n in [("stainnet", "StainNet"), ("paramnet", "ParamNet"), ("staingan", "StainGAN")]:
    rows[n] = seeds([R + f"learned_baselines/{k}{s}/eval_summary.csv" for s in ("", "_s1", "_s2")])
rows["A2 colour LoRA"] = summ(R + "phase1_ablation/a2h_r8/eval_summary.csv", "0.30")
rows["A3"] = summ(R + "phase1_ablation/a3/eval_summary.csv", "0.30")
rows["A4"] = summ(R + "phase1_ablation/a4/eval_summary.csv", "0.30")
rows["A5"] = summ(R + "phase1_ablation/a5/eval_summary.csv", "0.30")
rows["Source-cond. img2img"] = summ(R + "p1_10_source_conditioning/eval/full_heldout_summary/eval_summary_final.csv")
rows["+ DDIM inversion"] = summ(R + "p1_11_ddim_inversion/eval/full_heldout_summary/eval_summary_final.csv")
rows["Inversion + F3"] = summ(R + "p1_16_source_fusion/eval/f3_full_heldout/eval_summary.csv")
sx = subprocess.run(["ssh", "-o", "ConnectTimeout=20", "mhoosen@146.141.21.100",
                     "cat /datasets/mhoosen/stain-norm/eval/p3_08_fusion/f3_s8_b0.50/eval_summary.csv"],
                    capture_output=True, text=True).stdout
rows["SDXL 144 pairs + F3"] = pd.read_csv(io.StringIO(sx)).set_index("scope")

print(f"{'method':24s} " + " ".join(f"LAB{s[1:]:>4s}" for s in SL) + "   " + " ".join(f"SS{s[1:]:>4s}" for s in SL) + "   PSNR   MAE")
for n, d in rows.items():
    lab = " ".join(f"{d.loc[s, 'lab_total']:7.2f}" for s in SL)
    ss = " ".join(f"{d.loc[s, 'ssim']:6.3f}" for s in SL)
    print(f"{n:24s} {lab}   {ss}   {d.loc['ALL', 'psnr']:5.2f} {d.loc['ALL', 'mae']:5.2f}")

print("\nTypical-slide recovery s.d. (raw per-slide minus method):")
for n in ["Macenko", "Reinhard", "Histogram matching", "A3", "A4", "A5", "Source-cond. img2img", "+ DDIM inversion", "Inversion + F3", "ParamNet", "StainGAN"]:
    rec = np.array([base.loc[s, "lab_total"] - rows[n].loc[s, "lab_total"] for s in TY])
    print(f"  {n:22s} rec={np.round(rec, 2)}  sd(pop)={rec.std():.2f} sd(sample)={rec.std(ddof=1):.2f}")

print("\nDE2000 vs SSIM across Table I methods (ALL):")
names = [n for n in rows if n != "Raw Aperio" and "de2000_mean" in rows[n].columns]
de = [rows[n].loc["ALL", "de2000_mean"] for n in names]
ss = [rows[n].loc["ALL", "ssim"] for n in names]
print("  n =", len(names), " Spearman(dE2000, SSIM) =", round(spearmanr(de, ss).correlation, 3))
