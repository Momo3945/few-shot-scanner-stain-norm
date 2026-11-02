"""Cross-check every Table I number (run: python report/crosscheck_table1.py) in report/sections/results.tex against docs/results/."""
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
R = str(HERE.parent / "docs" / "results") + "/"
tex = open(HERE / "sections" / "results.tex", encoding="utf8").read()
tab = tex[tex.index("\\label{tab:main}"):]
tab = tab[:tab.index("\\end{tabular}")]


def paper_row(key):
    line = [l for l in tab.splitlines() if key in l]
    assert len(line) == 1, (key, line)
    nums = re.findall(r"(?<![\w{])\d+\.\d+", line[0].replace("\\textbf{", " "))
    return [float(x) for x in nums[-6:]]


def from_summary(path, strength=None):
    d = pd.read_csv(R + path, dtype={"strength": str})
    if strength is not None:
        d = d[d.strength == strength]
    g = lambda s, c: float(d[d.scope == s][c].iloc[0])
    t = d[d.scope.isin(["A08", "A09", "A13", "A16"])]
    tw = lambda c: float((t[c] * t.n_crops).sum() / t.n_crops.sum())
    return [g("ALL", "lab_total"), tw("lab_total"), g("A06", "lab_total"),
            g("ALL", "ssim"), tw("ssim"), g("A06", "ssim")]


agg = pd.read_csv(R + "learned_baselines/aggregate_seeds_summary.csv").set_index("method")
def from_agg(m):
    r = agg.loc[m]
    return [r[f"{s}_{k}_mean"] for k in ("lab", "ssim") for s in ("all", "typical", "a06")]


rows = {
    "Macenko~": from_summary("classical_baselines/macenko/eval_summary.csv"),
    "Reinhard~": from_summary("classical_baselines/reinhard/eval_summary.csv"),
    "& Histogram matching": from_summary("classical_baselines/histogram_matching/eval_summary.csv"),
    "StainNet~": from_agg("stainnet"), "ParamNet~": from_agg("paramnet"), "StainGAN~": from_agg("staingan"),
    "A0 frozen": from_summary("phase1_ablation/a0/eval_summary.csv", "0.30"),
    "A1 + ControlNet": from_summary("phase1_ablation/a1/eval_summary.csv", "0.30"),
    "A2 + colour LoRA": from_summary("phase1_ablation/a2h_r8/eval_summary.csv", "0.30"),
    "A3 ControlNet": from_summary("phase1_ablation/a3/eval_summary.csv", "0.30"),
    "A4 + LCM": from_summary("phase1_ablation/a4/eval_summary.csv", "0.30"),
    "A5 + histo": from_summary("phase1_ablation/a5/eval_summary.csv", "0.30"),
    "img2img$^": from_summary("p1_10_source_conditioning/eval/full_heldout_summary/eval_summary_final.csv"),
    "+ DDIM inversion &": from_summary("p1_11_ddim_inversion/eval/full_heldout_summary/eval_summary_final.csv"),
    "F3 fusion": from_summary("p1_16_source_fusion/eval/f3_full_heldout/eval_summary.csv"),
}
for path, mode, key in [("p1_15_alt_decoder/eval/full_correct_summary/per_crop.csv", "correct_alt_decoder", "ft-mse} decoder"),
                        ("p1_18_vae_finetune/g13/per_crop.csv", "correct_alt_decoder", "fine-tuned decoder &")]:
    d = pd.read_csv(R + path); d = d[(d.source_mode == mode) & (d.seed == 0)]
    t, a = d[d.slide != "A06"], d[d.slide == "A06"]
    rows[key] = [d.lab_total.mean(), t.lab_total.mean(), a.lab_total.mean(), d.ssim.mean(), t.ssim.mean(), a.ssim.mean()]

bad = 0
for key, val in rows.items():
    p = paper_row(key)
    ok = all(abs(x - y) <= (0.0006 if x < 1.5 else 0.006) for x, y in zip(p, val))
    bad += not ok
    print("OK " if ok else "BAD", f"{key:24s} paper={p} data={[round(float(v), 3) for v in val]}")
print("mismatches:", bad, "of", len(rows), "(raw row not in docs/results: computed from cluster pairs/ baseline metrics)")
