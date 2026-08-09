import pandas as pd, numpy as np

rungs = ['a0','a1','a2h_r4','a2h_r8','a3','a4','a5']
base = pd.read_csv('baseline/baseline_per_crop.csv').rename(columns={'aperio_slide':'slide','frame_id':'frame'})
base_key = base[['slide','frame','x','y','lab_total','ssim','tissue_frac']].rename(
    columns={'lab_total':'base_lab','ssim':'base_ssim'})

data = {}
for r in rungs:
    df = pd.read_csv(f'{r}/eval_per_crop.csv')
    df = df.merge(base_key, on=['slide','frame','x','y'], how='left')
    df['recovery'] = df['base_lab'] - df['lab_total']
    data[r] = df

print("="*70)
print("Q1: A06 recovery-delta-vs-strength trend + SSIM trajectory, per rung")
print("="*70)
for r in ['a3','a4','a5']:
    df = data[r]
    d = df[df.slide=='A06'].groupby('strength').agg(
        recovery=('recovery','mean'), ssim=('ssim','mean'), lab=('lab_total','mean')).reset_index()
    print(f"\n-- {r} A06 --")
    print(d.to_string(index=False))
    rec = d['recovery'].values
    d1 = np.diff(rec)
    d2 = np.diff(d1)
    print(f"first differences (0.3->0.4, 0.4->0.5): {d1}")
    print(f"second difference (curvature): {d2}")
    ssim = d['ssim'].values
    print(f"SSIM first differences: {np.diff(ssim)}")

print("\n" + "="*70)
print("Q2: Per-crop std dev per slide per rung (dispersion), + tissue_frac corr")
print("="*70)
for r in ['a3','a4','a5']:
    df = data[r]
    for s in [0.3, 0.5]:
        sub = df[df.strength==s]
        g = sub.groupby('slide').agg(mean_lab=('lab_total','mean'), std_lab=('lab_total','std'),
                                       mean_rec=('recovery','mean'), std_rec=('recovery','std'),
                                       mean_ssim=('ssim','mean'), std_ssim=('ssim','std'), n=('lab_total','size'))
        print(f"\n-- {r} strength={s} --")
        print(g.round(3).to_string())

print("\n-- Correlation: tissue_frac vs recovery/ssim, per rung/strength (A06 only, and pooled) --")
for r in ['a3','a4','a5']:
    df = data[r]
    for s in [0.3,0.5]:
        sub = df[(df.strength==s)]
        for scope,name in [(sub, 'ALL'), (sub[sub.slide=='A06'], 'A06')]:
            if len(scope) < 5: continue
            c_rec = scope['tissue_frac'].corr(scope['recovery'])
            c_ssim = scope['tissue_frac'].corr(scope['ssim'])
            print(f"{r} s={s} {name}: corr(tissue_frac, recovery)={c_rec:.3f}, corr(tissue_frac, ssim)={c_ssim:.3f}")

print("\n" + "="*70)
print("Q3: Pareto check across 3 strengths per rung/slide (recovery vs ssim)")
print("="*70)
for r in ['a3','a4','a5']:
    df = data[r]
    g = df.groupby(['slide','strength']).agg(recovery=('recovery','mean'), ssim=('ssim','mean')).reset_index()
    print(f"\n-- {r} --")
    print(g.round(3).to_string(index=False))

print("\n" + "="*70)
print("Q4: A5 vs A4 gap on typical slides (A08,A16) vs A06, across strength")
print("="*70)
a4 = data['a4'].groupby(['slide','strength']).agg(recovery=('recovery','mean')).reset_index()
a5 = data['a5'].groupby(['slide','strength']).agg(recovery=('recovery','mean')).reset_index()
m = a4.merge(a5, on=['slide','strength'], suffixes=('_a4','_a5'))
m['gap_a5_minus_a4'] = m['recovery_a5'] - m['recovery_a4']
print(m[m.slide.isin(['A06','A08','A09','A13','A16'])].round(3).to_string(index=False))
