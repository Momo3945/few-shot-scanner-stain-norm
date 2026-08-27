#!/bin/bash
# =====================================================================
# measure_1024_candidates.slurm -- one-off, read-only measurement: how many
# tissue-passing candidate crop positions exist across all 11 MITOS-ATYPIA-14
# training slides at crop=1024/768, various overlaps -- informs whether a
# 1024-native P3-06/SDXL training set needs more than A03 alone. Uses
# extract_pairs.py's own candidate_boxes()/tissue_fraction() unchanged, reads
# only mitos_atypia_train_aperio/ (Aperio side, already on the cluster).
# Writes no files, modifies nothing.
#
# Submit with no args:
#   sbatch measure_1024_candidates.slurm
# =====================================================================

#SBATCH --job-name=measure_1024
#SBATCH --partition=stampede
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm

python - << 'PYEOF'
import sys, os, glob
sys.path.insert(0, '/home-mscluster/mhoosen/stain-norm/src/data')
from extract_pairs import candidate_boxes, tissue_fraction
from PIL import Image
import numpy as np
Image.MAX_IMAGE_PIXELS = None

ROOT = '/datasets/mhoosen/stain-norm/mitos_atypia_train_aperio'
SLIDES = ['A03','A04','A05','A07','A10','A11','A12','A14','A15','A17','A18']

def frames_x20_local(slide):
    d = os.path.join(ROOT, slide, 'frames', 'x20')
    out = {}
    for p in sorted(glob.glob(os.path.join(d, '*.tiff'))):
        fid = os.path.basename(p).split('_', 1)[1].rsplit('.', 1)[0]
        out[fid.upper()] = p
    return out

# cache decoded arrays once, reuse across all (crop, overlap) configs
cache = {}
for slide in SLIDES:
    for fid, path in frames_x20_local(slide).items():
        img = Image.open(path).convert('RGB')
        cache[(slide, fid)] = np.asarray(img)
print(f"Cached {len(cache)} frames across {len(SLIDES)} slides.")

for crop, overlap in [(1024, 0.0), (1024, 0.25), (1024, 0.5), (768, 0.0)]:
    stride = max(1, int(round(crop * (1.0 - overlap))))
    grand_total = 0
    per_slide = {}
    for slide in SLIDES:
        slide_total = 0
        for (s, fid), arr in cache.items():
            if s != slide:
                continue
            H, W = arr.shape[:2]
            boxes = candidate_boxes(W, H, crop, stride)
            kept = sum(1 for (x, y, cw, ch) in boxes
                       if tissue_fraction(arr[y:y+ch, x:x+cw]) >= 0.30)
            slide_total += kept
        per_slide[slide] = slide_total
        grand_total += slide_total
    print(f"crop={crop} overlap={overlap} stride={stride}: total_candidates={grand_total}")
    print("  per-slide:", per_slide)
PYEOF
