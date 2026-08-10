#!/bin/bash
# =====================================================================
# score_baseline.slurm -- (re)score the RAW / do-nothing baseline (P2-01), CPU.
#
#   sbatch score_baseline.slurm
#
# No positional args -- always scores the fixed held-out inventory into
# pairs/baseline_metrics/. Safe to rerun (e.g. after a metrics.py change);
# overwrites baseline_per_crop.csv/baseline_summary.csv, does not touch
# heldout_frames.csv or any raw data.
# =====================================================================

#SBATCH --job-name=score_baseline
#SBATCH --partition=stampede
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail

CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_ROOT=/datasets/mhoosen/stain-norm
MITOS_ROOT=${DATA_ROOT}/mitos_heldout
HELDOUT=${DATA_ROOT}/pairs/heldout_frames.csv
OUT_DIR=${DATA_ROOT}/pairs/baseline_metrics

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import cv2, scipy, skimage" 2>/dev/null; then
    pip install --quiet opencv-python-headless scipy scikit-image tifffile
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Root=$MITOS_ROOT  Heldout=$HELDOUT  Out=$OUT_DIR"
echo "=================================================================="

cd "${CODE_DIR}"
python src/eval/metrics.py baseline \
    --root       "${MITOS_ROOT}" \
    --heldout    "${HELDOUT}" \
    --out        "${OUT_DIR}" \
    --crop 512 --tissue-thresh 0.30

echo "------------------------------------------------------------------"
echo "Baseline scoring done. baseline_summary.csv + baseline_per_crop.csv in ${OUT_DIR}"
