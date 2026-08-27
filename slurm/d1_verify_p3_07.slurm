#!/bin/bash
# =====================================================================
# d1_verify_p3_07.slurm -- P3-07 diagnostic D1: recompute the raw baseline
# at NATIVE 1024x1024 resolution (score_baseline.slurm hardcodes --crop 512,
# and that same 512 baseline_summary.csv is what aggregate_p3_07_full.slurm
# diffs the 1024 model output against -- this checks whether that resolution
# mismatch (H5) explains or contributes to P3-07's -5.60 recovery Δlab).
# Read-only: writes only to eval/p3_07_d1_verify/, touches no existing file.
# On stampede (CPU). See tickets/PHASE3-TICKETS.md P3-07, diagnostic D1.
#
# Submit with no args:
#   sbatch d1_verify_p3_07.slurm
# =====================================================================

#SBATCH --job-name=d1_verify_p3_07
#SBATCH --partition=stampede
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail

CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_ROOT=/datasets/mhoosen/stain-norm
MITOS_ROOT=${DATA_ROOT}/mitos_heldout
HELDOUT=${DATA_ROOT}/pairs/heldout_frames.csv
MANIFEST=${DATA_ROOT}/eval/p3_07_full_heldout/eval_manifest.csv
EVAL_DIR=${DATA_ROOT}/eval/p3_07_full_heldout
PER_CROP=${DATA_ROOT}/eval/p3_07_full_heldout_summary/per_crop.csv
BASELINE_512=${DATA_ROOT}/pairs/baseline_metrics/baseline_summary.csv
OUT=${DATA_ROOT}/eval/p3_07_d1_verify

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Manifest: $MANIFEST"
echo "Per-crop: $PER_CROP"
echo "Out: $OUT"
echo "=================================================================="

cd "${CODE_DIR}/src/eval"
python d1_verify_p3_07.py \
    --root "${MITOS_ROOT}" --heldout "${HELDOUT}" \
    --manifest "${MANIFEST}" --eval-dir "${EVAL_DIR}" \
    --per-crop "${PER_CROP}" --baseline-512 "${BASELINE_512}" \
    --out "${OUT}" --crop 1024 --n-sample 10 --seed 0

echo "------------------------------------------------------------------"
echo "D1 verification done. Summary in ${OUT}"
