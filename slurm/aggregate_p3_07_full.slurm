#!/bin/bash
# =====================================================================
# aggregate_p3_07_full.slurm -- P3-07: per-slide/ALL/ALL_excl_outliers +
# recovery-delta aggregation for the SDXL-at-1024 full held-out run, via
# the existing (unmodified) aggregate_p1_10_full.py -- fully generic over
# backbone/resolution, confirmed by P1-10's and P3-06's own use of it.
# On stampede (CPU). See tickets/PHASE3-TICKETS.md P3-07.
#
# Submit with no args:
#   sbatch aggregate_p3_07_full.slurm
# =====================================================================

#SBATCH --job-name=agg_p3_07_full
#SBATCH --partition=stampede
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail

CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_ROOT=/datasets/mhoosen/stain-norm
PER_CROP=${DATA_ROOT}/eval/p3_07_full_heldout_summary/per_crop.csv
BASELINE=${DATA_ROOT}/pairs/baseline_metrics/baseline_summary.csv
OUT=${DATA_ROOT}/eval/p3_07_full_heldout_summary/eval_summary_final.csv

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Per-crop: $PER_CROP"
echo "Out: $OUT"
echo "=================================================================="

cd "${CODE_DIR}/src/eval"
python aggregate_p1_10_full.py --per-crop "${PER_CROP}" --baseline "${BASELINE}" --out "${OUT}"

echo "------------------------------------------------------------------"
echo "Aggregation done. Summary in ${OUT}"
