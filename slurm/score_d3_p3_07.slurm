#!/bin/bash
# =====================================================================
# score_d3_p3_07.slurm -- P3-07 diagnostic D3: score all three
# rgb_canny/rgb_only/canny_only condition-mode runs (each independently,
# since all three write source_mode=correct in their manifests -- scoring
# them together in one score_p1_10_ablation.py call would pool them
# indistinguishably), then run d3_verify_p3_07.py to compare against D1's
# baseline and print the H1 interpretation. On stampede (CPU). Requires all
# three d3_infer_condition_split_p3_07.slurm jobs to have COMPLETED first.
# See tickets/PHASE3-TICKETS.md P3-07, D3.
#
# Submit with no args:
#   sbatch -J score_d3_p3_07 score_d3_p3_07.slurm
# =====================================================================

#SBATCH --job-name=score_d3_p3_07
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
D1_SUMMARY=${DATA_ROOT}/eval/p3_07_d1_verify/d1_summary.csv
OUT_DIR=${DATA_ROOT}/eval/p3_07_d3_summary

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "=================================================================="

cd "${CODE_DIR}/src/eval"
for MODE in rgb_canny rgb_only canny_only; do
    echo "-- scoring ${MODE} --"
    python score_p1_10_ablation.py \
        --eval-dirs "${DATA_ROOT}/eval/p3_07_d3_${MODE}" \
        --out "${DATA_ROOT}/eval/p3_07_d3_${MODE}_summary"
done

echo "------------------------------------------------------------------"
python d3_verify_p3_07.py \
    --rgb-canny-per-crop "${DATA_ROOT}/eval/p3_07_d3_rgb_canny_summary/per_crop.csv" \
    --rgb-only-per-crop "${DATA_ROOT}/eval/p3_07_d3_rgb_only_summary/per_crop.csv" \
    --canny-only-per-crop "${DATA_ROOT}/eval/p3_07_d3_canny_only_summary/per_crop.csv" \
    --d1-summary "${D1_SUMMARY}" \
    --out "${OUT_DIR}/d3_verdict.csv"

echo "------------------------------------------------------------------"
echo "D3 scoring + verdict done. See ${OUT_DIR}"
