#!/bin/bash
# =====================================================================
# score_p3_07b_ablation.slurm -- P3-07b: score the mandatory
# source-conditioning ablation (correct/zero/shuffled, jobs 47807/47808/
# 47809) via score_p1_10_ablation.py, mirroring P3-07's own ablation
# scoring exactly. On stampede (CPU). See tickets/PHASE3-TICKETS.md P3-07
# (P3-07b section).
#
# Submit with no args:
#   sbatch -J score_p3_07b_ablation score_p3_07b_ablation.slurm
# =====================================================================

#SBATCH --job-name=score_p3_07b_ablation
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

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "=================================================================="

cd "${CODE_DIR}/src/eval"
python score_p1_10_ablation.py \
    --eval-dirs "${DATA_ROOT}/eval/p3_07b_ablation_correct" \
                "${DATA_ROOT}/eval/p3_07b_ablation_zero" \
                "${DATA_ROOT}/eval/p3_07b_ablation_shuffled" \
    --out "${DATA_ROOT}/eval/p3_07b_ablation_summary"

echo "------------------------------------------------------------------"
echo "Scoring done. Summary in ${DATA_ROOT}/eval/p3_07b_ablation_summary"
