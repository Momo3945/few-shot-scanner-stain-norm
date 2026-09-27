#!/bin/bash
# =====================================================================
# score_p3_07_prodigy_ablation.slurm -- informal curiosity test: score the
# mandatory source-conditioning ablation (correct/zero/shuffled, jobs
# 60813/60814/60815) for the Prodigy-optimizer SDXL checkpoint, via
# score_p1_10_ablation.py, mirroring P3-07/P3-07b/P3-07c's own ablation
# scoring exactly. On stampede (CPU). NOT a ticketed experiment -- see
# src/train/train_p3_07_prodigy_curiosity.py's docstring.
#
# Submit with no args:
#   sbatch -J score_p3_07_prodigy_ablation score_p3_07_prodigy_ablation.slurm
# =====================================================================

#SBATCH --job-name=score_p3_07_prodigy_ablation
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
    --eval-dirs "${DATA_ROOT}/eval/p3_07_prodigy_ablation_correct" \
                "${DATA_ROOT}/eval/p3_07_prodigy_ablation_zero" \
                "${DATA_ROOT}/eval/p3_07_prodigy_ablation_shuffled" \
    --out "${DATA_ROOT}/eval/p3_07_prodigy_ablation_summary"

echo "------------------------------------------------------------------"
echo "Scoring done. Summary in ${DATA_ROOT}/eval/p3_07_prodigy_ablation_summary"
