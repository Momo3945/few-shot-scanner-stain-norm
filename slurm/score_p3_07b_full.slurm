#!/bin/bash
# =====================================================================
# score_p3_07b_full.slurm -- P3-07b: score the full 5-slide/3-seed
# held-out run (job 47846) via score_p1_10_ablation.py, mirroring
# score_p3_07.slurm exactly. On stampede (CPU). See
# tickets/PHASE3-TICKETS.md P3-07 (P3-07b section).
#
# Submit with no args:
#   sbatch --time=02:30:00 -J score_p3_07b_full score_p3_07b_full.slurm
# =====================================================================

#SBATCH --job-name=score_p3_07b_full
#SBATCH --partition=stampede
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:30:00
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
    --eval-dirs "${DATA_ROOT}/eval/p3_07b_full_heldout" \
    --out "${DATA_ROOT}/eval/p3_07b_full_heldout_summary"

echo "------------------------------------------------------------------"
echo "Scoring done. Summary in ${DATA_ROOT}/eval/p3_07b_full_heldout_summary"
