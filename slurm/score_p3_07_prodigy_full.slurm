#!/bin/bash
# =====================================================================
# score_p3_07_prodigy_full.slurm -- informal curiosity test: score the
# full 5-slide/3-seed held-out run (job 60962, COMPLETED 5:59:05, 1485/1485
# outputs) for the Prodigy-optimizer SDXL checkpoint via
# score_p1_10_ablation.py, mirroring score_p3_07b_full.slurm/
# score_p3_07c_full.slurm exactly. On stampede (CPU). NOT a ticketed
# experiment -- see src/train/train_p3_07_prodigy_curiosity.py's docstring.
#
# Submit with no args (after the full held-out inference has COMPLETED):
#   sbatch --time=02:30:00 -J score_p3_07_prodigy_full score_p3_07_prodigy_full.slurm
# =====================================================================

#SBATCH --job-name=score_p3_07_prodigy_full
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
    --eval-dirs "${DATA_ROOT}/eval/p3_07_prodigy_full_heldout" \
    --out "${DATA_ROOT}/eval/p3_07_prodigy_full_heldout_summary"

echo "------------------------------------------------------------------"
echo "Scoring done. Summary in ${DATA_ROOT}/eval/p3_07_prodigy_full_heldout_summary"
