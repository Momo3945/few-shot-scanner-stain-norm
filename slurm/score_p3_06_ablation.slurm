#!/bin/bash
# =====================================================================
# score_p3_06_ablation.slurm -- P3-06: score the SDXL source-conditioning
# ablation (eval/p3_06_ablation_{correct,zero,shuffled}, jobs 46004/46005/
# 46006) via score_p1_10_ablation.py, on stampede (CPU). One-off scorer
# since these output dirs sit directly under eval/, not under the
# eval/p1_10_ddim_inversion/<tag> convention score_p1_10_ddim_inversion.slurm
# assumes. See tickets/PHASE3-TICKETS.md P3-06.
#
# Submit with no args:
#   sbatch score_p3_06_ablation.slurm
# =====================================================================

#SBATCH --job-name=score_p3_06_abl
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
EVAL_DIRS=(
    "${DATA_ROOT}/eval/p3_06_ablation_correct"
    "${DATA_ROOT}/eval/p3_06_ablation_zero"
    "${DATA_ROOT}/eval/p3_06_ablation_shuffled"
)
OUT_DIR=${DATA_ROOT}/eval/p3_06_ablation_summary

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Eval dirs: ${EVAL_DIRS[*]}"
echo "Out: $OUT_DIR"
echo "=================================================================="

cd "${CODE_DIR}/src/eval"
python score_p1_10_ablation.py --eval-dirs "${EVAL_DIRS[@]}" --out "${OUT_DIR}"

echo "------------------------------------------------------------------"
echo "Scoring done. Summary in ${OUT_DIR}"
