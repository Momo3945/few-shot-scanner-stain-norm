#!/bin/bash
# =====================================================================
# score_p1_10_sweep.slurm -- score one P1-10 strength-sweep output dir
# (eval/p1_10_sweep_s<NN>) via score_p1_10_ablation.py, on stampede (CPU).
#
# Submit with:  STRENGTH_TAG   (e.g. s020, s030, s040, s070 -- matches the
#                                OUT_DIR suffix infer_p1_10_strength_sweep.slurm
#                                writes to)
#   sbatch score_p1_10_sweep.slurm s020
# =====================================================================

#SBATCH --job-name=score_p1_10_sweep
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

STRENGTH_TAG=${1:?"Usage: sbatch score_p1_10_sweep.slurm STRENGTH_TAG (e.g. s020)"}
EVAL_DIR=${DATA_ROOT}/eval/p1_10_sweep_${STRENGTH_TAG}
OUT_DIR=${DATA_ROOT}/eval/p1_10_sweep_${STRENGTH_TAG}_summary

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Eval dir=$EVAL_DIR"
echo "=================================================================="

cd "${CODE_DIR}/src/eval"
python score_p1_10_ablation.py --eval-dirs "${EVAL_DIR}" --out "${OUT_DIR}"

echo "------------------------------------------------------------------"
echo "Scoring done. Summary in ${OUT_DIR}"
