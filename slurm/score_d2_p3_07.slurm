#!/bin/bash
# =====================================================================
# score_d2_p3_07.slurm -- P3-07 diagnostic D2: score the no-LoRA inference
# run (d2_infer_no_lora_p3_07.slurm) via score_p1_10_ablation.py, then run
# d2_verify_p3_07.py to compare against D1's WITH-LoRA recovery numbers and
# print the Case A / Case B verdict. On stampede (CPU).
# See tickets/PHASE3-TICKETS.md P3-07, diagnostic D2.
#
# Submit with no args (after d2_infer_no_lora_p3_07.slurm has COMPLETED):
#   sbatch -J score_d2_p3_07 score_d2_p3_07.slurm
# =====================================================================

#SBATCH --job-name=score_d2_p3_07
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
EVAL_DIR=${DATA_ROOT}/eval/p3_07_d2_no_lora
SUMMARY_DIR=${DATA_ROOT}/eval/p3_07_d2_no_lora_summary
D1_SUMMARY=${DATA_ROOT}/eval/p3_07_d1_verify/d1_summary.csv

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Eval dir: ${EVAL_DIR}"
echo "=================================================================="

cd "${CODE_DIR}/src/eval"
python score_p1_10_ablation.py --eval-dirs "${EVAL_DIR}" --out "${SUMMARY_DIR}"

echo "------------------------------------------------------------------"
python d2_verify_p3_07.py \
    --no-lora-per-crop "${SUMMARY_DIR}/per_crop.csv" \
    --d1-summary "${D1_SUMMARY}" \
    --out "${SUMMARY_DIR}/d2_verdict.csv"

echo "------------------------------------------------------------------"
echo "D2 scoring + verdict done. See ${SUMMARY_DIR}"
