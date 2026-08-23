#!/bin/bash
# =====================================================================
# score_p1_10_ddim_vs_lcm_diag.slurm -- P1-12 few-step DDIM vs LCM diagnostic:
# score eval/p1_10_ddim_vs_lcm_diag (job 45654) via score_p1_10_ablation.py
# on stampede (CPU). One-off scorer since this output dir sits outside the
# eval/p1_10_ddim_inversion/<tag> convention score_p1_10_ddim_inversion.slurm
# assumes. See tickets/PHASE1-TICKETS.md P1-12.
#
# Submit with no args:
#   sbatch score_p1_10_ddim_vs_lcm_diag.slurm
# =====================================================================

#SBATCH --job-name=score_ddim_vs_lcm
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
EVAL_DIR=${DATA_ROOT}/eval/p1_10_ddim_vs_lcm_diag
OUT_DIR=${DATA_ROOT}/eval/p1_10_ddim_vs_lcm_diag_summary

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Eval dir: $EVAL_DIR"
echo "Out: $OUT_DIR"
echo "=================================================================="

cd "${CODE_DIR}/src/eval"
python score_p1_10_ablation.py --eval-dirs "${EVAL_DIR}" --out "${OUT_DIR}"

echo "------------------------------------------------------------------"
echo "Scoring done. Summary in ${OUT_DIR}"
