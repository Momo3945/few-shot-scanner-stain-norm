#!/bin/bash
# =====================================================================
# score_p1_10_ddim_inversion.slurm -- P1-11: score one or more DDIM-inversion
# output dirs (eval/p1_10_ddim_inversion/<tag>) via score_p1_10_ablation.py,
# on stampede (CPU). No changes needed to score_p1_10_ablation.py -- the new
# inference script's manifest schema is a superset of infer_colour_
# translation.py's, and the scorer already tolerates extra columns.
#
# Submit with:  SUMMARY_TAG TAG [TAG2 ...]
#   # score a single identity run:
#   sbatch score_p1_10_ddim_inversion.slurm identity_inv100 identity_inv100
#
#   # score all four identity partial-inversion runs together (paired
#   # win-rate comparison across inversion fractions, per score_p1_10_
#   # ablation.py's own multi-eval-dir support):
#   sbatch score_p1_10_ddim_inversion.slurm identity_all \
#       identity_inv025 identity_inv050 identity_inv075 identity_inv100
#
#   # score the translate-mode source-conditioning ablation together:
#   sbatch score_p1_10_ddim_inversion.slurm translate_ablation \
#       translate_inv050_correct translate_inv050_shuffled translate_inv050_zero
# =====================================================================

#SBATCH --job-name=score_p1_10_ddim_inv
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

SUMMARY_TAG=${1:?"Usage: sbatch score_p1_10_ddim_inversion.slurm SUMMARY_TAG TAG [TAG2 ...]"}
shift
if [[ $# -lt 1 ]]; then
    echo "ABORT: need at least one TAG (eval/p1_10_ddim_inversion/<tag> dir name)."
    exit 1
fi

EVAL_DIRS=()
for t in "$@"; do
    EVAL_DIRS+=("${DATA_ROOT}/eval/p1_10_ddim_inversion/${t}")
done
OUT_DIR=${DATA_ROOT}/eval/p1_10_ddim_inversion/${SUMMARY_TAG}_summary

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
