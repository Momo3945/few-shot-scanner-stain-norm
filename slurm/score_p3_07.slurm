#!/bin/bash
# =====================================================================
# score_p3_07.slurm -- P3-07: score one or more eval/<tag> dirs (SDXL
# transfer of P1-10 at native 1024 resolution) via score_p1_10_ablation.py,
# on stampede (CPU). Mirrors score_p3_06.slurm exactly, separate file only
# to keep output naming (p3_07_*) unambiguous from P3-06's 512 results.
# See tickets/PHASE3-TICKETS.md P3-07.
#
# Submit with:  SUMMARY_TAG TAG [TAG2 ...]
#   sbatch score_p3_07.slurm ablation p3_07_ablation_correct p3_07_ablation_zero p3_07_ablation_shuffled
# =====================================================================

#SBATCH --job-name=score_p3_07
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

SUMMARY_TAG=${1:?"Usage: sbatch score_p3_07.slurm SUMMARY_TAG TAG [TAG2 ...]"}
shift
if [[ $# -lt 1 ]]; then
    echo "ABORT: need at least one TAG (eval/<tag> dir name)."
    exit 1
fi

EVAL_DIRS=()
for t in "$@"; do
    EVAL_DIRS+=("${DATA_ROOT}/eval/${t}")
done
OUT_DIR=${DATA_ROOT}/eval/p3_07_${SUMMARY_TAG}_summary

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
