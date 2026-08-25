#!/bin/bash
# =====================================================================
# score_p3_06.slurm -- P3-06: score one or more eval/<tag> dirs (SDXL
# transfer of P1-10) via score_p1_10_ablation.py, on stampede (CPU).
# Generic across every P3-06 stage (smoke, full held-out, future ablation
# reruns) -- eval dirs are eval/<tag> directly, no path-prefix assumption
# (unlike score_p1_10_ddim_inversion.slurm's eval/p1_10_ddim_inversion/<tag>
# convention, which does not apply here). See tickets/PHASE3-TICKETS.md P3-06.
#
# Submit with:  SUMMARY_TAG TAG [TAG2 ...]
#   sbatch score_p3_06.slurm smoke_heldout p3_06_smoke_heldout
#   sbatch score_p3_06.slurm full p3_06_full_heldout
# =====================================================================

#SBATCH --job-name=score_p3_06
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

SUMMARY_TAG=${1:?"Usage: sbatch score_p3_06.slurm SUMMARY_TAG TAG [TAG2 ...]"}
shift
if [[ $# -lt 1 ]]; then
    echo "ABORT: need at least one TAG (eval/<tag> dir name)."
    exit 1
fi

EVAL_DIRS=()
for t in "$@"; do
    EVAL_DIRS+=("${DATA_ROOT}/eval/${t}")
done
OUT_DIR=${DATA_ROOT}/eval/p3_06_${SUMMARY_TAG}_summary

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
