#!/bin/bash
# =====================================================================
# score_p1_11_ensemble.slurm -- score one or more self-ensembling output
# dirs (eval/p1_11_ensemble/<tag>) via score_p1_10_ablation.py, on stampede
# (CPU). No changes needed to the scorer -- each manifest row's source_mode
# column already carries "<source_mode>_<ensemble_role>" (e.g.
# "correct_member" / "correct_ensemble"), so the scorer's existing per-mode
# aggregation and paired-win-rate logic directly answers "does ensembling
# help" with no new code (same precedent as score_p1_15_alt_decoder.slurm).
# See tickets/PHASE1-TICKETS.md P1-11 "Future work" #2.
#
# Submit with:  SUMMARY_TAG TAG [TAG2 ...]
#   sbatch score_p1_11_ensemble.slurm eta05_n3 eta05_n3
# =====================================================================

#SBATCH --job-name=score_p1_11_ensemble
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

SUMMARY_TAG=${1:?"Usage: sbatch score_p1_11_ensemble.slurm SUMMARY_TAG TAG [TAG2 ...]"}
shift
if [[ $# -lt 1 ]]; then
    echo "ABORT: need at least one TAG (eval/p1_11_ensemble/<tag> dir name)."
    exit 1
fi

EVAL_DIRS=()
for t in "$@"; do
    EVAL_DIRS+=("${DATA_ROOT}/eval/p1_11_ensemble/${t}")
done
OUT_DIR=${DATA_ROOT}/eval/p1_11_ensemble/${SUMMARY_TAG}_summary

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
