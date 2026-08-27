#!/bin/bash
# =====================================================================
# score_p1_15_alt_decoder.slurm -- P1-15: score one or more alt-decoder
# output dirs (eval/p1_15_alt_decoder/<tag>) via score_p1_10_ablation.py,
# on stampede (CPU). No changes needed to the scorer -- each manifest row's
# source_mode column already carries "<source_mode>_<method>" (e.g.
# "correct_stock_decoder" / "correct_alt_decoder"), so the scorer's
# existing per-mode aggregation and paired-win-rate logic (grouped by that
# column, paired by shared crop_id) directly answers the Stage 1 go/no-go
# gate (SSIM_alt vs SSIM_stock) with no new code. See
# tickets/P1-15_p1_11_alternate_decoder.md and tickets/PHASE1-TICKETS.md P1-15.
#
# Submit with:  SUMMARY_TAG TAG [TAG2 ...]
#   # score the Stage 1 smoke test (correct source, A06+A08):
#   sbatch score_p1_15_alt_decoder.slurm smoke_correct smoke_correct
#
#   # score correct + shuffled together (source-conditioning sanity check
#   # comparison across all 4 arms in one paired win-rate table):
#   sbatch score_p1_15_alt_decoder.slurm smoke_all smoke_correct smoke_shuffled
# =====================================================================

#SBATCH --job-name=score_p1_15_alt_decoder
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

SUMMARY_TAG=${1:?"Usage: sbatch score_p1_15_alt_decoder.slurm SUMMARY_TAG TAG [TAG2 ...]"}
shift
if [[ $# -lt 1 ]]; then
    echo "ABORT: need at least one TAG (eval/p1_15_alt_decoder/<tag> dir name)."
    exit 1
fi

EVAL_DIRS=()
for t in "$@"; do
    EVAL_DIRS+=("${DATA_ROOT}/eval/p1_15_alt_decoder/${t}")
done
OUT_DIR=${DATA_ROOT}/eval/p1_15_alt_decoder/${SUMMARY_TAG}_summary

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
