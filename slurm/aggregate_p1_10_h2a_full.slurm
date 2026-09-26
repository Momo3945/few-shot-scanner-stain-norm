#!/bin/bash
# =====================================================================
# aggregate_p1_10_h2a_full.slurm -- P1-11 H2A: per-slide/ALL/ALL_excl_outliers
# + recovery-delta aggregation for the H2A full held-out run, via the
# existing (unmodified) aggregate_p1_10_full.py -- fully generic over
# backbone/resolution/direction, confirmed by P1-10 A2H's, P3-06's, and
# P3-07's own use of it. On stampede (CPU). See tickets/PHASE1-TICKETS.md
# P1-11.
#
# Requires Step A (score_p1_10_ddim_inversion.slurm) to have already
# produced eval/p1_10_ddim_inversion/h2a_translate_inv100_correct_full_
# summary/per_crop.csv.
#
# Submit with no args:
#   sbatch aggregate_p1_10_h2a_full.slurm
# =====================================================================

#SBATCH --job-name=agg_p1_11_h2a_full
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
# RUN_TAG override (non-breaking): lets this launcher target any H2A full-run
# summary dir, not just the original f=1.00 one -- added for the f=0.25
# fraction-sweep follow-up run.
RUN_TAG=${RUN_TAG:-h2a_translate_inv100_correct_full}
PER_CROP=${DATA_ROOT}/eval/p1_10_ddim_inversion/${RUN_TAG}_summary/per_crop.csv
BASELINE=${DATA_ROOT}/pairs/baseline_metrics/baseline_summary.csv
OUT=${DATA_ROOT}/eval/p1_10_ddim_inversion/${RUN_TAG}_summary/eval_summary_final.csv

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Per-crop: $PER_CROP"
echo "Out: $OUT"
echo "=================================================================="

cd "${CODE_DIR}/src/eval"
python aggregate_p1_10_full.py --per-crop "${PER_CROP}" --baseline "${BASELINE}" --out "${OUT}"

echo "------------------------------------------------------------------"
echo "Aggregation done. Summary in ${OUT}"
