#!/bin/bash
# =====================================================================
# score_lizard.slurm -- score HoVer-Net predictions vs Lizard ground truth (CPU).
# P2-08, Structural Safety. Run AFTER infer_hovernet.slurm has produced
# eval/<PRED_TAG>/{instances,masks}/.
#
# Submit with:  PRED_TAG [AGAINST_TAG]
#   sbatch score_lizard.slurm lizard_original                  # Dice(A,G) validation gate
#   sbatch score_lizard.slurm lizard_normalised lizard_original # Relative Dice = Dice(B,G)/Dice(A,G)
#
# CPU-only (no tiatoolbox import), runs in the plain stainnorm env on stampede --
# deliberately NOT stainnorm-hovernet, see tickets/PHASE2-TICKETS.md P2-08.
# =====================================================================

#SBATCH --job-name=score_lizard
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
LABELS_DIR=${DATA_ROOT}/lizard_heldout/labels

PRED_TAG=${1:?"Usage: sbatch score_lizard.slurm PRED_TAG [AGAINST_TAG]"}
AGAINST_TAG=${2:-}

PRED_DIR=${DATA_ROOT}/eval/${PRED_TAG}
OUT_DIR=${PRED_DIR}

mkdir -p "${CODE_DIR}/logs"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import numpy, scipy" 2>/dev/null; then
    pip install --quiet numpy scipy
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Pred=$PRED_DIR  Labels=$LABELS_DIR  Against=${AGAINST_TAG:-<none, validation-gate run>}"
echo "=================================================================="

cd "${CODE_DIR}/src/eval"
AGAINST_ARG=""
if [[ -n "$AGAINST_TAG" ]]; then
    AGAINST_SUMMARY=${DATA_ROOT}/eval/${AGAINST_TAG}/summary.csv
    if [[ ! -f "$AGAINST_SUMMARY" ]]; then
        echo "ABORT: --against summary not found at ${AGAINST_SUMMARY} -- score that run first."
        exit 1
    fi
    AGAINST_ARG="--against ${AGAINST_SUMMARY}"
fi

python lizard_dice.py \
    --pred-dir   "${PRED_DIR}" \
    --labels-dir "${LABELS_DIR}" \
    --out        "${OUT_DIR}" \
    ${AGAINST_ARG}

echo "------------------------------------------------------------------"
echo "Scoring done. per_image.csv + summary.csv in ${OUT_DIR}"
