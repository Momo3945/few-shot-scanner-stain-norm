#!/bin/bash
# =====================================================================
# score_outputs.slurm -- score img2img outputs vs held-out ground truth (CPU).
#
# Run AFTER infer_colour_lora.slurm has produced eval/<LORA_TAG>/eval_manifest.csv.
#   sbatch score_outputs.slurm a2h_r8
#   sbatch score_outputs.slurm a2h_r4
# CPU-only (no model), so it runs on stampede and keeps GPU nodes free.
# =====================================================================

#SBATCH --job-name=score_lora
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
BASELINE=${DATA_ROOT}/eval/baseline_metrics/baseline_summary.csv   # <-- edit to your baseline path

LORA_TAG=${1:?"Usage: sbatch score_outputs.slurm LORA_TAG"}
EVAL_DIR=${DATA_ROOT}/eval/${LORA_TAG}

mkdir -p "${CODE_DIR}/logs"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import cv2, scipy, skimage" 2>/dev/null; then
    pip install --quiet opencv-python-headless scipy scikit-image tifffile
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Eval dir=$EVAL_DIR"
echo "Baseline=$BASELINE"
echo "=================================================================="

cd "${CODE_DIR}"
BASELINE_ARG=""
if [[ -f "$BASELINE" ]]; then
    BASELINE_ARG="--baseline ${BASELINE}"
else
    echo "NOTE: baseline summary not found at ${BASELINE} -- scoring without recovery delta."
fi

python src/eval/score_outputs.py --eval-dir "${EVAL_DIR}" ${BASELINE_ARG}

echo "------------------------------------------------------------------"
echo "Scoring done. eval_summary.csv + eval_per_crop.csv in ${EVAL_DIR}"
