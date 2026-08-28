#!/bin/bash
# =====================================================================
# infer_baseline.slurm -- classical stain-normalisation baseline (P2-11), CPU-only.
#
# Submit with positional args:  METHOD [LIMIT]
#   sbatch infer_baseline.slurm macenko              # full held-out set
#   sbatch infer_baseline.slurm reinhard 8            # smoke test, 8 frames
#   sbatch infer_baseline.slurm histogram_matching
# METHOD = macenko | reinhard | histogram_matching
#
# No GPU needed -- runs on stampede alongside score_outputs.slurm, keeps
# bigbatch free for LoRA training/inference.
# =====================================================================

#SBATCH --job-name=infer_baseline
#SBATCH --partition=stampede
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail

# ---------------------------------------------------------------------
# Paths (edit here if your layout differs)
# ---------------------------------------------------------------------
CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_ROOT=/datasets/mhoosen/stain-norm
MITOS_ROOT=${DATA_ROOT}/mitos_heldout
HELDOUT=${DATA_ROOT}/pairs/heldout_frames.csv
OUT_ROOT=${DATA_ROOT}/eval

# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------
METHOD=${1:?"Usage: sbatch infer_baseline.slurm METHOD [LIMIT]  (METHOD = macenko | reinhard | histogram_matching)"}
LIMIT=${2:-0}

# P2-12: optional overrides, unset -> identical to every prior invocation.
# Fixed few-shot reference crop -- same A03/H03 pair the colour LoRA trains on;
# A2H's target is the Hamamatsu side, H2A's is the Aperio side of the same pair.
DIRECTION=${DIRECTION:-A2H}
TARGET_IMAGE=${TARGET_IMAGE:-${DATA_ROOT}/pairs/train/A03_00A_c000_hamamatsu.png}
OUT_TAG=${OUT_TAG:-${METHOD}}
OUT_DIR="${OUT_ROOT}/${OUT_TAG}"

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------
source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm

if ! python -c "import cv2, skimage" 2>/dev/null; then
    echo "Installing missing Python deps (opencv-python-headless/scikit-image)..."
    pip install --quiet opencv-python-headless scikit-image
else
    echo "Python deps present."
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Method=$METHOD  Direction=$DIRECTION  Limit=$LIMIT"
echo "Target=$TARGET_IMAGE"
echo "Out=$OUT_DIR"
echo "=================================================================="

cd "${CODE_DIR}"
python src/eval/infer_baseline.py \
    --method       "${METHOD}" \
    --root         "${MITOS_ROOT}" \
    --heldout      "${HELDOUT}" \
    --target-image "${TARGET_IMAGE}" \
    --direction    "${DIRECTION}" \
    --out          "${OUT_DIR}" \
    --limit        "${LIMIT}" \
    --seed 0

echo "------------------------------------------------------------------"
echo "Inference done. Outputs + eval_manifest.csv in ${OUT_DIR}"
echo "Next: sbatch slurm/score_outputs.slurm ${METHOD}"
