#!/bin/bash
# =====================================================================
# normalize_lizard_classical.slurm -- Macenko / Reinhard / Histogram-Matching normalisation of
# the Lizard held-out images (A2H), CPU-only on stampede. P2-11b structure-safety arm.
#
# Submit with positional args:  METHOD [LIMIT]
#   sbatch normalize_lizard_classical.slurm macenko 3     # smoke, 3 images
#   sbatch normalize_lizard_classical.slurm reinhard      # full 130
# METHOD = macenko | reinhard | histogram_matching
# Output: eval/lizard_normalised_images_<METHOD>/  (+ fallbacks.json)
# Then:   sbatch infer_hovernet.slurm lizard_normalised_<METHOD> <that dir> "" ""
#         sbatch score_lizard.slurm  lizard_normalised_<METHOD> lizard_original
# Macenko is the slow one (~6 s/tile in the held-out runs): give it --time=03:00:00.
# =====================================================================

#SBATCH --job-name=norm_lizard_classical
#SBATCH --partition=stampede
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_ROOT=/datasets/mhoosen/stain-norm
IMAGES_DIR=${DATA_ROOT}/lizard_heldout/images
TARGET_IMAGE=${TARGET_IMAGE:-${DATA_ROOT}/pairs/train/A03_00A_c000_hamamatsu.png}

METHOD=${1:?"Usage: sbatch normalize_lizard_classical.slurm METHOD [LIMIT]  (macenko | reinhard | histogram_matching)"}
LIMIT=${2:-0}
OUT_DIR=${DATA_ROOT}/eval/lizard_normalised_images_${METHOD}
if [[ "${LIMIT}" != "0" ]]; then OUT_DIR="${OUT_DIR}_smoke"; fi

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import cv2, skimage" 2>/dev/null; then
    pip install --quiet opencv-python-headless scikit-image
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Method=$METHOD  Limit=$LIMIT  Target=$TARGET_IMAGE"
echo "Images=$IMAGES_DIR  Out=$OUT_DIR"
echo "=================================================================="

cd "${CODE_DIR}/src/eval"
python normalize_lizard_classical.py \
    --method       "${METHOD}" \
    --target-image "${TARGET_IMAGE}" \
    --images-dir   "${IMAGES_DIR}" \
    --out          "${OUT_DIR}" \
    --limit        "${LIMIT}"

echo "------------------------------------------------------------------"
echo "Normalisation done. Images in ${OUT_DIR}"
echo "Next: sbatch slurm/infer_hovernet.slurm lizard_normalised_${METHOD} ${OUT_DIR} \"\" \"\""
