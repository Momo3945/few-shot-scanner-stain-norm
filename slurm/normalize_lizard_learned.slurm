#!/bin/bash
# =====================================================================
# normalize_lizard_learned.slurm -- normalise the Lizard held-out images with a trained
# StainNet / ParamNet / StainGAN (A2H), GPU. P2-11b structure-safety arm; learned-baseline
# counterpart of normalize_lizard.slurm. Produces the "B" side of the G/A/B triangle.
#
# Submit with positional args:  METHOD [LIMIT]
#   sbatch normalize_lizard_learned.slurm paramnet 3     # smoke, 3 images
#   sbatch normalize_lizard_learned.slurm stainnet       # full 130
# Output: eval/lizard_normalised_images_<METHOD>/
# Then:   sbatch infer_hovernet.slurm lizard_normalised_<METHOD> <that dir> "" ""
#         sbatch score_lizard.slurm  lizard_normalised_<METHOD> lizard_original
# Always add the bad-node --exclude list from CLAUDE.md on submission.
# =====================================================================

#SBATCH --job-name=norm_lizard_learned
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_ROOT=/datasets/mhoosen/stain-norm
IMAGES_DIR=${DATA_ROOT}/lizard_heldout/images

METHOD=${1:?"Usage: sbatch normalize_lizard_learned.slurm METHOD [LIMIT]  (METHOD = stainnet | paramnet | staingan)"}
LIMIT=${2:-0}
CKPT=${DATA_ROOT}/baselines/a2h_${METHOD}/${METHOD}_a2h.pt
OUT_DIR=${DATA_ROOT}/eval/lizard_normalised_images_${METHOD}
if [[ "${LIMIT}" != "0" ]]; then OUT_DIR="${OUT_DIR}_smoke"; fi

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"
[[ -f "${CKPT}" ]] || { echo "ABORT: checkpoint not found: ${CKPT}"; exit 1; }

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import cv2, skimage" 2>/dev/null; then
    pip install --quiet opencv-python-headless scikit-image
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Method=$METHOD  Limit=$LIMIT  Ckpt=$CKPT"
echo "Images=$IMAGES_DIR  Out=$OUT_DIR"
nvidia-smi -L || true
echo "=================================================================="
python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo "ABORT: no GPU on this node -- resubmit."; exit 1; }

cd "${CODE_DIR}/src/eval"
python normalize_lizard_learned.py \
    --method     "${METHOD}" \
    --ckpt       "${CKPT}" \
    --images-dir "${IMAGES_DIR}" \
    --out        "${OUT_DIR}" \
    --limit      "${LIMIT}"

echo "------------------------------------------------------------------"
echo "Normalisation done. Images in ${OUT_DIR}"
echo "Next: sbatch slurm/infer_hovernet.slurm lizard_normalised_${METHOD} ${OUT_DIR} \"\" \"\""
