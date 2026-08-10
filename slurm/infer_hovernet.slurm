#!/bin/bash
# =====================================================================
# infer_hovernet.slurm -- run pretrained HoVer-Net over a directory of images (GPU).
# P2-08, Structural Safety: this produces the "A" or "B" side of the
# G/A/B triangle (score against G with lizard_dice.py afterward).
#
# Submit with:  TAG [IMAGES_DIR] [PRETRAINED_MODEL] [LIMIT]
#   sbatch infer_hovernet.slurm lizard_original                     # default images dir, all 130
#   sbatch infer_hovernet.slurm lizard_original "" "" 5              # smoke test, 5 images
#   sbatch infer_hovernet.slurm lizard_normalised /datasets/mhoosen/stain-norm/eval/lizard_normalised_images
# TAG names the output folder eval/<TAG>/. IMAGES_DIR defaults to the
# original Lizard dpath/glas subset; pass an override for the normalised-B run
# once that exists (P2-08 step 5, not yet built).
#
# Uses the SEPARATE stainnorm-hovernet conda env (tiatoolbox 1.6.0), NOT
# stainnorm -- tiatoolbox's dependency stack downgrades scipy/scikit-image/
# opencv-headless below what the rest of the metrics pipeline needs. See
# tickets/PHASE2-TICKETS.md P2-08.
# =====================================================================

#SBATCH --job-name=infer_hovernet
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail

CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_ROOT=/datasets/mhoosen/stain-norm

TAG=${1:?"Usage: sbatch infer_hovernet.slurm TAG [IMAGES_DIR] [PRETRAINED_MODEL] [LIMIT]"}
IMAGES_DIR=${2:-${DATA_ROOT}/lizard_heldout/images}
PRETRAINED_MODEL=${3:-hovernet_fast-pannuke}
LIMIT=${4:-0}

OUT_DIR=${DATA_ROOT}/eval/${TAG}

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm-hovernet
if ! python -c "import tiatoolbox, torch, joblib, cv2" 2>/dev/null; then
    echo "ABORT: stainnorm-hovernet env is missing an expected package (tiatoolbox/torch/joblib/cv2)."
    echo "This env is hand-built (see tickets/PHASE2-TICKETS.md P2-08) -- do not auto-pip-install here,"
    echo "fix it manually and re-submit."
    exit 1
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Images=$IMAGES_DIR  Model=$PRETRAINED_MODEL  Limit=$LIMIT"
echo "Out=$OUT_DIR"
echo "=================================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed"
# fail fast on a GPU-less node instead of crawling on CPU
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo 'ABORT: no GPU on this node -- resubmit.'; exit 1; }
echo "------------------------------------------------------------------"

cd "${CODE_DIR}/src/eval"
python hovernet_wrapper.py \
    --images-dir       "${IMAGES_DIR}" \
    --out              "${OUT_DIR}" \
    --pretrained-model "${PRETRAINED_MODEL}" \
    --device cuda \
    --limit "${LIMIT}"

echo "------------------------------------------------------------------"
echo "HoVer-Net inference done. instances/ + masks/ + manifest.csv in ${OUT_DIR}"
echo "Next: score with lizard_dice.py (runs in the plain stainnorm env, not this one)."
