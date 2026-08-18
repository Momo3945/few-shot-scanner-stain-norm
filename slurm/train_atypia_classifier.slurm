#!/bin/bash
# =====================================================================
# train_atypia_classifier.slurm -- fine-tune a ResNet18 atypia-score classifier
# on raw Aperio x20 patches (P2-09, Clinical Utility, GPU).
#
# Requires the training-Aperio data + labels already uploaded/extracted to
# DATA_DIR and the atypia_manifest.csv already built (build_atypia_manifest.py) --
# both are one-time, explicitly-confirmed steps, not run by this launcher.
#
# Submit with:  [MANIFEST] [OUT_TAG]
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 train_atypia_classifier.slurm                          # defaults below
#   SMOKE=1 sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 --time=00:30:00 -J atypia_smoke train_atypia_classifier.slurm
#
# Uses the plain stainnorm env -- torchvision already installed (0.20.1+cu121,
# verified 2026-08-18), no new env needed. TORCH_HOME redirected to /datasets
# (not the 50GB home dir) since ResNet18's ImageNet weights are not yet cached
# anywhere on this cluster -- first run downloads them via the compute node's
# outbound network access (same access fetch_models.slurm already relies on).
# =====================================================================

#SBATCH --job-name=atypia_classifier
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail

CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_ROOT=/datasets/mhoosen/stain-norm
DATA_DIR=${DATA_ROOT}/mitos_atypia_train_aperio
export TORCH_HOME=/datasets/mhoosen/torch_cache

MANIFEST=${1:-${DATA_ROOT}/pairs/atypia_train_manifest.csv}
OUT_TAG=${2:-atypia_r18}
OUT_DIR=${DATA_ROOT}/classifier/${OUT_TAG}
SMOKE_FLAG=""
if [[ "${SMOKE:-0}" == "1" ]]; then
    SMOKE_FLAG="--smoke"
fi

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import torch, torchvision, tifffile, cv2" 2>/dev/null; then
    echo "Installing missing Python deps..."
    pip install --quiet torchvision opencv-python-headless tifffile
else
    echo "Python deps present."
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Manifest=$MANIFEST  Root=$DATA_DIR  Smoke='${SMOKE_FLAG}'"
echo "Out=$OUT_DIR"
echo "=================================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
# fail fast on a GPU-less node instead of crawling on CPU
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo 'ABORT: no GPU on this node -- resubmit.'; exit 1; }
echo "------------------------------------------------------------------"

cd "${CODE_DIR}/src/train"
python train_atypia_classifier.py \
    --manifest    "${MANIFEST}" \
    --root        "${DATA_DIR}" \
    --output-dir  "${OUT_DIR}" \
    --seed 0 \
    ${SMOKE_FLAG}

echo "------------------------------------------------------------------"
echo "Job $SLURM_JOB_ID finished. Best checkpoint in ${OUT_DIR}/best.pt"
