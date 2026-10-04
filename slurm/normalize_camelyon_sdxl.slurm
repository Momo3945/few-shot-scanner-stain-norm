#!/bin/bash
# =====================================================================
# normalize_camelyon_sdxl.slurm -- P3-09: SDXL source-conditioned img2img
# normalisation of CAMELYON17 patches (bigbatch, GPU). See
# tickets/PHASE3-TICKETS.md P3-09, src/eval/normalize_camelyon_sdxl.py.
#
# Submit with:  MODE [OUT_TAG]
#   MODE = vae_only | correct
#
#   # VAE-only floor (no checkpoint needed, cheap -- run first):
#   sbatch --exclude=mscluster48,... -J p3_09_vae_only normalize_camelyon_sdxl.slurm vae_only p3_09_sdxl_vae_only
#
#   # full translation (expensive -- 2103 patches x 50-step DDIM @ 1024):
#   sbatch --exclude=mscluster48,... -J p3_09_correct normalize_camelyon_sdxl.slurm correct p3_09_sdxl_correct
# =====================================================================

#SBATCH --job-name=p3_09_normalize
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail

CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_ROOT=/datasets/mhoosen/stain-norm
CHECKPOINT_DIR=${DATA_ROOT}/lora/a2h_cond_r8_sdxl_1024_overlap144/final
IMAGES_DIR=${DATA_ROOT}/camelyon17_patches_1024
export HF_HOME=/datasets/mhoosen/hf_cache

MODE=${1:?"Usage: sbatch normalize_camelyon_sdxl.slurm MODE [OUT_TAG]  (MODE = vae_only|correct)"}
OUT_TAG=${2:-p3_09_sdxl_${MODE}}
OUT_DIR=${DATA_ROOT}/eval/${OUT_TAG}
LIMIT=${LIMIT:-0}

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "P3-09 CAMELYON17 SDXL normalise -- Mode=$MODE  Out=$OUT_DIR  Limit=$LIMIT"
echo "=================================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo 'ABORT: no GPU on this node -- resubmit.'; exit 1; }
echo "------------------------------------------------------------------"

cd "${CODE_DIR}/src/eval"
if [[ "$MODE" == "vae_only" ]]; then
    python normalize_camelyon_sdxl.py --vae-only \
        --images-dir "${IMAGES_DIR}" --out "${OUT_DIR}" --limit "${LIMIT}"
else
    python normalize_camelyon_sdxl.py \
        --lora "${CHECKPOINT_DIR}" --controlnet "${CHECKPOINT_DIR}/controlnet" \
        --images-dir "${IMAGES_DIR}" --out "${OUT_DIR}" --limit "${LIMIT}"
fi

echo "------------------------------------------------------------------"
echo "Normalisation done. Outputs in ${OUT_DIR}"
echo "Next (both modes done): fuse_camelyon_patches.py, then score_camelyon_wasserstein.py"
