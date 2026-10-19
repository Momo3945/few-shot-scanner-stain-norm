#!/bin/bash
# =====================================================================
# normalize_lizard_inversion_fusion.slurm -- P2-08b: the final SD1.5 pipeline (P1-10 source-conditioned
# model + P1-11 DDIM inversion + P1-16 F3 fusion) over the 130 Lizard structural-safety images (GPU).
# Writes eval/lizard_normalised_images_inv_f3/ (fused) and eval/lizard_normalised_images_inv/ (unfused).
# Then: infer_hovernet.slurm on each dir, score_lizard.slurm <tag> lizard_original.
# See tickets/PHASE2-TICKETS.md P2-08b.
#
# Smoke (2 images):   LIMIT=2 PYTHONUNBUFFERED=1 sbatch --exclude=<BAD_NODES> --time=00:20:00 \
#                       -J p2_08b_smoke slurm/normalize_lizard_inversion_fusion.slurm smoke
# Full run:           PYTHONUNBUFFERED=1 sbatch --exclude=<BAD_NODES> --time=03:00:00 \
#                       -J p2_08b_norm slurm/normalize_lizard_inversion_fusion.slurm full
# Resumable: re-submitting skips images already written.
# =====================================================================

#SBATCH --job-name=p2_08b_norm
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail

CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_ROOT=/datasets/mhoosen/stain-norm
CHECKPOINT_DIR=${DATA_ROOT}/lora/a2h_cond_r8/best
export HF_HOME=/datasets/mhoosen/hf_cache

RUN=${1:?"Usage: sbatch normalize_lizard_inversion_fusion.slurm smoke|full"}
if [[ "${RUN}" == "smoke" ]]; then
    SUFFIX="_smoke"; LIMIT=${LIMIT:-2}
else
    SUFFIX=""; LIMIT=${LIMIT:-0}
fi
OUT_FUSED=${DATA_ROOT}/eval/lizard_normalised_images_inv_f3${SUFFIX}
OUT_UNFUSED=${DATA_ROOT}/eval/lizard_normalised_images_inv${SUFFIX}

if [[ ! -f "${CHECKPOINT_DIR}/pytorch_lora_weights.safetensors" ]]; then
    echo "ABORT: P1-10 checkpoint not found at ${CHECKPOINT_DIR}"; exit 1
fi
mkdir -p "${CODE_DIR}/logs"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "P2-08b -- run=${RUN} limit=${LIMIT}"
echo "Fused -> ${OUT_FUSED}   Unfused -> ${OUT_UNFUSED}"
echo "=================================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed -- no GPU visible?"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo 'ABORT: no GPU on this node -- resubmit.'; exit 1; }
echo "------------------------------------------------------------------"

cd "${CODE_DIR}/src/eval"
python normalize_lizard_inversion_fusion.py \
    --lora "${CHECKPOINT_DIR}" --controlnet "${CHECKPOINT_DIR}/controlnet" \
    --images-dir "${DATA_ROOT}/lizard_heldout/images" \
    --out-fused "${OUT_FUSED}" --out-unfused "${OUT_UNFUSED}" \
    --limit "${LIMIT}"

echo "------------------------------------------------------------------"
echo "Done. Next: infer_hovernet.slurm on each output dir, then score_lizard.slurm <tag> lizard_original."
