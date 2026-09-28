#!/bin/bash
# =====================================================================
# train_lora_pixcell.slurm -- PXC-01b: LoRA training-cost measurement
# (wall-clock + peak VRAM only). See tickets/PROBE-PIXCELL-TICKETS.md,
# src/train/train_lora_pixcell.py.
#
# NOT part of the graded proposal scope -- see that ticket file's caveat.
# NOT a quality benchmark -- PXC-02 already found embedding-only
# conditioning can't hold structure for this project's metrics (job
# 61123); this measures training cost only.
#
# Smoke test first:
#   SMOKE=1 sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44,mscluster75,mscluster74,mscluster51,mscluster83,mscluster76,mscluster50,mscluster45,mscluster57,mscluster59,mscluster62 \
#     --time=00:20:00 -J pxc_01b_smoke train_lora_pixcell.slurm
#
# Real run (cost measurement):
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44,mscluster75,mscluster74,mscluster51,mscluster83,mscluster76,mscluster50,mscluster45,mscluster57,mscluster59,mscluster62 \
#     -J pxc_01b_train train_lora_pixcell.slurm
# =====================================================================

#SBATCH --job-name=pxc_01b_train
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -uo pipefail

CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_ROOT=/datasets/mhoosen/stain-norm
PAIRS_DIR=${DATA_ROOT}/pairs/train_256
OUT_DIR=${DATA_ROOT}/lora/pxc_01b_lora
export HF_HOME=/datasets/mhoosen/hf_cache

SMOKE_FLAG=""
if [[ "${SMOKE:-0}" == "1" ]]; then
    SMOKE_FLAG="--smoke"
    OUT_DIR=${DATA_ROOT}/lora/pxc_01b_smoke
fi

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import timm" 2>/dev/null; then
    pip install --quiet timm
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "PairsDir=$PAIRS_DIR  Out=$OUT_DIR  Smoke='${SMOKE_FLAG}'"
echo "=================================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed -- no GPU visible?"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo 'ABORT: no GPU on this node -- resubmit.'; exit 1; }
echo "------------------------------------------------------------------"

cd "${CODE_DIR}/src/train"
# --online: trust_remote_code re-checks the Hub on every load (confirmed
# PXC-02 Part B, job 61122).
python train_lora_pixcell.py \
    --pairs-dir "${PAIRS_DIR}" \
    --output-dir "${OUT_DIR}" \
    --seed 0 --online ${SMOKE_FLAG}

echo "------------------------------------------------------------------"
echo "Job $SLURM_JOB_ID finished. Cost measurement in ${OUT_DIR}/training_config.json"
