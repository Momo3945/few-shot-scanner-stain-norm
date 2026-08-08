#!/bin/bash
# =====================================================================
# train_hist_lora.slurm -- A5 histopathology warm-start LoRA training on bigbatch.
#
# Submit with positional args:  [RANK] [TRAIN_STEPS]
#   sbatch train_hist_lora.slurm                # rank 32, 3000 steps (proposal default)
#   sbatch train_hist_lora.slurm 32 3000         # explicit, same as above
#
# Smoke test (5 steps) -- override time + set SMOKE=1:
#   SMOKE=1 sbatch --time=00:15:00 -J hist_smoke train_hist_lora.slurm
#
# Per sec:training_order, this must be trained and FROZEN before colour-LoRA
# training begins for the A5 leg -- do not run this concurrently with an A5
# colour-LoRA training job that expects to load it.
# =====================================================================

#SBATCH --job-name=hist_lora
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail

# ---------------------------------------------------------------------
# Paths (edit here if your layout differs)
# ---------------------------------------------------------------------
CODE_DIR=/home-mscluster/mhoosen/stain-norm
POOL_DIR=/datasets/mhoosen/stain-norm/hist_lora_pool
MANIFEST=${POOL_DIR}/manifest.csv
PATCHES_DIR=${POOL_DIR}/patches
OUT_ROOT=/datasets/mhoosen/stain-norm/lora
export HF_HOME=/datasets/mhoosen/hf_cache

# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------
RANK=${1:-32}
STEPS=${2:-3000}
SMOKE_FLAG=""
if [[ "${SMOKE:-0}" == "1" ]]; then
    SMOKE_FLAG="--smoke"
fi

OUT_DIR="${OUT_ROOT}/hist_r${RANK}"

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------
source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm

# ---------------------------------------------------------------------
# Ensure Python deps (installs ONLY if something is missing; instant otherwise).
# torch is deliberately NOT listed -- it is already installed with the correct
# CUDA build and must not be reinstalled/changed by a job.
# ---------------------------------------------------------------------
if ! python -c "import transformers, peft, accelerate, safetensors, diffusers" 2>/dev/null; then
    echo "Installing missing Python deps (transformers/peft/accelerate/safetensors/diffusers)..."
    pip install --quiet transformers peft accelerate safetensors diffusers
else
    echo "Python deps present."
fi

# ---------------------------------------------------------------------
# Diagnostics (so a GPU/CUDA problem is obvious at the top of the log)
# ---------------------------------------------------------------------
echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Rank=$RANK  Steps=$STEPS  Smoke='${SMOKE_FLAG}'"
echo "Manifest=$MANIFEST"
echo "Out=$OUT_DIR"
echo "HF_HOME=$HF_HOME"
echo "=================================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed -- no GPU visible?"
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
# fail fast on a GPU-less node instead of crawling on CPU
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo 'ABORT: no GPU on this node -- resubmit.'; exit 1; }
echo "------------------------------------------------------------------"

# ---------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------
cd "${CODE_DIR}"
python src/train/train_hist_lora.py \
    --manifest    "${MANIFEST}" \
    --patches-dir "${PATCHES_DIR}" \
    --rank        "${RANK}" \
    --train-steps "${STEPS}" \
    --output-dir  "${OUT_DIR}" \
    --seed 0 \
    ${SMOKE_FLAG}

echo "------------------------------------------------------------------"
echo "Job $SLURM_JOB_ID finished. LoRA in ${OUT_DIR}/final"
echo "Freeze this checkpoint before starting A5 colour-LoRA training (sec:training_order)."
