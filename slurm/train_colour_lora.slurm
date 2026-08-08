#!/bin/bash
# =====================================================================
# train_colour_lora.slurm -- scanner-colour LoRA training (A2) on bigbatch.
#
# Submit with positional args:  DIRECTION [RANK] [TRAIN_STEPS]
#   sbatch train_colour_lora.slurm A2H 8            # A->H, rank 8, default steps
#   sbatch train_colour_lora.slurm A2H 4            # A->H, rank 4
#   sbatch train_colour_lora.slurm H2A 8            # H->A, rank 8 (cycle consistency)
#
# Smoke test (5 steps) -- override time + set SMOKE=1:
#   SMOKE=1 sbatch --time=00:15:00 -J colour_smoke train_colour_lora.slurm A2H 8
# =====================================================================

#SBATCH --job-name=colour_lora
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail

# ---------------------------------------------------------------------
# Paths (edit here if your layout differs)
# ---------------------------------------------------------------------
CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_DIR=/datasets/mhoosen/stain-norm/pairs/train
OUT_ROOT=/datasets/mhoosen/stain-norm/lora
export HF_HOME=/datasets/mhoosen/hf_cache

# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------
DIRECTION=${1:?"Usage: sbatch train_colour_lora.slurm DIRECTION [RANK] [STEPS]  (DIRECTION = A2H | H2A)"}
RANK=${2:-8}
STEPS=${3:-1000}
SMOKE_FLAG=""
if [[ "${SMOKE:-0}" == "1" ]]; then
    SMOKE_FLAG="--smoke"
fi

OUT_DIR="${OUT_ROOT}/${DIRECTION,,}_r${RANK}"

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
echo "Direction=$DIRECTION  Rank=$RANK  Steps=$STEPS  Smoke='${SMOKE_FLAG}'"
echo "Out=$OUT_DIR"
echo "HF_HOME=$HF_HOME"
echo "=================================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed -- no GPU visible?"
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
echo "------------------------------------------------------------------"

# ---------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------
cd "${CODE_DIR}"
python src/train/train_colour_lora.py \
    --pairs-dir   "${DATA_DIR}" \
    --direction   "${DIRECTION}" \
    --rank        "${RANK}" \
    --train-steps "${STEPS}" \
    --output-dir  "${OUT_DIR}" \
    --seed 0 \
    ${SMOKE_FLAG}

echo "------------------------------------------------------------------"
echo "Job $SLURM_JOB_ID finished. LoRA in ${OUT_DIR}/final"

