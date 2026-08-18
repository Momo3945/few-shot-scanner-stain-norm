#!/bin/bash
# =====================================================================
# train_colour_lora_sdxl.slurm -- scanner-colour LoRA training on SDXL (P3-03).
#
# Transfers ONLY the A2 training objective to SDXL, A2H direction (P3-03's scope:
# the direction the A4-SDXL-vs-SD1.5 comparison needs; H2A-on-SDXL is out of
# scope, see tickets/PHASE3-TICKETS.md P3-03). Trains at the existing 512x512
# pairs/train crops, told honestly to SDXL via micro-conditioning rather than
# spoofed as 1024 -- see train_colour_lora_sdxl.py's docstring and
# tickets/PHASE3-TICKETS.md P3-03's "Resolution decision".
#
# Submit with positional args:  [DIRECTION] [RANK] [TRAIN_STEPS]
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 train_colour_lora_sdxl.slurm            # A2H, rank 8, 1000 steps (defaults)
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 train_colour_lora_sdxl.slurm A2H 8 1000
#
# Smoke test (5 steps) -- override time + set SMOKE=1:
#   SMOKE=1 sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 --time=00:30:00 -J colour_smoke_sdxl train_colour_lora_sdxl.slurm A2H 8
#
# --time=04:00:00 (vs SD1.5's 2hr): SDXL's UNet (~2.6B params) is ~3x SD1.5's --
# this is a starting estimate, re-tune from the smoke test's measured step timing
# before submitting the full run (see tickets/PHASE3-TICKETS.md P3-03 run sequence).
#
# Uses the plain stainnorm env (main diffusion pipeline; SDXL classes confirmed
# importable there, diffusers 0.39.0 -- verified 2026-08-17, no new env needed).
# =====================================================================

#SBATCH --job-name=colour_lora_sdxl
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
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
# Arguments (DIRECTION defaults to A2H -- P3-03 only trains this direction)
# ---------------------------------------------------------------------
DIRECTION=${1:-A2H}
RANK=${2:-8}
STEPS=${3:-1000}
SMOKE_FLAG=""
if [[ "${SMOKE:-0}" == "1" ]]; then
    SMOKE_FLAG="--smoke"
fi

OUT_DIR="${OUT_ROOT}/${DIRECTION,,}_r${RANK}_sdxl"

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
echo "SDXL -- Direction=$DIRECTION  Rank=$RANK  Steps=$STEPS  Smoke='${SMOKE_FLAG}'"
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
python src/train/train_colour_lora_sdxl.py \
    --pairs-dir   "${DATA_DIR}" \
    --direction   "${DIRECTION}" \
    --rank        "${RANK}" \
    --train-steps "${STEPS}" \
    --output-dir  "${OUT_DIR}" \
    --seed 0 \
    ${SMOKE_FLAG}

echo "------------------------------------------------------------------"
echo "Job $SLURM_JOB_ID finished. LoRA in ${OUT_DIR}/final"
