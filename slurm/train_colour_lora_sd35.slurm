#!/bin/bash
# =====================================================================
# train_colour_lora_sd35.slurm -- SD 3.5 A2H colour-LoRA training (PR-01 probe).
#
# Auxiliary probe only (tickets/PROBE-SD35-TICKETS.md) -- NOT part of A0-A5.
# Measures wall-clock training time and peak VRAM for an A2H colour LoRA on
# the same <=50 A03/H03 crop pairs and hyperparameters as the SD1.5 A2
# baseline (rank 8, 1000 steps).
#
# Submit with positional args:  [DIRECTION] [RANK] [TRAIN_STEPS]
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 train_colour_lora_sd35.slurm            # A2H, rank 8, 1000 steps (defaults)
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 train_colour_lora_sd35.slurm A2H 8 1000
#
# Smoke test (5 steps) -- override time + set SMOKE=1:
#   SMOKE=1 sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 --time=00:30:00 -J colour_smoke_sd35 train_colour_lora_sd35.slurm A2H 8
#
# --partition=bigbatch (NOT biggpu): the PR-01 plan originally assumed 27GB
# of simultaneous frozen-fp16-weight residency (text encoders + transformer
# all resident together), which would have exceeded bigbatch's RTX 3090
# 24GB. Empirically WRONG -- fixed 2026-08-22 (job 45158, smoke test): the
# script frees the text encoders before the transformer loads, and the
# transformer is stored directly in bf16 (not fp32), so real peak VRAM is
# 17.77GB, comfortably under 24GB. biggpu was never actually required once
# both bugs (offline sharded-checkpoint loading, fp32-transformer OOM) were
# fixed -- see tickets/PROBE-SD35-TICKETS.md PR-01. bigbatch is also the
# far better default in practice: biggpu was ~30 jobs backlogged from other
# users when this was tested. Pass the standard bad-node --exclude list
# like every other bigbatch GPU job in this project.
#
# --time=06:00:00: generous headroom over the smoke-test-measured pace
# (~0.6-0.7s/step steady-state -> ~1000-step run should take roughly
# 10-15 minutes wall-clock plus model-load overhead).
#
# Uses the plain stainnorm env (diffusers 0.39.0 -- confirmed to ship
# SD3Transformer2DModel, StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler,
# and the training_utils SD3 helpers -- verified 2026-08-21, no new env needed).
# =====================================================================

#SBATCH --job-name=colour_lora_sd35
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
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
# Arguments (DIRECTION defaults to A2H -- parity with the SD1.5 A2 baseline)
# ---------------------------------------------------------------------
DIRECTION=${1:-A2H}
RANK=${2:-8}
STEPS=${3:-1000}
SMOKE_FLAG=""
if [[ "${SMOKE:-0}" == "1" ]]; then
    SMOKE_FLAG="--smoke"
fi

OUT_DIR="${OUT_ROOT}/${DIRECTION,,}_r${RANK}_sd35"

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
if ! python -c "import transformers, peft, accelerate, safetensors, diffusers, sentencepiece" 2>/dev/null; then
    echo "Installing missing Python deps (transformers/peft/accelerate/safetensors/diffusers/sentencepiece)..."
    pip install --quiet transformers peft accelerate safetensors diffusers sentencepiece
else
    echo "Python deps present."
fi

# ---------------------------------------------------------------------
# Diagnostics (so a GPU/CUDA problem is obvious at the top of the log)
# ---------------------------------------------------------------------
echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "SD3.5 -- Direction=$DIRECTION  Rank=$RANK  Steps=$STEPS  Smoke='${SMOKE_FLAG}'"
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
python src/train/train_colour_lora_sd35.py \
    --pairs-dir   "${DATA_DIR}" \
    --direction   "${DIRECTION}" \
    --rank        "${RANK}" \
    --train-steps "${STEPS}" \
    --output-dir  "${OUT_DIR}" \
    --seed 0 \
    ${SMOKE_FLAG}

echo "------------------------------------------------------------------"
echo "Job $SLURM_JOB_ID finished. LoRA in ${OUT_DIR}/final"
