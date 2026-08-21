#!/bin/bash
# =====================================================================
# train_hist_lora_sdxl.slurm -- P3-05: A5 histopathology warm-start LoRA
# training on SDXL (bigbatch). Mirrors train_hist_lora.slurm's structure
# exactly, pointed at the SDXL script/output.
#
# Submit with positional args:  [RANK] [TRAIN_STEPS]
#   sbatch train_hist_lora_sdxl.slurm                # rank 32, 3000 steps (proposal default)
#   sbatch train_hist_lora_sdxl.slurm 32 3000        # explicit, same as above
#
# Smoke test (5 steps) -- override time + set SMOKE=1:
#   SMOKE=1 sbatch --time=00:20:00 -J hist_sdxl_smoke train_hist_lora_sdxl.slurm
#
# Per sec:training_order, this must be trained and FROZEN before it is
# stacked at inference (infer_colour_lora_sdxl.py --hist-lora, P3-05's
# actual "A5-on-SDXL" step).
# =====================================================================

#SBATCH --job-name=hist_lora_sdxl
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
POOL_DIR=/datasets/mhoosen/stain-norm/hist_lora_pool
MANIFEST=${POOL_DIR}/manifest.csv
PATCHES_DIR=${POOL_DIR}/patches
OUT_ROOT=/datasets/mhoosen/stain-norm/lora
export HF_HOME=/datasets/mhoosen/hf_cache

RANK=${1:-32}
STEPS=${2:-3000}
SMOKE_FLAG=""
if [[ "${SMOKE:-0}" == "1" ]]; then
    SMOKE_FLAG="--smoke"
fi

OUT_DIR="${OUT_ROOT}/hist_r${RANK}_sdxl"

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import transformers, peft, accelerate, safetensors, diffusers" 2>/dev/null; then
    echo "Installing missing Python deps..."
    pip install --quiet transformers peft accelerate safetensors diffusers
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "P3-05 -- SDXL hist LoRA -- Rank=$RANK  Steps=$STEPS  Smoke='${SMOKE_FLAG}'"
echo "Manifest=$MANIFEST"
echo "Out=$OUT_DIR"
echo "=================================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed -- no GPU visible?"
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo 'ABORT: no GPU on this node -- resubmit.'; exit 1; }
echo "------------------------------------------------------------------"

cd "${CODE_DIR}"
python src/train/train_hist_lora_sdxl.py \
    --manifest    "${MANIFEST}" \
    --patches-dir "${PATCHES_DIR}" \
    --rank        "${RANK}" \
    --train-steps "${STEPS}" \
    --output-dir  "${OUT_DIR}" \
    --seed 0 \
    ${SMOKE_FLAG}

echo "------------------------------------------------------------------"
echo "Job $SLURM_JOB_ID finished. LoRA in ${OUT_DIR}/final"
echo "Freeze this checkpoint before stacking at inference (P3-05 A5-on-SDXL)."
