#!/bin/bash
# =====================================================================
# diagnose_text2img_colour_default.slurm -- follow-up to D1
# (tickets/PHASE3-TICKETS.md P3-06b/P3-07 H2A): pure text2img generation
# (colour LoRA only, no ControlNet, no source image) to isolate each
# backbone+direction's actual colour-free default. See
# src/eval/diagnose_text2img_colour_default.py's docstring.
#
# Submit with:  BACKBONE LORA_DIR RESOLUTION OUT_TAG
#   sbatch --exclude=<bad-node-list> diagnose_text2img_colour_default.slurm sd15 /datasets/mhoosen/stain-norm/lora/a2h_cond_r8/best 512 sd15_a2h
#   sbatch --exclude=<bad-node-list> diagnose_text2img_colour_default.slurm sdxl /datasets/mhoosen/stain-norm/lora/h2a_cond_r8_sdxl_1024/final 1024 sdxl1024_h2a
# =====================================================================

#SBATCH --job-name=diagnose_t2i
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail

CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_ROOT=/datasets/mhoosen/stain-norm
export HF_HOME=/datasets/mhoosen/hf_cache

BACKBONE=${1:?"Usage: sbatch diagnose_text2img_colour_default.slurm BACKBONE LORA_DIR RESOLUTION OUT_TAG"}
LORA_DIR=${2:?"need LORA_DIR"}
RESOLUTION=${3:-512}
OUT_TAG=${4:?"need OUT_TAG"}
OUT_DIR=${DATA_ROOT}/eval/diagnose_t2i/${OUT_TAG}

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import transformers, peft, accelerate, safetensors, diffusers" 2>/dev/null; then
    pip install --quiet transformers peft accelerate safetensors diffusers
fi

if [[ ! -f "${LORA_DIR}/pytorch_lora_weights.safetensors" ]]; then
    echo "ABORT: ${LORA_DIR}/pytorch_lora_weights.safetensors not found."
    exit 1
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Backbone=$BACKBONE  LoRA=$LORA_DIR  Resolution=$RESOLUTION  Out=$OUT_DIR"
echo "=================================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo 'ABORT: no GPU on this node -- resubmit.'; exit 1; }
echo "------------------------------------------------------------------"

cd "${CODE_DIR}"
python src/eval/diagnose_text2img_colour_default.py \
    --backbone "${BACKBONE}" \
    --lora "${LORA_DIR}" \
    --resolution "${RESOLUTION}" \
    --out "${OUT_DIR}" \
    --n-samples 24

echo "------------------------------------------------------------------"
echo "Diagnostic generation done. Samples in ${OUT_DIR}"
