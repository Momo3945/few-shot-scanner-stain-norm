#!/bin/bash
# =====================================================================
# probe_p1_10_controlnet_sd35_vram.slurm -- one-off feasibility check: does a
# P1-10-style source-conditioning ControlNet + colour LoRA fit on SD3.5's
# transformer within bigbatch's 24GB RTX 3090? Not a trainer -- see
# src/train/probe_p1_10_controlnet_sd35_vram.py's docstring. Pre-ticket
# exploration, not yet part of PROBE-SD35-TICKETS.md.
#
# Submit with:  [NUM_LAYERS] [RANK] [RESOLUTION]
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 probe_p1_10_controlnet_sd35_vram.slurm            # 12 layers, rank 8, 1024px
# =====================================================================

#SBATCH --job-name=probe_p1_10_sd35_vram
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail

CODE_DIR=/home-mscluster/mhoosen/stain-norm
export HF_HOME=/datasets/mhoosen/hf_cache

NUM_LAYERS=${1:-12}
RANK=${2:-8}
RESOLUTION=${3:-1024}

mkdir -p "${CODE_DIR}/logs"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import transformers, peft, accelerate, safetensors, diffusers" 2>/dev/null; then
    pip install --quiet transformers peft accelerate safetensors diffusers
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "NUM_LAYERS=$NUM_LAYERS  RANK=$RANK  RESOLUTION=$RESOLUTION"
echo "=================================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo 'ABORT: no GPU on this node -- resubmit.'; exit 1; }
echo "------------------------------------------------------------------"

cd "${CODE_DIR}"
python src/train/probe_p1_10_controlnet_sd35_vram.py \
    --num-layers "${NUM_LAYERS}" \
    --rank "${RANK}" \
    --resolution "${RESOLUTION}"

echo "------------------------------------------------------------------"
echo "Probe done."
