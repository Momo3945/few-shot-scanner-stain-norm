#!/bin/bash
#SBATCH --job-name=fetch_models
#SBATCH --partition=stampede
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=10:00:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail

# --- conda env ---
source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm

# --- cache location (the variable that bit you before) ---
export HF_HOME=/datasets/mhoosen/hf_cache
mkdir -p "$HF_HOME"
echo "HF_HOME = $HF_HOME"

# --- fail early if this node cannot reach Hugging Face ---
if ! curl -sI https://huggingface.co | head -1 | grep -q HTTP; then
    echo "ERROR: this compute node cannot reach huggingface.co. Aborting." >&2
    exit 1
fi
echo "huggingface.co reachable — starting downloads."

# --- downloads (resumable; safe to rerun) ---
echo "== SD 1.5 =="
hf download stable-diffusion-v1-5/stable-diffusion-v1-5
echo "== LCM-LoRA (SD1.5) =="
hf download latent-consistency/lcm-lora-sdv1-5
echo "== ControlNet Canny (SD1.5) =="
hf download lllyasviel/sd-controlnet-canny
echo "== SDXL base (safetensors only) =="
hf download stabilityai/stable-diffusion-xl-base-1.0 --include "*.safetensors" "*.json" "*.txt" "*.model"
echo "== SD 3.5 large (safetensors only) =="
hf download stabilityai/stable-diffusion-3.5-large --include "*.safetensors" "*.json" "*.txt" "*.model"

echo "ALL DOWNLOADS COMPLETE"
