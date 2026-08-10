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
# NOTE: --include only takes ONE pattern per flag (hf CLI is Click-based, `--include TEXT`,
# not nargs). The old space-separated form silently bound only "*.safetensors" to --include
# and turned "*.json" "*.txt" "*.model" into positional FILENAMES, which switches hf download
# to a different code path that ignores --include entirely -- confirmed via --dry-run: it
# fetched only the 17 config/tokenizer files (3.2M), zero .safetensors. Repeated --include
# flags is the documented correct syntax (see `hf download --help` examples).
hf download stabilityai/stable-diffusion-xl-base-1.0 --include "*.safetensors" --include "*.json" --include "*.txt" --include "*.model"
echo "== SD 3.5 large (safetensors only) =="
# NOTE: this repo is GATED. Confirmed via --dry-run: "Error: Access denied. This repository
# requires approval." even with the --include fix above. Requires: (1) requesting/accepting
# access at https://huggingface.co/stabilityai/stable-diffusion-3.5-large with your HF
# account, and (2) `hf auth login` (or HF_TOKEN) on this cluster -- currently "Not logged in".
# Neither can be done from a script; this line will keep failing until both are done manually.
hf download stabilityai/stable-diffusion-3.5-large --include "*.safetensors" --include "*.json" --include "*.txt" --include "*.model"
echo "== LCM-LoRA (SDXL) =="
# Named explicitly in the proposal (sec:phase3_sdxl): "the latent-consistency/lcm-lora-sdxl
# adapter provides an SDXL-compatible LCM-LoRA pathway for 2-8 step inference". Needed for
# P3-03/P3-04 -- confirmed NOT present in the cache as of 2026-08-10 (PHASE3-TICKETS.md P3-01).
hf download latent-consistency/lcm-lora-sdxl
echo "== ControlNet Canny (SDXL) =="
# Proposal only says "SDXL ControlNet checkpoints support Canny" generically, no repo named.
# diffusers/controlnet-canny-sdxl-1.0 is the canonical SDXL Canny ControlNet released by the
# diffusers team (same role as lllyasviel/sd-controlnet-canny for SD1.5 above). Also confirmed
# NOT present in the cache as of 2026-08-10.
hf download diffusers/controlnet-canny-sdxl-1.0

echo "ALL DOWNLOADS COMPLETE"
