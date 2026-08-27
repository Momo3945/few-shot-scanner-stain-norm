#!/bin/bash
# =====================================================================
# fetch_sd_vae_ft_mse.slurm -- one-off fetch of stabilityai/sd-vae-ft-mse
# for P1-14 (not yet cached as of 2026-08-26, confirmed via
# /datasets/mhoosen/hf_cache/hub inventory). Public, not gated. Also added
# to fetch_models.slurm's own download list for future re-fetches.
#
# Submit with no args:
#   sbatch fetch_sd_vae_ft_mse.slurm
# =====================================================================

#SBATCH --job-name=fetch_vae_ft_mse
#SBATCH --partition=stampede
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
export HF_HOME=/datasets/mhoosen/hf_cache
mkdir -p "$HF_HOME"

echo "HF_HOME = $HF_HOME"
if ! curl -sI https://huggingface.co | head -1 | grep -q HTTP; then
    echo "ERROR: this compute node cannot reach huggingface.co. Aborting." >&2
    exit 1
fi

hf download stabilityai/sd-vae-ft-mse

echo "DOWNLOAD COMPLETE"
