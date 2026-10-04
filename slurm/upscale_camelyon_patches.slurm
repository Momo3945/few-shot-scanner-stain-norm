#!/bin/bash
# =====================================================================
# upscale_camelyon_patches.slurm -- P3-09: upscale CAMELYON17's 512x512
# patches to 1024x1024 (CPU, no model). See tickets/PHASE3-TICKETS.md P3-09,
# src/eval/upscale_camelyon_patches.py.
#
# Submit with no args:
#   sbatch -J p3_09_upscale upscale_camelyon_patches.slurm
# =====================================================================

#SBATCH --job-name=p3_09_upscale
#SBATCH --partition=stampede
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail

CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_ROOT=/datasets/mhoosen/stain-norm

mkdir -p "${CODE_DIR}/logs"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "=================================================================="

cd "${CODE_DIR}/src/eval"
python upscale_camelyon_patches.py \
    --root "${DATA_ROOT}/camelyon17_patches" \
    --out "${DATA_ROOT}/camelyon17_patches_1024" \
    --size 1024

echo "------------------------------------------------------------------"
echo "Upscale done. Output in ${DATA_ROOT}/camelyon17_patches_1024"
