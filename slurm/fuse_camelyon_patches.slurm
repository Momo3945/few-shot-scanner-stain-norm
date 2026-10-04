#!/bin/bash
# =====================================================================
# fuse_camelyon_patches.slurm -- P3-09: F3 fusion (sigma=8, beta=0.50) over the
# CAMELYON17 SDXL outputs (CPU, stampede). See tickets/PHASE3-TICKETS.md P3-09,
# src/eval/fuse_camelyon_patches.py.
#
# Prerequisite: normalize_camelyon_sdxl.slurm vae_only AND correct runs done.
#   sbatch -J p3_09_fuse slurm/fuse_camelyon_patches.slurm
# Then: score_camelyon_wasserstein.slurm <fused dir> <tag> <D_pre pairwise.csv>
# =====================================================================

#SBATCH --job-name=p3_09_fuse
#SBATCH --partition=stampede
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail

CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_ROOT=/datasets/mhoosen/stain-norm
OUT_DIR=${DATA_ROOT}/eval/p3_09_sdxl_fused

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Out=${OUT_DIR}"
echo "=================================================================="

cd "${CODE_DIR}/src/eval"
python fuse_camelyon_patches.py \
    --raw      "${DATA_ROOT}/camelyon17_patches_1024" \
    --vae-only "${DATA_ROOT}/eval/p3_09_sdxl_vae_only" \
    --pred     "${DATA_ROOT}/eval/p3_09_sdxl_correct" \
    --out      "${OUT_DIR}" --workers 4

echo "------------------------------------------------------------------"
echo "Fusion done. Outputs in ${OUT_DIR}"
