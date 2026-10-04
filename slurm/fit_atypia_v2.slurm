#!/bin/bash
# =====================================================================
# fit_atypia_v2.slurm -- P2-15 step 2: nested leave-one-slide-out CV for the
# three candidate frame-level models (M1 ordinal, M2 attention MIL, M3 criteria
# two-stage) with the score gate and the within-slide criteria gate (CPU,
# stampede). See tickets/PHASE2-TICKETS.md P2-15, src/train/fit_atypia_v2.py.
# Held-out slides are NOT used here.
#
# Submit with:  [OUT_TAG]
#   PYTHONUNBUFFERED=1 sbatch --exclude=mscluster40,mscluster38 -J p2_15_fit slurm/fit_atypia_v2.slurm atypia_v2_uni2h
# =====================================================================

#SBATCH --job-name=p2_15_fit
#SBATCH --partition=stampede
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=04:00:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail

CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_ROOT=/datasets/mhoosen/stain-norm
OUT_TAG=${1:-atypia_v2_uni2h}
FEATS=${DATA_ROOT}/classifier/features
OUT_DIR=${DATA_ROOT}/classifier/${OUT_TAG}

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
export OMP_NUM_THREADS=8

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "x20=${FEATS}/uni2h_train.npz  x40=${FEATS}/uni2h_x40_train.npz  Out=${OUT_DIR}"
echo "=================================================================="

cd "${CODE_DIR}/src/train"
python fit_atypia_v2.py \
    --x20-features "${FEATS}/uni2h_train.npz" \
    --x40-features "${FEATS}/uni2h_x40_train.npz" \
    --out "${OUT_DIR}"

echo "------------------------------------------------------------------"
echo "Done: ${OUT_DIR}/v2_results.json"
