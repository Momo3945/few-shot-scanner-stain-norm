#!/bin/bash
# =====================================================================
# score_camelyon_wasserstein.slurm -- pairwise LAB-Wasserstein between all 5
# CAMELYON17 centres, pooled per centre (CPU). P2-10, multi-centre
# generalisation. Run once for D_pre directly against the raw patches (no GPU
# job needed, no dependency on normalize_camelyon.slurm), and again for D_post
# against normalize_camelyon.slurm's output with AGAINST_PAIRWISE_CSV pointing
# at the D_pre run's pairwise.csv.
#
# Takes a full ROOT_DIR path (not a short tag) because the raw patches and the
# normalised output live under DIFFERENT prefixes: raw CAMELYON17 patches were
# uploaded directly under DATA_ROOT (source data, not an eval output), while
# normalize_camelyon.slurm writes its output under DATA_ROOT/eval/<tag>
# (matching every other normalize_*.slurm convention in this project) -- a
# single ${DATA_ROOT}/${TAG} template can't handle both, so this launcher
# takes the caller-resolved root explicitly instead of guessing.
#
# Submit with:  ROOT_DIR OUT_TAG [AGAINST_PAIRWISE_CSV]
#   sbatch score_camelyon_wasserstein.slurm /datasets/mhoosen/stain-norm/camelyon17_patches camelyon17_patches
#       # D_pre: raw patches
#   sbatch score_camelyon_wasserstein.slurm /datasets/mhoosen/stain-norm/eval/camelyon17_normalised camelyon17_normalised \
#       /datasets/mhoosen/stain-norm/eval/camelyon17_patches/pairwise.csv
#       # D_post: normalised patches, diffed against the D_pre run
#
# CPU-only (pure numpy/scipy/opencv, no GPU), runs in the plain stainnorm env
# on stampede.
# =====================================================================

#SBATCH --job-name=score_camelyon
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

ROOT_DIR=${1:?"Usage: sbatch score_camelyon_wasserstein.slurm ROOT_DIR OUT_TAG [AGAINST_PAIRWISE_CSV]"}
OUT_TAG=${2:?"Usage: sbatch score_camelyon_wasserstein.slurm ROOT_DIR OUT_TAG [AGAINST_PAIRWISE_CSV]"}
AGAINST=${3:-}

OUT_DIR=${DATA_ROOT}/eval/${OUT_TAG}

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import numpy, scipy, cv2" 2>/dev/null; then
    pip install --quiet numpy scipy opencv-python-headless
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Root=$ROOT_DIR  Out=$OUT_DIR  Against=${AGAINST:-<none, D_pre run>}"
echo "=================================================================="

cd "${CODE_DIR}/src/eval"
AGAINST_ARG=""
if [[ -n "$AGAINST" ]]; then
    if [[ ! -f "$AGAINST" ]]; then
        echo "ABORT: --against pairwise.csv not found at ${AGAINST} -- score the D_pre run first."
        exit 1
    fi
    AGAINST_ARG="--against ${AGAINST}"
fi

python score_camelyon_wasserstein.py \
    --root "${ROOT_DIR}" \
    --out  "${OUT_DIR}" \
    ${AGAINST_ARG}

echo "------------------------------------------------------------------"
echo "Scoring done. pairwise.csv + summary.csv in ${OUT_DIR}"
