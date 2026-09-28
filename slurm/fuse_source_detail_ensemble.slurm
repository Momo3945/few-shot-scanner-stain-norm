#!/bin/bash
# =====================================================================
# fuse_source_detail_ensemble.slurm -- combines P1-11's self-ensembling
# (future-work #2) with P1-16's post-hoc F3 colour-residual fusion, to
# test whether the two techniques' SSIM gains compound. CPU, no model --
# reads already-generated outputs. See tickets/PHASE1-TICKETS.md P1-11
# and P1-16.
#
# Prerequisite: a full-scale P1-11 --mode identity run (for A_raw/A_V) and
# a full-scale self-ensembling run (infer_p1_11_ensemble.slurm, for
# H_pred) must already exist on the SAME held-out crops.
#
# Submit with:  IDENTITY_MANIFEST ENSEMBLE_MANIFEST OUT_TAG [SIGMA] [BETA]
#   IDENTITY_MANIFEST / ENSEMBLE_MANIFEST = paths under ${DATA_ROOT}/eval/,
#     relative to eval/ (e.g. p1_10_ddim_inversion/identity_full_heldout/
#     eval_manifest.csv, p1_11_ensemble/eta03_n3_full/eval_manifest.csv)
#   SIGMA/BETA default to 8/0.50 -- P1-16's own frozen F3 config, for a
#     direct apples-to-apples comparison against its 0.729 SSIM result.
#
#   sbatch fuse_source_detail_ensemble.slurm \
#     p1_10_ddim_inversion/identity_full_heldout/eval_manifest.csv \
#     p1_11_ensemble/eta03_n3_full/eval_manifest.csv \
#     p1_16_fusion_ensemble/f3_s8_b0.50_eta03
#
# Then score with the EXISTING scorer:
#   sbatch score_outputs.slurm p1_16_fusion_ensemble/f3_s8_b0.50_eta03
# =====================================================================

#SBATCH --job-name=p1_16_fuse_ensemble
#SBATCH --partition=stampede
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail

CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_ROOT=/datasets/mhoosen/stain-norm

IDENTITY_MANIFEST=${1:?"Usage: sbatch fuse_source_detail_ensemble.slurm IDENTITY_MANIFEST ENSEMBLE_MANIFEST OUT_TAG [SIGMA] [BETA]"}
ENSEMBLE_MANIFEST=${2:?"ENSEMBLE_MANIFEST required (relative to eval/)"}
OUT_TAG=${3:?"OUT_TAG required"}
SIGMA=${4:-8.0}
BETA=${5:-0.50}
OUT_DIR=${DATA_ROOT}/eval/${OUT_TAG}

mkdir -p "${CODE_DIR}/logs"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import cv2, scipy, skimage" 2>/dev/null; then
    pip install --quiet opencv-python-headless scipy scikit-image
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "P1-16+P1-11-ensemble fusion -- Sigma=$SIGMA Beta=$BETA"
echo "IdentityManifest=${DATA_ROOT}/eval/${IDENTITY_MANIFEST}"
echo "EnsembleManifest=${DATA_ROOT}/eval/${ENSEMBLE_MANIFEST}"
echo "Out=$OUT_DIR"
echo "=================================================================="

cd "${CODE_DIR}/src/eval"
python fuse_source_detail_ensemble.py \
    --identity-manifest "${DATA_ROOT}/eval/${IDENTITY_MANIFEST}" \
    --ensemble-manifest "${DATA_ROOT}/eval/${ENSEMBLE_MANIFEST}" \
    --sigma "${SIGMA}" --beta "${BETA}" \
    --out "${OUT_DIR}"

echo "------------------------------------------------------------------"
echo "Fusion done. Outputs + eval_manifest.csv in ${OUT_DIR}"
echo "Next: sbatch score_outputs.slurm ${OUT_TAG}"
