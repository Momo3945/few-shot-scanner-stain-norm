#!/bin/bash
# =====================================================================
# fuse_source_detail.slurm -- P1-16: raw-source-detail / learned colour-
# residual fusion (CPU, no model -- reads already-generated P1-11 outputs).
# See tickets/PHASE1-TICKETS.md P1-16 and
# tickets/P1-16_source_detail_colour_residual_fusion.md.
#
# Prerequisite: P1-11 identity-mode AND translate-mode (--source-mode
# correct) runs on the SAME internal-validation crops (PAIRS_DIR=1 via
# infer_p1_10_ddim_inversion.slurm) must already exist.
#
# Submit with:  METHOD IDENTITY_MANIFEST TRANSLATE_MANIFEST OUT_TAG [SIGMA] [ALPHA] [BETA]
#   METHOD = f1 | f2 | f3
#   IDENTITY_MANIFEST / TRANSLATE_MANIFEST = paths under
#     ${DATA_ROOT}/eval/..., relative to eval/ (e.g.
#     p1_10_ddim_inversion/identity_pairs_inv100/eval_manifest.csv)
#
#   sbatch fuse_source_detail.slurm f3 \
#     p1_10_ddim_inversion/identity_pairs_inv100/eval_manifest.csv \
#     p1_10_ddim_inversion/translate_pairs_inv100_correct/eval_manifest.csv \
#     p1_16_fusion/f3_s8_b0.75 8 1.0 0.75
#
# Then score with the EXISTING scorer:
#   sbatch score_outputs.slurm p1_16_fusion/f3_s8_b0.75
# =====================================================================

#SBATCH --job-name=p1_16_fuse
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

METHOD=${1:?"Usage: sbatch fuse_source_detail.slurm METHOD IDENTITY_MANIFEST TRANSLATE_MANIFEST OUT_TAG [SIGMA] [ALPHA] [BETA]"}
IDENTITY_MANIFEST=${2:?"IDENTITY_MANIFEST required (relative to eval/)"}
TRANSLATE_MANIFEST=${3:?"TRANSLATE_MANIFEST required (relative to eval/)"}
OUT_TAG=${4:?"OUT_TAG required"}
SIGMA=${5:-4.0}
ALPHA=${6:-1.0}
BETA=${7:-1.0}
OUT_DIR=${DATA_ROOT}/eval/${OUT_TAG}

mkdir -p "${CODE_DIR}/logs"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import cv2, scipy, skimage" 2>/dev/null; then
    pip install --quiet opencv-python-headless scipy scikit-image
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "P1-16 fusion -- Method=$METHOD Sigma=$SIGMA Alpha=$ALPHA Beta=$BETA"
echo "IdentityManifest=${DATA_ROOT}/eval/${IDENTITY_MANIFEST}"
echo "TranslateManifest=${DATA_ROOT}/eval/${TRANSLATE_MANIFEST}"
echo "Out=$OUT_DIR"
echo "=================================================================="

cd "${CODE_DIR}/src/eval"
python fuse_source_detail.py \
    --identity-manifest "${DATA_ROOT}/eval/${IDENTITY_MANIFEST}" \
    --translate-manifest "${DATA_ROOT}/eval/${TRANSLATE_MANIFEST}" \
    --method "${METHOD}" --sigma "${SIGMA}" --alpha "${ALPHA}" --beta "${BETA}" \
    --out "${OUT_DIR}"

echo "------------------------------------------------------------------"
echo "Fusion done. Outputs + eval_manifest.csv in ${OUT_DIR}"
echo "Next: sbatch score_outputs.slurm ${OUT_TAG}"
