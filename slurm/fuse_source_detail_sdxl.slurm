#!/bin/bash
# =====================================================================
# fuse_source_detail_sdxl.slurm -- P3-08: P1-16-style raw-source-detail /
# learned colour-residual fusion applied to an SDXL source-conditioned run
# (CPU, no model -- reads already-generated PNGs). Mirrors
# fuse_source_detail.slurm exactly, but uses fuse_source_detail.py's new
# --vae-only-manifest convention (P3-08) instead of --identity-manifest
# (SD1.5/P1-11) -- infer_colour_translation_sdxl.py has no "identity mode"
# concept, so A_raw comes from the translate manifest's own aperio_path
# column and A_V comes from a separate plain --vae-only run. See
# tickets/PHASE3-TICKETS.md P3-08.
#
# Prerequisite: an infer_colour_translation_sdxl.py full held-out
# --source-mode correct run AND a --vae-only run on the SAME crop grid
# (same CROP, same held-out set) must already exist.
#
# Submit with:  METHOD VAE_ONLY_MANIFEST TRANSLATE_MANIFEST OUT_TAG [SIGMA] [ALPHA] [BETA]
#   METHOD = f1 | f2 | f3
#   VAE_ONLY_MANIFEST / TRANSLATE_MANIFEST = paths under ${DATA_ROOT}/eval/...,
#     relative to eval/ (e.g. p3_08_vae_only/eval_manifest.csv)
#
#   sbatch fuse_source_detail_sdxl.slurm f3 \
#     p3_08_vae_only/eval_manifest.csv \
#     p3_07c_full_heldout/eval_manifest.csv \
#     p3_08_fusion/f3_s8_b0.50 8 1.0 0.50
#
#   # H2A extension (P3-08 H2A ticket): DIRECTION env var, not positional --
#   # additive, defaults to A2H so the invocation above is unchanged. Must
#   # match the direction BOTH the vae-only and translate manifests were
#   # generated with.
#   DIRECTION=H2A sbatch fuse_source_detail_sdxl.slurm f3 \
#     p3_08_h2a_vae_only/eval_manifest.csv \
#     p3_07_h2a_full_heldout/eval_manifest.csv \
#     p3_08_h2a_fusion/f3_s8_b0.50 8 1.0 0.50
#
# Then score with the EXISTING scorer:
#   sbatch score_outputs.slurm p3_08_fusion/f3_s8_b0.50
# =====================================================================

#SBATCH --job-name=p3_08_fuse
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
APERIO_ROOT=${DATA_ROOT}/mitos_heldout
HELDOUT_CSV=${DATA_ROOT}/pairs/heldout_frames.csv
CROP=${CROP:-1024}

METHOD=${1:?"Usage: sbatch fuse_source_detail_sdxl.slurm METHOD VAE_ONLY_MANIFEST TRANSLATE_MANIFEST OUT_TAG [SIGMA] [ALPHA] [BETA]"}
VAE_ONLY_MANIFEST=${2:?"VAE_ONLY_MANIFEST required (relative to eval/)"}
TRANSLATE_MANIFEST=${3:?"TRANSLATE_MANIFEST required (relative to eval/)"}
OUT_TAG=${4:?"OUT_TAG required"}
SIGMA=${5:-4.0}
ALPHA=${6:-1.0}
BETA=${7:-1.0}
OUT_DIR=${DATA_ROOT}/eval/${OUT_TAG}
DIRECTION=${DIRECTION:-A2H}

mkdir -p "${CODE_DIR}/logs"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import cv2, scipy, skimage" 2>/dev/null; then
    pip install --quiet opencv-python-headless scipy scikit-image
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "P3-08 fusion -- Method=$METHOD Sigma=$SIGMA Alpha=$ALPHA Beta=$BETA Direction=$DIRECTION"
echo "VaeOnlyManifest=${DATA_ROOT}/eval/${VAE_ONLY_MANIFEST}"
echo "TranslateManifest=${DATA_ROOT}/eval/${TRANSLATE_MANIFEST}"
echo "Out=$OUT_DIR"
echo "=================================================================="

cd "${CODE_DIR}/src/eval"
# A_raw is re-derived (registration + grid-crop), not read from a saved
# file -- infer_colour_translation_sdxl.py never saves the raw source crop
# to disk (confirmed job 61392). --crop MUST match the translate/vae-only
# runs' own resolution.
python fuse_source_detail.py \
    --vae-only-manifest "${DATA_ROOT}/eval/${VAE_ONLY_MANIFEST}" \
    --translate-manifest "${DATA_ROOT}/eval/${TRANSLATE_MANIFEST}" \
    --aperio-root "${APERIO_ROOT}" --heldout-csv "${HELDOUT_CSV}" --crop "${CROP}" \
    --direction "${DIRECTION}" \
    --method "${METHOD}" --sigma "${SIGMA}" --alpha "${ALPHA}" --beta "${BETA}" \
    --out "${OUT_DIR}"

echo "------------------------------------------------------------------"
echo "Fusion done. Outputs + eval_manifest.csv in ${OUT_DIR}"
echo "Next: sbatch score_outputs.slurm ${OUT_TAG}"
