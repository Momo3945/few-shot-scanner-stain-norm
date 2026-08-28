#!/bin/bash
# =====================================================================
# score_atypia_classifier.slurm -- score the trained atypia classifier against
# every method's existing eval outputs (P2-09, Clinical Utility, GPU).
#
# Requires train_atypia_classifier.slurm's best.pt to already exist, and the
# testing-side atypia_manifest.csv already built (build_atypia_manifest.py on
# mitos_heldout/mitos_atypia_2014_testing_aperio -- already on the cluster, no
# data transfer needed for this side).
#
# GPU (bigbatch), not stampede CPU: ResNet18 inference over ~15-20k crops across
# 9 methods + raw_hamamatsu/raw_aperio is cheap but still faster on GPU.
#
# Submit with:  [CHECKPOINT] [OUT_TAG]
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 score_atypia_classifier.slurm
# =====================================================================

#SBATCH --job-name=score_atypia
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail

CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_ROOT=/datasets/mhoosen/stain-norm
TESTING_ROOT=${DATA_ROOT}/mitos_heldout
export TORCH_HOME=/datasets/mhoosen/torch_cache

CHECKPOINT=${1:-${DATA_ROOT}/classifier/atypia_r18/best.pt}
OUT_TAG=${2:-atypia_classifier_scores}
OUT_DIR=${DATA_ROOT}/eval/${OUT_TAG}

# Optional override: TAGS="tag1 tag2 ..." (space-separated, relative to
# ${DATA_ROOT}/eval, may include subdirs e.g. "p1_10_ddim_inversion/p1_10_ddim_inversion_full").
# Unset/empty -> script's own DEFAULT_TAGS (the original 26-method A0-A5/baseline
# run) is used, so every prior invocation of this launcher is unaffected.
TAGS_ARGS=()
if [[ -n "${TAGS:-}" ]]; then
    read -ra TAGS_ARGS <<< "${TAGS}"
fi

# P2-12 hardening: score_atypia_classifier.py now refuses to score a tag whose
# translation direction can't be resolved or doesn't match --expect-direction
# (default H2A -- what the clinical-utility claim actually needs; P2-12 S:0 found
# every method scored to date was really A2H). Every prior/default invocation of
# this launcher is unaffected -- these are all optional, unset by default.
EXPECT_DIRECTION_ARGS=()
if [[ -n "${EXPECT_DIRECTION:-}" ]]; then
    EXPECT_DIRECTION_ARGS=(--expect-direction "${EXPECT_DIRECTION}")
fi
TAG_DIRECTION_ARGS=()
if [[ -n "${TAG_DIRECTION:-}" ]]; then
    read -ra TAG_DIRECTION_ARGS <<< "${TAG_DIRECTION}"
    TAG_DIRECTION_ARGS=(--tag-direction "${TAG_DIRECTION_ARGS[@]}")
fi
ALLOW_MISMATCH_ARGS=()
if [[ -n "${ALLOW_DIRECTION_MISMATCH:-}" ]]; then
    ALLOW_MISMATCH_ARGS=(--allow-direction-mismatch)
fi

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import torch, torchvision, sklearn, cv2, tifffile" 2>/dev/null; then
    echo "Installing missing Python deps..."
    pip install --quiet torchvision scikit-learn opencv-python-headless tifffile
else
    echo "Python deps present."
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "ABORT: ${CHECKPOINT} not found -- run train_atypia_classifier.slurm first."
    exit 1
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Checkpoint=$CHECKPOINT  TestingRoot=$TESTING_ROOT"
echo "Out=$OUT_DIR"
echo "=================================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import torch; print('cuda available:', torch.cuda.is_available())"
echo "------------------------------------------------------------------"

cd "${CODE_DIR}/src/eval"
CMD=(python score_atypia_classifier.py \
    --checkpoint       "${CHECKPOINT}" \
    --testing-manifest "${DATA_ROOT}/pairs/atypia_test_manifest.csv" \
    --testing-root     "${TESTING_ROOT}" \
    --eval-root        "${DATA_ROOT}/eval" \
    --out              "${OUT_DIR}")
if [[ ${#TAGS_ARGS[@]} -gt 0 ]]; then
    CMD+=(--tags "${TAGS_ARGS[@]}")
fi
CMD+=("${EXPECT_DIRECTION_ARGS[@]}" "${TAG_DIRECTION_ARGS[@]}" "${ALLOW_MISMATCH_ARGS[@]}")
"${CMD[@]}"

echo "------------------------------------------------------------------"
echo "Scoring done. per_crop.csv + per_frame.csv (PRIMARY) + summary.csv + integrity_report.csv in ${OUT_DIR}"
