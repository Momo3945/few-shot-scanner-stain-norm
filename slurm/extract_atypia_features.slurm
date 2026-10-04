#!/bin/bash
# =====================================================================
# extract_atypia_features.slurm -- P2-14 step 1: frozen UNI2-h tile embeddings
# for the atypia probe (GPU). See tickets/PHASE2-TICKETS.md P2-14.
#
# Submit with:  SPLIT [OUT_TAG]      SPLIT = train | test
#   SMOKE=1 sbatch --exclude=... --time=00:30:00 -J p2_14_feat_smoke extract_atypia_features.slurm train uni2h_smoke
#   sbatch --exclude=... -J p2_14_feat_train extract_atypia_features.slurm train uni2h_train
#   sbatch --exclude=... -J p2_14_feat_test  extract_atypia_features.slurm test  uni2h_test
# test = held-out Aperio, used only for the final sanity check (never fit/select).
# =====================================================================

#SBATCH --job-name=p2_14_features
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail

CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_ROOT=/datasets/mhoosen/stain-norm
export HF_HOME=/datasets/mhoosen/hf_cache

SPLIT=${1:?"Usage: sbatch extract_atypia_features.slurm SPLIT(train|test) [OUT_TAG]"}
OUT_TAG=${2:-uni2h_${SPLIT}}
case "$SPLIT" in
    train) MANIFEST=${DATA_ROOT}/pairs/atypia_train_manifest.csv
           ROOT=${DATA_ROOT}/mitos_atypia_train_aperio ;;
    test)  MANIFEST=${DATA_ROOT}/pairs/atypia_test_manifest.csv
           ROOT=${DATA_ROOT}/mitos_heldout/mitos_atypia_2014_testing_aperio ;;
    *) echo "SPLIT must be train or test"; exit 1 ;;
esac
OUT=${DATA_ROOT}/classifier/features/${OUT_TAG}.npz
LIMIT_ARGS=()
if [[ "${SMOKE:-0}" == "1" ]]; then
    LIMIT_ARGS=(--limit-frames 4)
fi

mkdir -p "${CODE_DIR}/logs" "${DATA_ROOT}/classifier/features"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Split=$SPLIT  Manifest=$MANIFEST  Root=$ROOT"
echo "Out=$OUT"
echo "=================================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo 'ABORT: no GPU on this node -- resubmit.'; exit 1; }
echo "------------------------------------------------------------------"

cd "${CODE_DIR}/src/train"
python extract_atypia_features.py \
    --manifest "${MANIFEST}" --root "${ROOT}" --out "${OUT}" "${LIMIT_ARGS[@]}"

echo "------------------------------------------------------------------"
echo "Features done: ${OUT} (+ .json sidecar)"
