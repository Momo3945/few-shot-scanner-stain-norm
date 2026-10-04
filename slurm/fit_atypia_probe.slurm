#!/bin/bash
# =====================================================================
# fit_atypia_probe.slurm -- P2-14 step 2: nested leave-one-slide-out CV + final
# head for the frozen-encoder atypia probe (CPU, stampede). See
# tickets/PHASE2-TICKETS.md P2-14, src/train/fit_atypia_probe.py.
#
# Requires extract_atypia_features.slurm outputs (train; test is optional and
# scored once as a sanity check only).
#
# Submit with:  [FEATURE_TAG] [OUT_TAG]
#   sbatch -J p2_14_fit slurm/fit_atypia_probe.slurm uni2h atypia_probe_uni2h
#   (reads classifier/features/<FEATURE_TAG>_train.npz and _test.npz)
# =====================================================================

#SBATCH --job-name=p2_14_fit
#SBATCH --partition=stampede
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail

CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_ROOT=/datasets/mhoosen/stain-norm

FEATURE_TAG=${1:-uni2h}
OUT_TAG=${2:-atypia_probe_${FEATURE_TAG}}
FEATS=${DATA_ROOT}/classifier/features
OUT_DIR=${DATA_ROOT}/classifier/${OUT_TAG}

HELD_ARGS=()
if [[ -f "${FEATS}/${FEATURE_TAG}_test.npz" ]]; then
    HELD_ARGS=(--heldout-features "${FEATS}/${FEATURE_TAG}_test.npz")
fi

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Features=${FEATS}/${FEATURE_TAG}_train.npz  Out=${OUT_DIR}  Held=${HELD_ARGS[*]:-none}"
echo "=================================================================="

cd "${CODE_DIR}/src/train"
python fit_atypia_probe.py --features "${FEATS}/${FEATURE_TAG}_train.npz" \
    "${HELD_ARGS[@]}" --out "${OUT_DIR}"

echo "------------------------------------------------------------------"
echo "Done: ${OUT_DIR}/probe_results.json + head.npz"
