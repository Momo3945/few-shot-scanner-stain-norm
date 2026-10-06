#!/bin/bash
# =====================================================================
# infer_learned_baseline.slurm -- run a trained StainNet/ParamNet/StainGAN over the
# held-out set (P2-11b). Writes eval/<tag>/eval_manifest.csv for score_outputs.slurm.
#
# Submit with positional args:  METHOD [DIRECTION] [LIMIT]
#   sbatch infer_learned_baseline.slurm stainnet A2H         # -> eval/stainnet
#   sbatch infer_learned_baseline.slurm paramnet H2A         # -> eval/h2a_paramnet
#   sbatch infer_learned_baseline.slurm staingan A2H 2       # smoke, 2 frames
# Env: SMOKE=1 reads the *_smoke checkpoint dir and writes eval/<tag>_smoke.
# Then: sbatch slurm/score_outputs.slurm <tag>
# Always add the bad-node --exclude list from CLAUDE.md on submission.
# =====================================================================

#SBATCH --job-name=infer_learned
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_ROOT=/datasets/mhoosen/stain-norm
MITOS_ROOT=${DATA_ROOT}/mitos_heldout
HELDOUT=${DATA_ROOT}/pairs/heldout_frames.csv

METHOD=${1:?"Usage: sbatch infer_learned_baseline.slurm METHOD [DIRECTION] [LIMIT]"}
DIRECTION=${2:-A2H}
LIMIT=${3:-0}

# eval tag convention matches the classical baselines: A2H -> <method>, H2A -> h2a_<method>
if [[ "${DIRECTION}" == "H2A" ]]; then TAG="h2a_${METHOD}"; else TAG="${METHOD}"; fi
CK_DIR="${DATA_ROOT}/baselines/${DIRECTION,,}_${METHOD}"
if [[ "${SMOKE:-0}" == "1" ]]; then CK_DIR="${CK_DIR}_smoke"; TAG="${TAG}_smoke"; fi
CKPT="${CK_DIR}/${METHOD}_${DIRECTION,,}.pt"
OUT_DIR="${DATA_ROOT}/eval/${TAG}"

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"
[[ -f "${CKPT}" ]] || { echo "ABORT: checkpoint not found: ${CKPT}"; exit 1; }

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import cv2, skimage" 2>/dev/null; then
    pip install --quiet opencv-python-headless scikit-image
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Method=$METHOD  Direction=$DIRECTION  Limit=$LIMIT  Tag=$TAG"
echo "Ckpt=$CKPT  Out=$OUT_DIR"
nvidia-smi -L || true
echo "=================================================================="

python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" || {
    echo "ABORT: no GPU on this node"; exit 1; }

cd "${CODE_DIR}"
python src/eval/infer_learned_baseline.py \
    --method    "${METHOD}" \
    --ckpt      "${CKPT}" \
    --root      "${MITOS_ROOT}" \
    --heldout   "${HELDOUT}" \
    --direction "${DIRECTION}" \
    --out       "${OUT_DIR}" \
    --limit     "${LIMIT}"

echo "------------------------------------------------------------------"
echo "Inference done. Next: sbatch slurm/score_outputs.slurm ${TAG}"
