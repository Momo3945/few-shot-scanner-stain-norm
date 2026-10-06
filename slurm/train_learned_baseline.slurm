#!/bin/bash
# =====================================================================
# train_learned_baseline.slurm -- StainNet / ParamNet / StainGAN baselines (P2-11b).
#
# Submit with positional args:  METHOD [DIRECTION]
#   sbatch train_learned_baseline.slurm stainnet A2H
#   sbatch train_learned_baseline.slurm paramnet A2H
#   sbatch train_learned_baseline.slurm staingan A2H
# METHOD = stainnet | paramnet | staingan      DIRECTION = A2H (default) | H2A
#
# Env-var overrides (unset -> defaults below):
#   SMOKE=1      5-step sanity run, writes to <out>_smoke (never clobbers a real run)
#   STEPS=N      override total steps (0 = method default)
#   REPEAT=K     stainnet only: passes over the pair set per epoch
#   RESUME=1     resume from <out>/last.pt
# StainNet needs only minutes: submit with --time=00:30:00. paramnet/staingan: default 3h,
# resumable (RESUME=1) if load-shedding kills it.
# Always add the bad-node --exclude list from CLAUDE.md on submission.
# =====================================================================

#SBATCH --job-name=learned_baseline
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_ROOT=/datasets/mhoosen/stain-norm
TRAIN_DIR=${DATA_ROOT}/pairs/train
OUT_ROOT=${DATA_ROOT}/baselines

METHOD=${1:?"Usage: sbatch train_learned_baseline.slurm METHOD [DIRECTION]  (METHOD = stainnet | paramnet | staingan)"}
DIRECTION=${2:-A2H}
STEPS=${STEPS:-0}
REPEAT=${REPEAT:-1}

EXTRA=""
OUT_DIR="${OUT_ROOT}/${DIRECTION,,}_${METHOD}"
if [[ "${SMOKE:-0}" == "1" ]]; then
    EXTRA="${EXTRA} --smoke"
    OUT_DIR="${OUT_DIR}_smoke"
fi
if [[ "${RESUME:-0}" == "1" ]]; then
    EXTRA="${EXTRA} --resume"
fi

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Method=$METHOD  Direction=$DIRECTION  Steps=$STEPS  Repeat=$REPEAT  Smoke=${SMOKE:-0}"
echo "Train=$TRAIN_DIR  Out=$OUT_DIR"
nvidia-smi -L || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "=================================================================="

# Fail fast on GPU-less/broken bigbatch nodes (see CLAUDE.md); the script also checks.
python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" || {
    echo "ABORT: no GPU on this node"; exit 1; }

cd "${CODE_DIR}"
python src/train/train_learned_baseline.py \
    --method    "${METHOD}" \
    --train-dir "${TRAIN_DIR}" \
    --out       "${OUT_DIR}" \
    --direction "${DIRECTION}" \
    --steps     "${STEPS}" \
    --repeat    "${REPEAT}" \
    ${EXTRA}

echo "------------------------------------------------------------------"
echo "Training done. Weights in ${OUT_DIR}"
echo "Next: sbatch slurm/infer_learned_baseline.slurm ${METHOD} ${DIRECTION}"
