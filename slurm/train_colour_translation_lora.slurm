#!/bin/bash
# =====================================================================
# train_colour_translation_lora.slurm -- P1-10 corrective experiment:
# source-conditioned colour LoRA + fresh ControlNet, trained jointly, on
# bigbatch. NEVER writes into lora/a2h_r8/ or eval/a{0-5}*/ -- new checkpoint
# tag lora/<direction>_cond_r<rank>/, per the ticket's isolation requirement.
#
# Submit with positional args:  DIRECTION [RANK] [TRAIN_STEPS]
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 train_colour_translation_lora.slurm A2H 8            # rank 8, default steps
#
# Mandatory first control -- overfit-test an 8-pair subset BEFORE anything
# else in this ticket is trusted (set OVERFIT_N, not a positional arg, since
# it's a diagnostic mode not a normal run parameter):
#   OVERFIT_N=8 sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 --time=00:30:00 -J p1_10_overfit train_colour_translation_lora.slurm A2H 8 300
#
# Smoke test (5 steps, env/loop check only, NOT the overfit control above):
#   SMOKE=1 sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 --time=00:15:00 -J p1_10_smoke train_colour_translation_lora.slurm A2H 8
# =====================================================================

#SBATCH --job-name=colour_translation_lora
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail

CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_DIR=/datasets/mhoosen/stain-norm/pairs/train
OUT_ROOT=/datasets/mhoosen/stain-norm/lora
export HF_HOME=/datasets/mhoosen/hf_cache

DIRECTION=${1:?"Usage: sbatch train_colour_translation_lora.slurm DIRECTION [RANK] [STEPS]  (DIRECTION = A2H | H2A)"}
RANK=${2:-8}
STEPS=${3:-1000}
SMOKE_FLAG=""
if [[ "${SMOKE:-0}" == "1" ]]; then
    SMOKE_FLAG="--smoke"
fi
OVERFIT_FLAG=""
OUT_SUFFIX="cond_r${RANK}"
if [[ -n "${OVERFIT_N:-}" ]]; then
    OVERFIT_FLAG="--overfit-n ${OVERFIT_N}"
    OUT_SUFFIX="cond_r${RANK}_overfit${OVERFIT_N}"
fi

OUT_DIR="${OUT_ROOT}/${DIRECTION,,}_${OUT_SUFFIX}"

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import transformers, peft, accelerate, safetensors, diffusers, cv2" 2>/dev/null; then
    echo "Installing missing Python deps..."
    pip install --quiet transformers peft accelerate safetensors diffusers opencv-python-headless
else
    echo "Python deps present."
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "P1-10 -- Direction=$DIRECTION  Rank=$RANK  Steps=$STEPS  Smoke='${SMOKE_FLAG}' Overfit='${OVERFIT_FLAG}'"
echo "Out=$OUT_DIR"
echo "=================================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed -- no GPU visible?"
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo 'ABORT: no GPU on this node -- resubmit.'; exit 1; }
echo "------------------------------------------------------------------"

cd "${CODE_DIR}"
python src/train/train_colour_translation_lora.py \
    --pairs-dir   "${DATA_DIR}" \
    --direction   "${DIRECTION}" \
    --rank        "${RANK}" \
    --train-steps "${STEPS}" \
    --output-dir  "${OUT_DIR}" \
    --seed 0 \
    ${SMOKE_FLAG} ${OVERFIT_FLAG}

echo "------------------------------------------------------------------"
echo "Job $SLURM_JOB_ID finished. LoRA + ControlNet in ${OUT_DIR}/final"
echo "Next (mandatory control): source-conditioning ablation via infer_colour_translation.py"
