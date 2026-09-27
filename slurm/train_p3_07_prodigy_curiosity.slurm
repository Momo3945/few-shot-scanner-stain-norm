#!/bin/bash
# =====================================================================
# train_p3_07_prodigy_curiosity.slurm -- informal curiosity test, NOT a
# ticketed experiment. See src/train/train_p3_07_prodigy_curiosity.py's
# docstring for full context: same SDXL source-conditioned colour-LoRA +
# fresh 6-channel ControlNet architecture as P3-06/P3-07/P3-07b/P3-07c,
# the ONLY variable changed is the optimizer (AdamW -> Prodigy,
# learning-rate-free). Defaults to the same pairs-dir as P3-07b
# (train_1024_full96, this project's best AdamW result so far) so a
# loss_log.csv/training_config.json comparison is at least apples-to-apples
# on the data side -- override PAIRS_DIR to compare against P3-07c's
# overlap144 instead. New checkpoint tag
# lora/a2h_cond_r8_sdxl_1024_full96_prodigy/ -- never touches any existing
# AdamW checkpoint (full96, overlap144, or the original P3-07 one).
#
# Submit with no args (real run):
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44,mscluster75,mscluster74,mscluster51,mscluster83,mscluster76,mscluster50,mscluster45,mscluster57,mscluster59,mscluster62 \
#     -J p3_07_prodigy train_p3_07_prodigy_curiosity.slurm
#
# Smoke test first (5 steps, env/loop check + confirms prodigyopt installs
# and Prodigy's own "d" self-estimated step size actually logs something
# sane -- mandatory before a real run, since this is new, unvalidated code):
#   SMOKE=1 sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44,mscluster75,mscluster74,mscluster51,mscluster83,mscluster76,mscluster50,mscluster45,mscluster57,mscluster59,mscluster62 \
#     --time=00:30:00 -J p3_07_prodigy_smoke train_p3_07_prodigy_curiosity.slurm
#
# Env-var overrides (this project's standard non-breaking-change pattern):
#   PAIRS_DIR=<path>     alternate pairs dir (default: train_1024_full96)
#   OUT_TAG=<name>       alternate checkpoint dir name under lora/
#   PRODIGY_LR=<val>     Prodigy's lr SCALE, not a literal step size (default 1.0)
#   PRODIGY_SLICE_P=<n>  memory-saving approximation knob (default 11 -- REQUIRED
#                        at n=1 on this hardware: job 60753's smoke test OOM'd at
#                        step 4 with slice_p=1, GPU already at 23.49/23.55 GiB
#                        even under the validated AdamW P3-07c run)
# =====================================================================

#SBATCH --job-name=p3_07_prodigy
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail

CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_DIR=${PAIRS_DIR:-/datasets/mhoosen/stain-norm/pairs/train_1024_full96}
RESOLUTION=1024
RANK=8
STEPS=4000
OUT_TAG=${OUT_TAG:-a2h_cond_r8_sdxl_1024_full96_prodigy}
OUT_DIR=/datasets/mhoosen/stain-norm/lora/${OUT_TAG}
PRODIGY_LR=${PRODIGY_LR:-1.0}
PRODIGY_SLICE_P=${PRODIGY_SLICE_P:-11}
export HF_HOME=/datasets/mhoosen/hf_cache

SMOKE_FLAG=""
if [[ "${SMOKE:-0}" == "1" ]]; then
    SMOKE_FLAG="--smoke"
fi

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import transformers, peft, accelerate, safetensors, diffusers, cv2" 2>/dev/null; then
    echo "Installing missing Python deps..."
    pip install --quiet transformers peft accelerate safetensors diffusers opencv-python-headless
else
    echo "Python deps present."
fi
if ! python -c "import prodigyopt" 2>/dev/null; then
    echo "Installing prodigyopt (not part of the base stainnorm env)..."
    pip install --quiet prodigyopt
else
    echo "prodigyopt present."
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Informal curiosity test: Prodigy optimizer swap on P3-07-family SDXL source-conditioning"
echo "NOT a ticketed experiment -- see script docstring."
echo "PairsDir=$DATA_DIR  Rank=$RANK  Steps=$STEPS  Resolution=$RESOLUTION  ProdigyLR=$PRODIGY_LR  ProdigySliceP=$PRODIGY_SLICE_P  Smoke='${SMOKE_FLAG}'"
echo "Out=$OUT_DIR"
echo "=================================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed -- no GPU visible?"
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo 'ABORT: no GPU on this node -- resubmit.'; exit 1; }
echo "------------------------------------------------------------------"

cd "${CODE_DIR}"
python src/train/train_p3_07_prodigy_curiosity.py \
    --pairs-dir   "${DATA_DIR}" \
    --direction   A2H \
    --rank        "${RANK}" \
    --train-steps "${STEPS}" \
    --resolution  "${RESOLUTION}" \
    --output-dir  "${OUT_DIR}" \
    --lr          "${PRODIGY_LR}" \
    --prodigy-slice-p "${PRODIGY_SLICE_P}" \
    --seed 0 \
    ${SMOKE_FLAG}

echo "------------------------------------------------------------------"
echo "Job $SLURM_JOB_ID finished. LoRA + ControlNet in ${OUT_DIR}/final"
echo "Informal curiosity test -- NOT ticketed, NOT H1 evidence. Compare loss_log.csv/"
echo "training_config.json's val_loss against P3-07b's own AdamW run on the same pairs-dir yourself."
echo "Next (mandatory before trusting this run for anything beyond curiosity): source-conditioning"
echo "ablation via infer_colour_translation_sdxl.py --source-mode {correct,zero,shuffled}."
