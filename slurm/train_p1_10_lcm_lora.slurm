#!/bin/bash
# =====================================================================
# train_p1_10_lcm_lora.slurm -- P1-13: distil a task-specific LCM-LoRA from
# the frozen P1-10 teacher (lora/a2h_cond_r8/best) on bigbatch. NEVER writes
# into lora/a2h_cond_r8/ -- new checkpoint tag lora/a2h_cond_r8_lcm_distilled/,
# per the ticket's isolation requirement. See tickets/PHASE1-TICKETS.md P1-13
# and tickets/P1-13_task_specific_lcm_distillation.md.
#
# Submit with no positional args (steps only) -- defaults now match the
# adopted, successful P1-13 configuration (LR=0.00003, GRAD_ACCUM=4,
# SAVE_EVERY=125, 1500 steps -- see tickets/PHASE1-TICKETS.md P1-13's
# "Real 39-pair staged run" and "Decision (2026-08-24)" entries). Prior to
# this, the bare defaults (LR=0.0001, GRAD_ACCUM=1, 4000 steps) reproduced
# an earlier configuration that was found to oscillate/degrade and was
# rejected -- override LR/GRAD_ACCUM/SAVE_EVERY below only to deliberately
# reproduce that earlier, rejected configuration:
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 train_p1_10_lcm_lora.slurm            # default 1500 steps
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 train_p1_10_lcm_lora.slurm 1000       # 1000 steps
#
# Mandatory wiring control -- overfit-test an 8-pair subset BEFORE trusting a
# full run (set OVERFIT_N, not a positional arg, matching
# train_colour_translation_lora.slurm's own convention):
#   OVERFIT_N=8 sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 --time=00:45:00 -J p1_13_overfit train_p1_10_lcm_lora.slurm 300
#
# Smoke test (5 steps, env/loop check only, NOT the overfit control above):
#   SMOKE=1 sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 --time=00:15:00 -J p1_13_smoke train_p1_10_lcm_lora.slurm
#
# LR ablation (set LR, not a positional arg -- now defaults to 0.00003, the
# adopted value). Appends _lr<LR> to the output dir so different LR runs
# never collide/overwrite each other -- this is how the adopted checkpoint
# path lora/a2h_cond_r8_lcm_distilled_lr0.00003_ga4/ got its name. Override
# to reproduce the earlier, rejected LR=0.0001 run (found to oscillate, not
# steadily converge) -- see tickets/PHASE1-TICKETS.md P1-13:
#   OVERFIT_N=8 LR=0.0001 sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 --time=01:15:00 -J p1_13_overfit_lr1e4 train_p1_10_lcm_lora.slurm 1500
#
# Gradient-accumulation ablation (set GRAD_ACCUM, not a positional arg --
# now defaults to 4, the adopted value). Appends _ga<N> to the output dir.
# Adopted because LAB still oscillated between adjacent checkpoints even
# after LR=3e-5 alone stabilised SSIM -- pointed at per-step
# target/gradient variance (random timestep/index/guidance draw every
# microbatch) rather than LR as the remaining driver. --train-steps counts
# OPTIMIZER steps, not microbatches -- effective batch size = 1 x
# GRAD_ACCUM. Override to reproduce the earlier GRAD_ACCUM=1 run:
#   OVERFIT_N=8 LR=0.00003 GRAD_ACCUM=1 SAVE_EVERY=125 sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 --time=02:00:00 -J p1_13_overfit_lr3e5_ga1 train_p1_10_lcm_lora.slurm 1500
# =====================================================================

#SBATCH --job-name=p1_13_lcm_distill
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail

CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_DIR=/datasets/mhoosen/stain-norm/pairs/train
TEACHER_DIR=/datasets/mhoosen/stain-norm/lora/a2h_cond_r8/best
OUT_DIR=/datasets/mhoosen/stain-norm/lora/a2h_cond_r8_lcm_distilled
export HF_HOME=/datasets/mhoosen/hf_cache

STEPS=${1:-1500}
SMOKE_FLAG=""
if [[ "${SMOKE:-0}" == "1" ]]; then
    SMOKE_FLAG="--smoke"
fi
OVERFIT_FLAG=""
if [[ -n "${OVERFIT_N:-}" ]]; then
    OVERFIT_FLAG="--overfit-n ${OVERFIT_N}"
    OUT_DIR="${OUT_DIR}_overfit${OVERFIT_N}"
fi
LR="${LR:-0.00003}"
if [[ "${LR}" != "0.0001" ]]; then
    OUT_DIR="${OUT_DIR}_lr${LR}"
fi
GRAD_ACCUM="${GRAD_ACCUM:-4}"
if [[ "${GRAD_ACCUM}" != "1" ]]; then
    OUT_DIR="${OUT_DIR}_ga${GRAD_ACCUM}"
fi
SAVE_EVERY="${SAVE_EVERY:-125}"

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import transformers, peft, accelerate, safetensors, diffusers, cv2, skimage" 2>/dev/null; then
    echo "Installing missing Python deps..."
    pip install --quiet diffusers==0.39.0 transformers==5.14.1 peft==0.19.1 accelerate==1.14.0 safetensors==0.8.0 opencv-python-headless scikit-image
else
    echo "Python deps present."
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "P1-13 -- Steps=$STEPS  LR=$LR  GradAccum=$GRAD_ACCUM  SaveEvery=$SAVE_EVERY  Smoke='${SMOKE_FLAG}' Overfit='${OVERFIT_FLAG}'"
echo "Teacher=$TEACHER_DIR"
echo "Out=$OUT_DIR"
echo "=================================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed -- no GPU visible?"
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo 'ABORT: no GPU on this node -- resubmit.'; exit 1; }
echo "------------------------------------------------------------------"

cd "${CODE_DIR}"
python src/train/train_p1_10_lcm_lora.py \
    --pairs-dir   "${DATA_DIR}" \
    --teacher-dir "${TEACHER_DIR}" \
    --train-steps "${STEPS}" \
    --output-dir  "${OUT_DIR}" \
    --lr "${LR}" \
    --grad-accum-steps "${GRAD_ACCUM}" \
    --save-every "${SAVE_EVERY}" \
    --seed 0 \
    ${SMOKE_FLAG} ${OVERFIT_FLAG}

echo "------------------------------------------------------------------"
echo "Job $SLURM_JOB_ID finished. 'lcm' adapter in ${OUT_DIR}/final. Checkpoint selected by Validation B"
echo "(image-level teacher-matching, not the scalar loss) in ${OUT_DIR}/best_by_validation_b -- use that one."
