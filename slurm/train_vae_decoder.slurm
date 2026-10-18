#!/bin/bash
# =====================================================================
# train_vae_decoder.slurm -- P1-18: fine-tune the SD1.5-compatible VAE on H&E
# self-reconstruction (bigbatch GPU). Stage 1 = --mode decoder (encoder frozen,
# latents unchanged). Outputs to /datasets/mhoosen/stain-norm/vae/<RUN_NAME>/
# (final/ and best_val/ are diffusers save_pretrained dirs, loadable with
# `benchmark_vae_reconstruction.py --extra-vae label=<dir>`). Resumable: re-submit
# the same command and it resumes from state.pt.
#
# Positional args:  RUN_NAME [STEPS]
#   Smoke (5 steps, env/loop/leak/hash check only):
#     SMOKE=1 PYTHONUNBUFFERED=1 sbatch --exclude=<BAD_NODES> --time=00:20:00 -J p1_18_smoke slurm/train_vae_decoder.slurm d1_smoke 5
#   Mandatory capacity/bottleneck control (Stage 0.3) -- train AND validate on the same N crops:
#     OVERFIT_N=16 PYTHONUNBUFFERED=1 sbatch --exclude=<BAD_NODES> --time=00:45:00 -J p1_18_overfit slurm/train_vae_decoder.slurm d1_overfit16 600
#   Converged capacity control (no augmentation, higher lr, val every 200 steps):
#     OVERFIT_N=16 NO_AUGMENT=1 LR=1e-4 VAL_EVERY=200 PYTHONUNBUFFERED=1 sbatch --exclude=<BAD_NODES> --time=00:50:00 -J p1_18_overfit2 slurm/train_vae_decoder.slurm d1_overfit16_conv 2000
#   Real Stage 1 run:
#     PYTHONUNBUFFERED=1 sbatch --exclude=<BAD_NODES> -J p1_18_d1 slurm/train_vae_decoder.slurm d1_ftmse 3000
#
# Other env overrides (each read as ${VAR:-default}; default behaviour unchanged):
#   INIT=ft_mse|stock   MODE=decoder|full   LR=1e-5   BATCH=8   CROP=256|512
#   NO_AUGMENT=1 (no rot/flip; use for the capacity control)
#   LPIPS_W=0.3   L1_W=1.0   SSIM_W=0.0   ANCHOR_LAMBDA=0.0   MITOS_W=3.0   PANNUKE_W=1.0   VAL_EVERY=500
# =====================================================================

#SBATCH --job-name=train_vae_decoder
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
DATA_ROOT=/datasets/mhoosen/stain-norm
OUT_ROOT=${DATA_ROOT}/vae
export HF_HOME=/datasets/mhoosen/hf_cache
export TORCH_HOME=/datasets/mhoosen/torch_cache

RUN_NAME=${1:?"Usage: sbatch train_vae_decoder.slurm RUN_NAME [STEPS]"}
STEPS=${2:-3000}
OUT_DIR="${OUT_ROOT}/${RUN_NAME}"

EXTRA=""
if [[ "${SMOKE:-0}" == "1" ]]; then EXTRA="${EXTRA} --smoke"; fi
if [[ -n "${OVERFIT_N:-}" ]]; then EXTRA="${EXTRA} --overfit-n ${OVERFIT_N}"; fi
if [[ "${NO_AUGMENT:-0}" == "1" ]]; then EXTRA="${EXTRA} --no-augment"; fi

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "P1-18 -- run=${RUN_NAME} steps=${STEPS} mode=${MODE:-decoder} init=${INIT:-ft_mse} extra='${EXTRA}'"
echo "Out=${OUT_DIR}"
echo "=================================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed -- no GPU visible?"
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo 'ABORT: no GPU on this node -- resubmit.'; exit 1; }
python -c "import lpips" || { echo 'ABORT: lpips missing (see CLAUDE.md, installed 2026-10-05).'; exit 1; }
echo "------------------------------------------------------------------"

cd "${CODE_DIR}"
python src/train/train_vae_decoder.py \
    --mode          "${MODE:-decoder}" \
    --init          "${INIT:-ft_mse}" \
    --pool-dir      "${DATA_ROOT}/hist_lora_pool" \
    --pairs-dir     "${DATA_ROOT}/pairs/train" \
    --output-dir    "${OUT_DIR}" \
    --train-steps   "${STEPS}" \
    --batch-size    "${BATCH:-8}" \
    --crop          "${CROP:-256}" \
    --lr            "${LR:-1e-5}" \
    --l1-weight     "${L1_W:-1.0}" \
    --lpips-weight  "${LPIPS_W:-0.3}" \
    --ssim-weight   "${SSIM_W:-0.0}" \
    --anchor-lambda "${ANCHOR_LAMBDA:-0.0}" \
    --mitos-weight  "${MITOS_W:-3.0}"     --pannuke-weight "${PANNUKE_W:-1.0}" \
    --val-every     "${VAL_EVERY:-500}" \
    --seed 0 \
    ${EXTRA}

echo "------------------------------------------------------------------"
echo "Job $SLURM_JOB_ID finished. final/ and best_val/ in ${OUT_DIR}"
