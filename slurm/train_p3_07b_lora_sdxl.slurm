#!/bin/bash
# =====================================================================
# train_p3_07b_lora_sdxl.slurm -- P3-07b: bounded >50-pair training
# follow-up to P3-07 (see src/train/train_p3_07b_lora_sdxl.py's docstring
# for full motivation/scope-note -- SUPPLEMENTARY, not H1 evidence).
# Identical architecture/hyperparameters to P3-07 (rank 8, 4000 steps,
# resolution 1024, A2H, A03/H03-only); the only variable is training-set
# size: all 96 non-overlapping A03/H03 1024 pairs instead of the capped 50.
# New checkpoint tag lora/a2h_cond_r8_sdxl_1024_full96/ -- never touches
# lora/a2h_cond_r8_sdxl_1024/ (P3-07) or any other existing checkpoint.
# See tickets/PHASE3-TICKETS.md P3-07 (P3-07b proposal).
#
# Submit with no args (real run, all defaults matching P3-07):
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44,mscluster61 \
#     -J p3_07b_train train_p3_07b_lora_sdxl.slurm
#
# Smoke test first (5 steps, env/loop check only -- the architecture itself
# was already validated by P3-07's own smoke/overfit-8/ablation controls;
# this run only changes pair count, so a smoke test is the only new-script
# sanity check needed before trusting the real run):
#   SMOKE=1 sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44,mscluster61 \
#     --time=00:30:00 -J p3_07b_smoke train_p3_07b_lora_sdxl.slurm
# =====================================================================

#SBATCH --job-name=p3_07b_train
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
DATA_DIR=/datasets/mhoosen/stain-norm/pairs/train_1024_full96
RESOLUTION=1024
RANK=8
STEPS=4000
OUT_DIR=/datasets/mhoosen/stain-norm/lora/a2h_cond_r8_sdxl_1024_full96
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

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "P3-07b (supplementary >50-pair follow-up -- NOT H1 evidence, see script docstring)"
echo "PairsDir=$DATA_DIR  Rank=$RANK  Steps=$STEPS  Resolution=$RESOLUTION  Smoke='${SMOKE_FLAG}'"
echo "Out=$OUT_DIR"
echo "=================================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed -- no GPU visible?"
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo 'ABORT: no GPU on this node -- resubmit.'; exit 1; }
echo "------------------------------------------------------------------"

cd "${CODE_DIR}"
python src/train/train_p3_07b_lora_sdxl.py \
    --pairs-dir   "${DATA_DIR}" \
    --direction   A2H \
    --rank        "${RANK}" \
    --train-steps "${STEPS}" \
    --resolution  "${RESOLUTION}" \
    --output-dir  "${OUT_DIR}" \
    --seed 0 \
    ${SMOKE_FLAG}

echo "------------------------------------------------------------------"
echo "Job $SLURM_JOB_ID finished. LoRA + ControlNet in ${OUT_DIR}/final"
echo "P3-07b is supplementary (>50 pairs) -- NOT evidence for/against the formal H1 claim."
echo "Next (mandatory control): source-conditioning ablation via infer_colour_translation_sdxl.py"
