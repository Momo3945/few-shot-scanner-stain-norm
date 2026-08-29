#!/bin/bash
# =====================================================================
# d5_infer_lora_scale_sweep_p3_07.slurm -- P3-07 diagnostic D5: colour-LoRA
# adapter-weight sweep {1.0, 1.25, 1.5, 2.0} at fixed strength=0.50,
# controlnet-scale=1.0, guidance=2.0, 50-step DDIM, condition-mode=rgb_canny
# (correct RGB+Canny conditioning) -- SAME internal validation split D4
# used (11 A03/H03 pairs, 5 frames held out for val in the training run's
# own pair_manifest.json), same default seeds. Never touches
# A06/A08/A09/A13/A16.
#
# Motivated by D2: the colour LoRA already moves LAB ~3 units in the
# correct direction at its trained weight (1.0) but isn't strong enough to
# overcome the negative frozen-base+ControlNet drift. This tests whether
# simply amplifying the LoRA's influence at inference (via diffusers
# set_adapters, infer_colour_translation_sdxl.py's new --lora-scale flag --
# no retraining) pushes recovery further toward zero/positive.
#
# On bigbatch (GPU). See tickets/PHASE3-TICKETS.md P3-07, D5.
#
# Submit with no args:
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44,mscluster61 \
#     -J d5_lora_scale_sweep_p3_07 d5_infer_lora_scale_sweep_p3_07.slurm
# =====================================================================

#SBATCH --job-name=d5_lora_scale_sweep_p3_07
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail

CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_ROOT=/datasets/mhoosen/stain-norm
PAIRS_DIR=${DATA_ROOT}/pairs/train_1024
CHECKPOINT_DIR=${DATA_ROOT}/lora/a2h_cond_r8_sdxl_1024/final
VAL_FRAMES_JSON=${DATA_ROOT}/lora/a2h_cond_r8_sdxl_1024/pair_manifest.json
export HF_HOME=/datasets/mhoosen/hf_cache

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import transformers, peft, accelerate, safetensors, diffusers, cv2" 2>/dev/null; then
    pip install --quiet transformers peft accelerate safetensors diffusers opencv-python-headless
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "P3-07 D5 -- colour-LoRA-scale sweep on internal validation"
echo "Pairs dir: ${PAIRS_DIR}"
echo "Val frames: ${VAL_FRAMES_JSON}"
echo "=================================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo 'ABORT: no GPU on this node -- resubmit.'; exit 1; }
if [[ ! -f "${CHECKPOINT_DIR}/pytorch_lora_weights.safetensors" ]]; then
    echo "ABORT: ${CHECKPOINT_DIR}/pytorch_lora_weights.safetensors not found."
    exit 1
fi
echo "------------------------------------------------------------------"

cd "${CODE_DIR}/src/eval"
for LORA_SCALE in 1.0 1.25 1.5 2.0; do
    OUT_DIR=${DATA_ROOT}/eval/p3_07_d5_lora_${LORA_SCALE}
    mkdir -p "${OUT_DIR}"
    echo "-- lora_scale=${LORA_SCALE} --"
    python infer_colour_translation_sdxl.py \
        --lora "${CHECKPOINT_DIR}" --controlnet "${CHECKPOINT_DIR}/controlnet" \
        --source-mode correct --lora-scale "${LORA_SCALE}" \
        --pairs-dir "${PAIRS_DIR}" --val-frames-json "${VAL_FRAMES_JSON}" \
        --out "${OUT_DIR}" \
        --steps 50 --strength 0.50 --guidance 2.0 --controlnet-scale 1.0
done

echo "------------------------------------------------------------------"
echo "D5 lora-scale sweep done. eval/p3_07_d5_lora_{1.0,1.25,1.5,2.0}/"
echo "Next: score_d5_p3_07.slurm"
