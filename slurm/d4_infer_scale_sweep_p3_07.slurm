#!/bin/bash
# =====================================================================
# d4_infer_scale_sweep_p3_07.slurm -- P3-07 diagnostic D4: small
# controlnet_conditioning_scale grid {0.25, 0.50, 0.75, 1.00} at fixed
# strength=0.50/steps=50/guidance=2.0, on the INTERNAL VALIDATION split
# (the 11 A03/H03 pairs the P3-07 1024 training run held out for val --
# NOT the A06/A08/A09/A13/A16 held-out test set D1-D3 used). Per this
# project's methodology guardrail ("Do not tune on the full held-out
# set"), D4 must never touch held-out slides. Uses
# infer_colour_translation_sdxl.py's new --val-frames-json flag, which
# filters --pairs-dir to exactly the frame_ids recorded as val_frames in
# the trained checkpoint's own pair_manifest.json. Trained colour LoRA
# stays enabled (D2 already isolated that question); condition-mode stays
# the trained default (rgb_canny). Runs all 4 scales sequentially in one
# job -- each is only 11 pairs x 3 seeds = 33 outputs, order of minutes,
# not hours. On bigbatch (GPU). See tickets/PHASE3-TICKETS.md P3-07, D4.
#
# Submit with no args:
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44,mscluster61 \
#     -J d4_scale_sweep_p3_07 d4_infer_scale_sweep_p3_07.slurm
# =====================================================================

#SBATCH --job-name=d4_scale_sweep_p3_07
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
echo "P3-07 D4 -- controlnet_conditioning_scale sweep on internal validation"
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
for SCALE in 0.25 0.50 0.75 1.00; do
    OUT_DIR=${DATA_ROOT}/eval/p3_07_d4_scale_${SCALE}
    mkdir -p "${OUT_DIR}"
    echo "-- controlnet_conditioning_scale=${SCALE} --"
    python infer_colour_translation_sdxl.py \
        --lora "${CHECKPOINT_DIR}" --controlnet "${CHECKPOINT_DIR}/controlnet" \
        --source-mode correct \
        --pairs-dir "${PAIRS_DIR}" --val-frames-json "${VAL_FRAMES_JSON}" \
        --out "${OUT_DIR}" \
        --steps 50 --strength 0.50 --guidance 2.0 --controlnet-scale "${SCALE}"
done

echo "------------------------------------------------------------------"
echo "D4 scale sweep done. eval/p3_07_d4_scale_{0.25,0.50,0.75,1.00}/"
echo "Next: score_d4_p3_07.slurm"
