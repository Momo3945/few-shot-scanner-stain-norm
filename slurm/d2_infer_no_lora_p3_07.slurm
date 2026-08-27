#!/bin/bash
# =====================================================================
# d2_infer_no_lora_p3_07.slurm -- P3-07 diagnostic D2: run the trained 1024
# ControlNet + source conditioning + frozen SDXL base with the colour LoRA
# DISABLED (infer_colour_translation_sdxl.py's new --no-lora flag), on a
# balanced A06+A08 held-out diagnostic subset (not A06 alone, per the D2
# spec -- A06 is this project's known colour-gap outlier).
#
# Distinguishes:
#   Case A: no-LoRA recovery already ~-5 to -6 -> the frozen SDXL +
#           source-conditioning path itself drifts colour negative; the
#           colour LoRA is too weak to overcome it.
#   Case B: no-LoRA recovery near zero/positive -> the 1024 colour LoRA
#           itself learned the wrong mapping.
#
# Same operating point as the full P3-07 run: strength=0.50, steps=50,
# guidance=2.0, crop=1024, source-mode=correct. On bigbatch (GPU).
# See tickets/PHASE3-TICKETS.md P3-07, diagnostic D2.
#
# Submit with no args:
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 -J d2_no_lora_p3_07 d2_infer_no_lora_p3_07.slurm
# =====================================================================

#SBATCH --job-name=d2_no_lora_p3_07
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail

CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_ROOT=/datasets/mhoosen/stain-norm
MITOS_ROOT=${DATA_ROOT}/mitos_heldout
HELDOUT_FULL=${DATA_ROOT}/pairs/heldout_frames.csv
CHECKPOINT_DIR=${DATA_ROOT}/lora/a2h_cond_r8_sdxl_1024/final
OUT_DIR=${DATA_ROOT}/eval/p3_07_d2_no_lora
export HF_HOME=/datasets/mhoosen/hf_cache

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

# Balanced A06+A08 diagnostic subset -- filtered here (on the compute node,
# inside the job), never on the login node. Header + rows whose first
# column (aperio_slide) is A06 or A08.
HELDOUT_SUBSET=${OUT_DIR}/heldout_frames_a06_a08.csv
awk -F, 'NR==1 || $1=="A06" || $1=="A08"' "${HELDOUT_FULL}" > "${HELDOUT_SUBSET}"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import transformers, peft, accelerate, safetensors, diffusers, cv2" 2>/dev/null; then
    pip install --quiet transformers peft accelerate safetensors diffusers opencv-python-headless
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "P3-07 D2 -- colour LoRA DISABLED, ControlNet=${CHECKPOINT_DIR}/controlnet"
echo "Heldout subset: ${HELDOUT_SUBSET} ($(wc -l < "${HELDOUT_SUBSET}") lines incl. header)"
echo "Out: ${OUT_DIR}"
echo "=================================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo 'ABORT: no GPU on this node -- resubmit.'; exit 1; }
echo "------------------------------------------------------------------"

cd "${CODE_DIR}/src/eval"
if [[ ! -f "${CHECKPOINT_DIR}/pytorch_lora_weights.safetensors" ]]; then
    echo "ABORT: ${CHECKPOINT_DIR}/pytorch_lora_weights.safetensors not found."
    exit 1
fi
python infer_colour_translation_sdxl.py \
    --lora "${CHECKPOINT_DIR}" --controlnet "${CHECKPOINT_DIR}/controlnet" \
    --no-lora --source-mode correct \
    --root "${MITOS_ROOT}" --heldout "${HELDOUT_SUBSET}" --out "${OUT_DIR}" \
    --limit 0 --max-crops-per-frame 4 --steps 50 --strength 0.50 --guidance 2.0 --crop 1024

echo "------------------------------------------------------------------"
echo "Inference done. Outputs + eval_manifest.csv in ${OUT_DIR}"
echo "Next: score_p1_10_ablation.py --eval-dirs ${OUT_DIR} --out eval/p3_07_d2_no_lora_summary"
