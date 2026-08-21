#!/bin/bash
# =====================================================================
# infer_a5_sdxl.slurm -- A5-SDXL: SDXL + ControlNet-Canny + colour LoRA
# + histopathology warm-start LoRA + LCM-LoRA inference (P3-05: transfer of
# the A5 config to SDXL). Separate file from infer_colour_lora_sdxl.slurm
# (the A4-SDXL launcher) rather than an env-var branch on it -- mirrors how
# infer_a5_full.slurm is a separate file from infer_a4_lcm.slurm on SD1.5,
# so the already-validated A4-SDXL launcher is never touched.
#
# Requires BOTH lora/a2h_r8_sdxl/final (train_colour_lora_sdxl.slurm, P3-03)
# AND lora/hist_r32_sdxl/final (train_hist_lora_sdxl.slurm, P3-05) to exist.
#
# Submit with:  [STRENGTHS] [LIMIT] [STEPS] [OUT_TAG]
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 infer_a5_sdxl.slurm                  # default strengths, 8 frames, 8 steps, eval/a5_sdxl
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 infer_a5_sdxl.slurm "0.20" 0          # 0 = ALL held-out frames
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 infer_a5_sdxl.slurm "0.20" 2 8 a5_sdxl_smoke  # smoke test
# =====================================================================

#SBATCH --job-name=infer_a5_sdxl
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
MITOS_ROOT=/datasets/mhoosen/stain-norm/mitos_heldout
HELDOUT=${DATA_ROOT}/pairs/heldout_frames.csv
LORA_DIR=${DATA_ROOT}/lora/a2h_r8_sdxl/final
HIST_LORA_DIR=${DATA_ROOT}/lora/hist_r32_sdxl/final
CONTROLNET=diffusers/controlnet-canny-sdxl-1.0
export HF_HOME=/datasets/mhoosen/hf_cache

STRENGTHS=${1:-"0.20"}
LIMIT=${2:-8}
STEPS=${3:-8}
OUT_TAG=${4:-a5_sdxl}

OUT_DIR=${DATA_ROOT}/eval/${OUT_TAG}

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import transformers, peft, accelerate, safetensors, diffusers, cv2" 2>/dev/null; then
    pip install --quiet transformers peft accelerate safetensors diffusers opencv-python-headless
fi

if [[ ! -f "${LORA_DIR}/pytorch_lora_weights.safetensors" ]]; then
    echo "ABORT: ${LORA_DIR}/pytorch_lora_weights.safetensors not found -- run"
    echo "train_colour_lora_sdxl.slurm first (P3-03 run sequence)."
    exit 1
fi
if [[ ! -f "${HIST_LORA_DIR}/pytorch_lora_weights.safetensors" ]]; then
    echo "ABORT: ${HIST_LORA_DIR}/pytorch_lora_weights.safetensors not found -- run"
    echo "train_hist_lora_sdxl.slurm first (P3-05 run sequence)."
    exit 1
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "A5-SDXL -- LoRA=$LORA_DIR  HistLoRA=$HIST_LORA_DIR  ControlNet=$CONTROLNET  +LCM-LoRA(sdxl)"
echo "Strengths='$STRENGTHS'  Limit=$LIMIT  Steps=$STEPS"
echo "Out=$OUT_DIR"
echo "=================================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import torch; print('cuda available:', torch.cuda.is_available())"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo 'ABORT: no GPU on this node -- resubmit.'; exit 1; }
echo "------------------------------------------------------------------"

cd "${CODE_DIR}"
python src/eval/infer_colour_lora_sdxl.py \
    --lora       "${LORA_DIR}" \
    --hist-lora  "${HIST_LORA_DIR}" \
    --root       "${MITOS_ROOT}" \
    --heldout    "${HELDOUT}" \
    --out        "${OUT_DIR}" \
    --controlnet "${CONTROLNET}" \
    --lcm \
    --direction A2H \
    --strengths ${STRENGTHS} \
    --steps "${STEPS}" \
    --guidance 1.5 \
    --limit "${LIMIT}" \
    --max-crops-per-frame 4 \
    --seed 0

echo "------------------------------------------------------------------"
echo "Inference done. Outputs + eval_manifest.csv in ${OUT_DIR}"
echo "Next: sbatch slurm/score_outputs.slurm ${OUT_TAG}"
