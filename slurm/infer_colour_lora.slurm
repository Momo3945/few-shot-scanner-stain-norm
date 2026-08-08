#!/bin/bash
# =====================================================================
# infer_colour_lora.slurm -- A2 img2img inference + denoising-strength sweep (GPU).
#
# Submit with:  LORA_TAG [STRENGTHS] [LIMIT]
#   sbatch infer_colour_lora.slurm a2h_r8                 # default strengths, 8 frames
#   sbatch infer_colour_lora.slurm a2h_r8 "0.25 0.35 0.45 0.55"   # custom sweep
#   sbatch infer_colour_lora.slurm a2h_r4 "0.3 0.4 0.5" 0         # 0 = ALL held-out frames
# LORA_TAG is the folder name under the lora output root (a2h_r8 / a2h_r4 / h2a_r8).
# =====================================================================

#SBATCH --job-name=infer_lora
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
MITOS_ROOT=/datasets/mhoosen/stain-norm/data/mitos     # NOTE: raw held-out testing frames not yet uploaded here -- see CLAUDE.md
HELDOUT=${DATA_ROOT}/pairs/heldout_frames.csv
LORA_ROOT=${DATA_ROOT}/lora
export HF_HOME=/datasets/mhoosen/hf_cache

LORA_TAG=${1:?"Usage: sbatch infer_colour_lora.slurm LORA_TAG [STRENGTHS] [LIMIT]"}
STRENGTHS=${2:-"0.3 0.4 0.5"}
LIMIT=${3:-8}

# direction inferred from tag prefix (a2h_* -> A2H, h2a_* -> H2A)
case "$LORA_TAG" in
    a2h_*) DIRECTION=A2H ;;
    h2a_*) DIRECTION=H2A ;;
    *) echo "Cannot infer direction from tag '$LORA_TAG' (expect a2h_* or h2a_*)"; exit 1 ;;
esac

LORA_DIR=${LORA_ROOT}/${LORA_TAG}/final
OUT_DIR=${DATA_ROOT}/eval/${LORA_TAG}

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import transformers, peft, accelerate, safetensors, diffusers, cv2" 2>/dev/null; then
    pip install --quiet transformers peft accelerate safetensors diffusers opencv-python-headless
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "LoRA=$LORA_DIR  Direction=$DIRECTION"
echo "Strengths='$STRENGTHS'  Limit=$LIMIT"
echo "Out=$OUT_DIR"
echo "=================================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import torch; print('cuda available:', torch.cuda.is_available())"
# fail fast on a GPU-less node instead of crawling on CPU
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo 'ABORT: no GPU on this node -- resubmit.'; exit 1; }
echo "------------------------------------------------------------------"

cd "${CODE_DIR}"
python src/eval/infer_colour_lora.py \
    --lora      "${LORA_DIR}" \
    --root      "${MITOS_ROOT}" \
    --heldout   "${HELDOUT}" \
    --out       "${OUT_DIR}" \
    --direction "${DIRECTION}" \
    --strengths ${STRENGTHS} \
    --steps 50 \
    --limit "${LIMIT}" \
    --max-crops-per-frame 4 \
    --seed 0

echo "------------------------------------------------------------------"
echo "Inference done. Outputs + eval_manifest.csv in ${OUT_DIR}"
