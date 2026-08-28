#!/bin/bash
# =====================================================================
# infer_a1_controlnet.slurm -- A1 ablation: SD1.5 + ControlNet-Canny, no LoRA (GPU).
#
# Structural-conditioning-only ablation condition (tab:ablation_ladder row A1) --
# no colour LoRA, no LCM. Canny edge maps (canny.py) are the "prototype signal"
# ControlNet conditions on (proposal's ControlNet section) -- the same GaussianBlur
# + Otsu-adaptive thresholds that produced fig:canny1.
#
# Submit with:  [STRENGTHS] [LIMIT]
#   sbatch infer_a1_controlnet.slurm                        # default strengths, 8 frames
#   sbatch infer_a1_controlnet.slurm "0.3 0.4 0.5" 0         # 0 = ALL held-out frames
# =====================================================================

#SBATCH --job-name=infer_a1
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
MITOS_ROOT=/datasets/mhoosen/stain-norm/mitos_heldout
HELDOUT=${DATA_ROOT}/pairs/heldout_frames.csv
CONTROLNET=lllyasviel/sd-controlnet-canny
export HF_HOME=/datasets/mhoosen/hf_cache

STRENGTHS=${1:-"0.3 0.4 0.5"}
LIMIT=${2:-8}

# P2-12: optional overrides, unset -> identical to every prior invocation.
DIRECTION=${DIRECTION:-A2H}
OUT_TAG=${OUT_TAG:-a1}
OUT_DIR=${DATA_ROOT}/eval/${OUT_TAG}

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import transformers, peft, accelerate, safetensors, diffusers, cv2" 2>/dev/null; then
    pip install --quiet transformers peft accelerate safetensors diffusers opencv-python-headless
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "A1 baseline -- ControlNet=$CONTROLNET, no LoRA, no LCM"
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
    --root       "${MITOS_ROOT}" \
    --heldout    "${HELDOUT}" \
    --out        "${OUT_DIR}" \
    --controlnet "${CONTROLNET}" \
    --direction "${DIRECTION}" \
    --strengths ${STRENGTHS} \
    --steps 50 \
    --limit "${LIMIT}" \
    --max-crops-per-frame 4 \
    --seed 0

echo "------------------------------------------------------------------"
echo "Inference done. Outputs + eval_manifest.csv in ${OUT_DIR}"
echo "Next: sbatch slurm/score_outputs.slurm a1"
