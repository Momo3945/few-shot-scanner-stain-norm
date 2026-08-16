#!/bin/bash
# =====================================================================
# normalize_lizard.slurm -- normalise Lizard images with the colour LoRA +
# ControlNet-Canny + LCM-LoRA pipeline (GPU). P2-08 step 5: this produces the
# "B" side of the G/A/B triangle (score against G with lizard_dice.py
# afterward, --against pointing at the "A" run's summary.csv for Relative
# Dice).
#
# Default config is the P1-09 best general-purpose operating point (1-step
# LCM, strength 0.20): colour LoRA (a2h_r8) + ControlNet-Canny + LCM-LoRA,
# matching infer_a4_lcm.slurm's A4 config. Not A5@0.70's "rescue mode" --
# Lizard is a general held-out set, not the A06 hard case.
#
# Submit with:  [LIMIT]
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 normalize_lizard.slurm 5   # smoke test
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 normalize_lizard.slurm      # full 130
#
# Uses the plain stainnorm env (main diffusion pipeline, NOT tiatoolbox --
# unlike infer_hovernet.slurm/score_lizard.slurm, see tickets/PHASE2-TICKETS.md P2-08).
# =====================================================================

#SBATCH --job-name=normalize_lizard
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
IMAGES_DIR=${DATA_ROOT}/lizard_heldout/images
LORA_DIR=${DATA_ROOT}/lora/a2h_r8/final
CONTROLNET=lllyasviel/sd-controlnet-canny
OUT_DIR=${DATA_ROOT}/eval/lizard_normalised_images
export HF_HOME=/datasets/mhoosen/hf_cache

LIMIT=${1:-0}

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import transformers, peft, accelerate, safetensors, diffusers, cv2" 2>/dev/null; then
    pip install --quiet transformers peft accelerate safetensors diffusers opencv-python-headless
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Images=$IMAGES_DIR  LoRA=$LORA_DIR  ControlNet=$CONTROLNET  +LCM-LoRA"
echo "Limit=$LIMIT"
echo "Out=$OUT_DIR"
echo "=================================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import torch; print('cuda available:', torch.cuda.is_available())"
# fail fast on a GPU-less node instead of crawling on CPU
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo 'ABORT: no GPU on this node -- resubmit.'; exit 1; }
echo "------------------------------------------------------------------"

cd "${CODE_DIR}/src/eval"
python normalize_lizard.py \
    --images-dir  "${IMAGES_DIR}" \
    --out         "${OUT_DIR}" \
    --lora        "${LORA_DIR}" \
    --controlnet  "${CONTROLNET}" \
    --lcm \
    --strength 0.20 \
    --steps 8 \
    --guidance 1.5 \
    --seed 0 \
    --limit "${LIMIT}"

echo "------------------------------------------------------------------"
echo "Normalisation done. Images in ${OUT_DIR}"
echo "Next: sbatch infer_hovernet.slurm lizard_normalised ${OUT_DIR} \"\" \"\""
