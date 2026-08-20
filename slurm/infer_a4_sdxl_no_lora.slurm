#!/bin/bash
# =====================================================================
# infer_a4_sdxl_no_lora.slurm -- P3-04 diagnostic: SDXL base-model-only
# ablation. Same A4-SDXL config as infer_colour_lora_sdxl.slurm (ControlNet-
# Canny + LCM-LoRA, strength 0.20, A2H) but WITHOUT the trained colour LoRA --
# checks whether P3-04's negative colour-recovery finding (SDXL Delta lab
# -5.60 vs SD1.5's +1.53 at the same nominal operating point) comes from the
# frozen SDXL base's own prior, or from the trained adapter doing something
# wrong. infer_colour_lora_sdxl.py already supports omitting --lora (runs the
# frozen base only) -- no new Python code needed, just a launcher that doesn't
# hard-require the LoRA checkpoint the way infer_colour_lora_sdxl.slurm does.
#
# Full 496-crop set (same cost as job 44206's a4_sdxl run, ~1hr) for a
# directly comparable number against the existing eval/a4_sdxl results.
#
# Submit with:
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 infer_a4_sdxl_no_lora.slurm
#
# Uses the plain stainnorm env (SDXL classes confirmed importable, P3-03).
# =====================================================================

#SBATCH --job-name=a4_sdxl_no_lora
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
HELDOUT=${DATA_ROOT}/pairs/heldout_frames.csv
CONTROLNET=diffusers/controlnet-canny-sdxl-1.0
OUT_DIR=${DATA_ROOT}/eval/a4_sdxl_no_lora
export HF_HOME=/datasets/mhoosen/hf_cache

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import transformers, peft, accelerate, safetensors, diffusers, cv2" 2>/dev/null; then
    pip install --quiet transformers peft accelerate safetensors diffusers opencv-python-headless
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "A4-SDXL, NO COLOUR LORA (base-model-only ablation) -- ControlNet=$CONTROLNET  +LCM-LoRA(sdxl)"
echo "Out=$OUT_DIR"
echo "=================================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo 'ABORT: no GPU on this node -- resubmit.'; exit 1; }
echo "------------------------------------------------------------------"

cd "${CODE_DIR}"
python src/eval/infer_colour_lora_sdxl.py \
    --root       "${MITOS_ROOT}" \
    --heldout    "${HELDOUT}" \
    --out        "${OUT_DIR}" \
    --controlnet "${CONTROLNET}" \
    --lcm \
    --direction A2H \
    --strengths 0.20 \
    --steps 8 \
    --guidance 1.5 \
    --limit 0 \
    --max-crops-per-frame 4 \
    --seed 0

echo "------------------------------------------------------------------"
echo "Base-model-only inference done. Outputs in ${OUT_DIR}"
echo "Next: sbatch slurm/score_outputs.slurm a4_sdxl_no_lora"
