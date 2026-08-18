#!/bin/bash
# =====================================================================
# normalize_camelyon.slurm -- normalise CAMELYON17 patches with the colour LoRA +
# ControlNet-Canny + LCM-LoRA pipeline (GPU). P2-10: produces the "after"
# (D_post) side of the multi-centre pairwise LAB-Wasserstein measurement --
# score with score_camelyon_wasserstein.slurm afterward, --against pointing at
# the D_pre run's pairwise.csv.
#
# Default config is the P1-09 best general-purpose operating point (1-step
# LCM, strength 0.20): colour LoRA (a2h_r8) + ControlNet-Canny + LCM-LoRA,
# matching infer_a4_lcm.slurm's A4 config and every other normalisation run
# this session (P2-07, P2-08). Applied uniformly to all 5 CAMELYON17 centres --
# the LoRA never saw any of them; that's the generalisation test itself.
#
# No tiling needed (unlike normalize_lizard.slurm) -- CAMELYON17 patches are
# already exactly 512x512 by construction (camelyon_patches.py's extraction).
#
# Submit with:  [LIMIT]
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 normalize_camelyon.slurm 20   # smoke test
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 normalize_camelyon.slurm       # full ~2103 patches
#
# Uses the plain stainnorm env (main diffusion pipeline, NOT tiatoolbox).
# =====================================================================

#SBATCH --job-name=normalize_camelyon
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail

CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_ROOT=/datasets/mhoosen/stain-norm
IMAGES_DIR=${DATA_ROOT}/camelyon17_patches
LORA_DIR=${DATA_ROOT}/lora/a2h_r8/final
CONTROLNET=lllyasviel/sd-controlnet-canny
OUT_DIR=${DATA_ROOT}/eval/camelyon17_normalised
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
python normalize_camelyon.py \
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
echo "Normalisation done. Patches in ${OUT_DIR}"
echo "Next: sbatch score_camelyon_wasserstein.slurm ${OUT_DIR} camelyon17_normalised ${DATA_ROOT}/eval/camelyon17_patches/pairwise.csv"
