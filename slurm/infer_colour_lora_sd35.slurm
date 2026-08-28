#!/bin/bash
# =====================================================================
# infer_colour_lora_sd35.slurm -- PR-02: SD3.5 A2H colour-LoRA img2img
# inference on held-out MITOS patches (SD3.5 probe, LoRA-only -- no
# ControlNet/LCM, see tickets/PROBE-SD35-TICKETS.md PR-02/PR-03/PR-04).
#
# Submit with positional args:  [STRENGTHS] [LIMIT] [STEPS] [OUT_TAG]
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 infer_colour_lora_sd35.slurm                       # 0.30, 5 frames, 28 steps, eval/pr02_sd35
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 infer_colour_lora_sd35.slurm "0.30" 1 8 pr02_sd35_smoke   # smoke test
#
# --partition=bigbatch: PR-01's training run proved the same LoRA/transformer
# fits bigbatch's 24GB RTX 3090 (17.77GB peak); inference carries no
# optimizer/gradient state on top, so no biggpu needed here either.
#
# Crops run at 1024x1024 (the Python script's own default), not 512:
# SD3.5's img2img pipeline always outputs 1024x1024 regardless of input size
# (job 47457 confirmed a 512-crop run mismatches the reference and crashes
# score_outputs.py) -- same resolution choice already used for P3-07's
# native-1024 SDXL variant on this same held-out set.
#
# Requires lora/a2h_r8_sd35/final to already exist (train_colour_lora_sd35.slurm,
# PR-01, done 2026-08-27).
#
# Uses the plain stainnorm env (diffusers 0.39.0, StableDiffusion3Img2ImgPipeline
# confirmed importable -- verified 2026-08-28, no new env needed).
# =====================================================================

#SBATCH --job-name=infer_pr02_sd35
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail

CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_ROOT=/datasets/mhoosen/stain-norm
MITOS_ROOT=/datasets/mhoosen/stain-norm/mitos_heldout
HELDOUT=${DATA_ROOT}/pairs/heldout_frames.csv
LORA_DIR=${DATA_ROOT}/lora/a2h_r8_sd35/final
export HF_HOME=/datasets/mhoosen/hf_cache

STRENGTHS=${1:-"0.30"}
LIMIT=${2:-5}
STEPS=${3:-28}
OUT_TAG=${4:-pr02_sd35}

OUT_DIR=${DATA_ROOT}/eval/${OUT_TAG}

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import transformers, peft, accelerate, safetensors, diffusers, cv2" 2>/dev/null; then
    pip install --quiet transformers peft accelerate safetensors diffusers opencv-python-headless
fi

if [[ ! -f "${LORA_DIR}/pytorch_lora_weights.safetensors" ]]; then
    echo "ABORT: ${LORA_DIR}/pytorch_lora_weights.safetensors not found -- run"
    echo "train_colour_lora_sd35.slurm first (PR-01)."
    exit 1
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "PR-02 SD3.5 -- LoRA=$LORA_DIR"
echo "Strengths='$STRENGTHS'  Limit=$LIMIT  Steps=$STEPS"
echo "Out=$OUT_DIR"
echo "=================================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import torch; print('cuda available:', torch.cuda.is_available())"
# fail fast on a GPU-less node instead of crawling on CPU
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo 'ABORT: no GPU on this node -- resubmit.'; exit 1; }
echo "------------------------------------------------------------------"

cd "${CODE_DIR}"
python src/eval/infer_colour_lora_sd35.py \
    --lora       "${LORA_DIR}" \
    --root       "${MITOS_ROOT}" \
    --heldout    "${HELDOUT}" \
    --out        "${OUT_DIR}" \
    --direction  A2H \
    --strengths  ${STRENGTHS} \
    --steps      "${STEPS}" \
    --guidance   2.0 \
    --limit      "${LIMIT}" \
    --max-crops-per-frame 4 \
    --seed 0

echo "------------------------------------------------------------------"
echo "Inference done. Outputs + eval_manifest.csv in ${OUT_DIR}"
echo "Next: sbatch slurm/score_outputs.slurm ${OUT_TAG}"
