#!/bin/bash
# =====================================================================
# infer_a5_full.slurm -- A5 ablation: SD1.5 + hist. warm-start LoRA + colour LoRA
# + ControlNet-Canny + LCM-LoRA, few-step inference (GPU).
#
# Full warm-start pipeline (tab:ablation_ladder row A5, all four adapters active).
# Per sec:training_order the histopathology LoRA (hist_r32, P1-06) is trained and
# frozen INDEPENDENTLY of the colour LoRA -- never jointly -- and stacked alongside
# it at inference via diffusers multi-adapter set_adapters (all three of "colour",
# "hist", "lcm" active together), same mechanism already proven for A4.
#
# Primary comparison: A5 vs A4 (eval/a4/) -- isolates whether the histopathology
# prior contributes beyond ControlNet + colour LoRA + LCM alone (sec:hist_lora).
# A negative result supports that ControlNet + colour LoRA is sufficient without
# the warm-start; this decides whether P3-05 (A5-on-SDXL) is attempted.
#
# Submit with:  [STRENGTHS] [LIMIT] [STEPS] [OUT_TAG]
#   sbatch --exclude=mscluster65,mscluster75 infer_a5_full.slurm                  # default strengths, 8 frames, 8 steps, eval/a5
#   sbatch --exclude=mscluster65,mscluster75 infer_a5_full.slurm "0.3 0.4 0.5" 0   # 0 = ALL held-out frames
#   sbatch --exclude=mscluster65,mscluster75 infer_a5_full.slurm "0.6 0.7" 8 8 a5_ext_s67   # exploratory strengths -> eval/a5_ext_s67 (doesn't clobber eval/a5)
# =====================================================================

#SBATCH --job-name=infer_a5
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
LORA_DIR=${DATA_ROOT}/lora/a2h_r8/final
HIST_LORA_DIR=${DATA_ROOT}/lora/hist_r32/final
CONTROLNET=lllyasviel/sd-controlnet-canny
export HF_HOME=/datasets/mhoosen/hf_cache

STRENGTHS=${1:-"0.3 0.4 0.5"}
LIMIT=${2:-8}
STEPS=${3:-8}
OUT_TAG=${4:-a5}

OUT_DIR=${DATA_ROOT}/eval/${OUT_TAG}

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import transformers, peft, accelerate, safetensors, diffusers, cv2" 2>/dev/null; then
    pip install --quiet transformers peft accelerate safetensors diffusers opencv-python-headless
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "A5 -- LoRA=$LORA_DIR  HistLoRA=$HIST_LORA_DIR  ControlNet=$CONTROLNET  +LCM-LoRA"
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
python src/eval/infer_colour_lora.py \
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
