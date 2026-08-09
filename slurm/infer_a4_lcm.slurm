#!/bin/bash
# =====================================================================
# infer_a4_lcm.slurm -- A4 ablation: SD1.5 + ControlNet-Canny + colour LoRA +
# LCM-LoRA, few-step inference (GPU).
#
# Full pipeline (tab:ablation_ladder row A4) -- adds the pretrained LCM-LoRA
# (latent-consistency/lcm-lora-sdv1-5) on top of A3's frozen config (colour LoRA
# rank 8 + ControlNet-Canny), attached via diffusers multi-adapter set_adapters
# (both "colour" and "lcm" active together) with LCMScheduler replacing DDIM.
# The LCM-LoRA is NOT trained -- only its compatibility with the already-frozen
# stain-normalisation adapters is evaluated (sec:training_order, H4/RQ3).
#
# H4/RQ3: few-step LCM here vs the 50-step DDIM reference already produced by A3
# (eval/a3/, same colour-LoRA + ControlNet config) -- compare SSIM/PSNR/artefacts.
# Pre-committed fallback if quality drops (SSIM < 0.85 or visible artefacts):
# 20-step DDIM, i.e. rerun infer_a3_combined.slurm at --steps 20 (no code change).
#
# Default STEPS=8, not the literal "4-step" from the proposal text: at
# num_inference_steps=4, img2img's get_timesteps() (init_timestep =
# int(num_inference_steps * strength)) collapses strengths 0.30 and 0.40 to the
# SAME single real denoising step (confirmed empirically -- identical output on
# every crop, see PHASE1-TICKETS.md P1-05). 8 steps is still within diffusers'
# documented LCM range (4-8) and still far faster than 50-step DDIM, but gives
# 2/3/4 real steps for strengths 0.30/0.40/0.50 -- properly differentiated.
#
# Two GPU nodes (mscluster65, mscluster75) are currently broken ("Unable to
# determine the device handle for GPU0") but show as idle in sinfo -- exclude
# them explicitly until confirmed fixed.
#
# Submit with:  [STRENGTHS] [LIMIT] [STEPS] [OUT_TAG]
#   sbatch --exclude=mscluster65,mscluster75 infer_a4_lcm.slurm                  # default strengths, 8 frames, 8 steps, eval/a4
#   sbatch --exclude=mscluster65,mscluster75 infer_a4_lcm.slurm "0.3 0.4 0.5" 0   # 0 = ALL held-out frames
#   sbatch --exclude=mscluster65,mscluster75 infer_a4_lcm.slurm "0.3 0.4 0.5" 0 4 # literal 4-step (coarser)
#   sbatch --exclude=mscluster65,mscluster75 infer_a4_lcm.slurm "0.6 0.7" 8 8 a4_ext_s67  # exploratory -> eval/a4_ext_s67
# =====================================================================

#SBATCH --job-name=infer_a4
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
CONTROLNET=lllyasviel/sd-controlnet-canny
export HF_HOME=/datasets/mhoosen/hf_cache

STRENGTHS=${1:-"0.3 0.4 0.5"}
LIMIT=${2:-8}
STEPS=${3:-8}
OUT_TAG=${4:-a4}

OUT_DIR=${DATA_ROOT}/eval/${OUT_TAG}

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import transformers, peft, accelerate, safetensors, diffusers, cv2" 2>/dev/null; then
    pip install --quiet transformers peft accelerate safetensors diffusers opencv-python-headless
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "A4 -- LoRA=$LORA_DIR  ControlNet=$CONTROLNET  +LCM-LoRA"
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
