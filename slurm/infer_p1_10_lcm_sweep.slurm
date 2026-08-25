#!/bin/bash
# =====================================================================
# infer_p1_10_lcm_sweep.slurm -- P1-10 + LCM-LoRA: denoising-strength sweep
# for the source-conditioned checkpoint (lora/a2h_cond_r8/best), few-step
# LCM inference via infer_colour_translation_lcm.py. See
# tickets/PHASE1-TICKETS.md (P1-10 LCM follow-up).
#
# Manifest schema matches infer_colour_lora.py's strength-shaped output
# exactly -- scored with the EXISTING, unmodified score_outputs.slurm, not a
# new scorer.
#
# Submit with:  [STRENGTHS_CSV] [LIMIT] [STEPS] [GUIDANCE] [OUT_TAG]
#   STRENGTHS_CSV = COMMA-separated (e.g. "0.20,0.30,0.40,0.50,0.70"), default
#                   matches P1-09's established 8-step boundary points. Comma,
#                   not space-separated: a space-containing argument does not
#                   survive an `ssh host sbatch ... "0.2 0.3"` round trip
#                   intact (see the P1-11 ticket log for the identical bug
#                   hit with --seeds) -- converted to space-separated
#                   internally before being passed to --strengths.
#   LIMIT = max held-out frames, default 20 (A06+A08 smoke subset convention)
#   STEPS = LCM steps, default 8 (not 4 -- see P1-05's quantisation-collision
#           finding, applies identically here)
#   GUIDANCE = default 1.5 (LCM's documented 1.0-2.0 range, matches A4/A5)
#   OUT_TAG = subdir under eval/, default p1_10_lcm_sweep
#
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 \
#     infer_p1_10_lcm_sweep.slurm 0.20,0.30,0.40,0.50,0.70 20 8 1.5 p1_10_lcm_sweep
#
# Then score with the EXISTING scorer (no changes needed):
#   sbatch slurm/score_outputs.slurm p1_10_lcm_sweep
# =====================================================================

#SBATCH --job-name=p1_10_lcm
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail

CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_ROOT=/datasets/mhoosen/stain-norm
MITOS_ROOT=${DATA_ROOT}/mitos_heldout
HELDOUT=${DATA_ROOT}/pairs/heldout_frames.csv
CHECKPOINT_DIR=${DATA_ROOT}/lora/a2h_cond_r8/best
export HF_HOME=/datasets/mhoosen/hf_cache

STRENGTHS_CSV=${1:-"0.20,0.30,0.40,0.50,0.70"}
STRENGTHS=${STRENGTHS_CSV//,/ }
LIMIT=${2:-20}
STEPS=${3:-8}
GUIDANCE=${4:-1.5}
OUT_TAG=${5:-p1_10_lcm_sweep}
OUT_DIR=${DATA_ROOT}/eval/${OUT_TAG}

if [[ ! -f "${CHECKPOINT_DIR}/pytorch_lora_weights.safetensors" ]]; then
    echo "ABORT: ${CHECKPOINT_DIR}/pytorch_lora_weights.safetensors not found -- "
    echo "P1-10 must already be trained (train_colour_translation_lora.slurm)."
    exit 1
fi

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import transformers, peft, accelerate, safetensors, diffusers, cv2" 2>/dev/null; then
    pip install --quiet transformers peft accelerate safetensors diffusers opencv-python-headless
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "P1-10 + LCM -- Strengths='$STRENGTHS'  Limit=$LIMIT  Steps=$STEPS  Guidance=$GUIDANCE"
echo "Checkpoint=$CHECKPOINT_DIR"
echo "Out=$OUT_DIR"
echo "=================================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo 'ABORT: no GPU on this node -- resubmit.'; exit 1; }
echo "------------------------------------------------------------------"

cd "${CODE_DIR}/src/eval"
python infer_colour_translation_lcm.py \
    --lora "${CHECKPOINT_DIR}" --controlnet "${CHECKPOINT_DIR}/controlnet" \
    --root "${MITOS_ROOT}" --heldout "${HELDOUT}" --out "${OUT_DIR}" \
    --strengths ${STRENGTHS} \
    --steps "${STEPS}" --guidance "${GUIDANCE}" \
    --limit "${LIMIT}" --max-crops-per-frame 4 --seeds 0

echo "------------------------------------------------------------------"
echo "Inference done. Outputs + eval_manifest.csv in ${OUT_DIR}"
echo "Next: sbatch slurm/score_outputs.slurm ${OUT_TAG}"
