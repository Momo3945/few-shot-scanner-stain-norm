#!/bin/bash
# =====================================================================
# infer_colour_translation_sdxl.slurm -- P3-06: inference/ablation for the
# SDXL transfer of P1-10's source-conditioned colour LoRA + trained
# ControlNet (bigbatch). See tickets/PHASE3-TICKETS.md P3-06.
#
# Submit with:  MODE CHECKPOINT_DIR [LIMIT] [OUT_TAG]
#   MODE = correct | zero | shuffled | vae_only
#   CHECKPOINT_DIR = e.g. /datasets/mhoosen/stain-norm/lora/a2h_cond_r8_sdxl/final
#                    (ignored for vae_only)
#
#   # mandatory source-conditioning ablation on the overfit-test checkpoint,
#   # evaluated against its OWN training pairs (set PAIRS_DIR, not held-out):
#   PAIRS_DIR=/datasets/mhoosen/stain-norm/pairs/train OVERFIT_N=8 \
#     sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 infer_colour_translation_sdxl.slurm correct  /datasets/mhoosen/stain-norm/lora/a2h_cond_r8_sdxl_overfit8/final 0 p3_06_ablation_correct
#   PAIRS_DIR=/datasets/mhoosen/stain-norm/pairs/train OVERFIT_N=8 \
#     sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 infer_colour_translation_sdxl.slurm zero      /datasets/mhoosen/stain-norm/lora/a2h_cond_r8_sdxl_overfit8/final 0 p3_06_ablation_zero
#   PAIRS_DIR=/datasets/mhoosen/stain-norm/pairs/train OVERFIT_N=8 \
#     sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 infer_colour_translation_sdxl.slurm shuffled  /datasets/mhoosen/stain-norm/lora/a2h_cond_r8_sdxl_overfit8/final 0 p3_06_ablation_shuffled
#   # then: python src/eval/score_p1_10_ablation.py --eval-dirs eval/p3_06_ablation_{correct,zero,shuffled} --out eval/p3_06_ablation_summary
#
#   # Ablation against HELD-OUT frames instead (real, non-overfit checkpoint):
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 infer_colour_translation_sdxl.slurm correct /datasets/mhoosen/stain-norm/lora/a2h_cond_r8_sdxl/final 4 p3_06_ablation_correct
#
#   # VAE-only floor (no checkpoint needed -- SDXL's own VAE floor, NOT
#   # SD1.5's; expect a different number, do not compare directly):
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 infer_colour_translation_sdxl.slurm vae_only "" 0 p3_06_vae_floor
#
# Requires the LoRA/ControlNet checkpoint to already exist
# (train_colour_translation_lora_sdxl.slurm) unless MODE=vae_only.
#
# Scoring: NOT score_outputs.slurm -- this script's manifest is seed/
# source_mode-shaped, not strength-shaped. Use score_p1_10_ablation.py instead
# (see the ablation example above) -- confirmed fully generic over
# resolution/backbone, zero changes needed (same as the SD1.5 P1-10 version).
# =====================================================================

#SBATCH --job-name=infer_p3_06
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
MITOS_ROOT=/datasets/mhoosen/stain-norm/mitos_heldout
HELDOUT=${DATA_ROOT}/pairs/heldout_frames.csv
export HF_HOME=/datasets/mhoosen/hf_cache

MODE=${1:?"Usage: sbatch infer_colour_translation_sdxl.slurm MODE CHECKPOINT_DIR [LIMIT] [OUT_TAG]  (MODE = correct|zero|shuffled|vae_only)"}
CHECKPOINT_DIR=${2:-}
LIMIT=${3:-4}
OUT_TAG=${4:-p3_06_${MODE}}
OUT_DIR=${DATA_ROOT}/eval/${OUT_TAG}

# PAIRS_DIR/OVERFIT_N (env vars, not positional -- diagnostic mode, mirrors
# train_colour_translation_lora_sdxl.slurm's OVERFIT_N convention).
PAIRS_ARGS=""
if [[ -n "${PAIRS_DIR:-}" ]]; then
    PAIRS_ARGS="--pairs-dir ${PAIRS_DIR}"
    if [[ -n "${OVERFIT_N:-}" ]]; then
        PAIRS_ARGS="${PAIRS_ARGS} --overfit-n ${OVERFIT_N}"
    fi
fi

# CROP (env var, not positional): held-out grid crop size, e.g. P3-07's 1024
# checkpoints need --crop 1024 to match training resolution (default 512
# unchanged for P3-06). Ignored in PAIRS_ARGS mode (native image size used).
CROP_ARGS=""
if [[ -n "${CROP:-}" ]]; then
    CROP_ARGS="--crop ${CROP}"
fi

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import transformers, peft, accelerate, safetensors, diffusers, cv2" 2>/dev/null; then
    pip install --quiet transformers peft accelerate safetensors diffusers opencv-python-headless
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "P3-06 (P1-10 on SDXL) -- Mode=$MODE  Checkpoint=$CHECKPOINT_DIR  Limit=$LIMIT  PairsArgs='${PAIRS_ARGS}'"
echo "Out=$OUT_DIR"
echo "=================================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo 'ABORT: no GPU on this node -- resubmit.'; exit 1; }
echo "------------------------------------------------------------------"

cd "${CODE_DIR}/src/eval"
if [[ "$MODE" == "vae_only" ]]; then
    if [[ -n "$PAIRS_ARGS" ]]; then
        python infer_colour_translation_sdxl.py --vae-only ${PAIRS_ARGS} --out "${OUT_DIR}" --limit "${LIMIT}"
    else
        python infer_colour_translation_sdxl.py \
            --vae-only \
            --root "${MITOS_ROOT}" --heldout "${HELDOUT}" --out "${OUT_DIR}" \
            --limit "${LIMIT}" --max-crops-per-frame 4 ${CROP_ARGS}
    fi
else
    if [[ -z "$CHECKPOINT_DIR" || ! -f "${CHECKPOINT_DIR}/pytorch_lora_weights.safetensors" ]]; then
        echo "ABORT: ${CHECKPOINT_DIR}/pytorch_lora_weights.safetensors not found -- "
        echo "run train_colour_translation_lora_sdxl.slurm first."
        exit 1
    fi
    if [[ -n "$PAIRS_ARGS" ]]; then
        python infer_colour_translation_sdxl.py \
            --lora "${CHECKPOINT_DIR}" --controlnet "${CHECKPOINT_DIR}/controlnet" \
            --source-mode "${MODE}" ${PAIRS_ARGS} --out "${OUT_DIR}" \
            --limit "${LIMIT}" --steps 50
    else
        python infer_colour_translation_sdxl.py \
            --lora "${CHECKPOINT_DIR}" --controlnet "${CHECKPOINT_DIR}/controlnet" \
            --source-mode "${MODE}" \
            --root "${MITOS_ROOT}" --heldout "${HELDOUT}" --out "${OUT_DIR}" \
            --limit "${LIMIT}" --max-crops-per-frame 4 --steps 50 ${CROP_ARGS}
    fi
fi

echo "------------------------------------------------------------------"
echo "Inference done. Outputs + eval_manifest.csv in ${OUT_DIR}"
echo "Next: score_p1_10_ablation.py (NOT score_outputs.slurm -- manifest schema differs, see header)"
