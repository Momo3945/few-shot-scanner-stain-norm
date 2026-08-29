#!/bin/bash
# =====================================================================
# infer_colour_translation.slurm -- P1-10: inference/ablation for the
# source-conditioned colour LoRA + trained ControlNet (bigbatch).
#
# Submit with:  MODE CHECKPOINT_DIR [LIMIT] [OUT_TAG]
#   MODE = correct | zero | shuffled | vae_only
#   CHECKPOINT_DIR = e.g. /datasets/mhoosen/stain-norm/lora/a2h_cond_r8/final
#                    (ignored for vae_only)
#
#   # mandatory source-conditioning ablation on the overfit-test checkpoint,
#   # evaluated against its OWN training pairs (set PAIRS_DIR, not held-out --
#   # an overfit-8-pair checkpoint has no business being judged on unseen
#   # frames; --exclude bad-node list per CLAUDE.md):
#   PAIRS_DIR=/datasets/mhoosen/stain-norm/pairs/train OVERFIT_N=8 \
#     sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 infer_colour_translation.slurm correct  /datasets/mhoosen/stain-norm/lora/a2h_cond_r8_overfit8/final 0 p1_10_ablation_correct
#   PAIRS_DIR=/datasets/mhoosen/stain-norm/pairs/train OVERFIT_N=8 \
#     sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 infer_colour_translation.slurm zero      /datasets/mhoosen/stain-norm/lora/a2h_cond_r8_overfit8/final 0 p1_10_ablation_zero
#   PAIRS_DIR=/datasets/mhoosen/stain-norm/pairs/train OVERFIT_N=8 \
#     sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 infer_colour_translation.slurm shuffled  /datasets/mhoosen/stain-norm/lora/a2h_cond_r8_overfit8/final 0 p1_10_ablation_shuffled
#   # then: python src/eval/score_p1_10_ablation.py --eval-dirs eval/p1_10_ablation_{correct,zero,shuffled} --out eval/p1_10_ablation_summary
#
#   # Ablation against HELD-OUT frames instead (for a real, non-overfit checkpoint;
#   # PAIRS_DIR unset -> default heldout path):
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 infer_colour_translation.slurm correct /datasets/mhoosen/stain-norm/lora/a2h_cond_r8/final 4 p1_10_ablation_correct
#
#   # VAE-only floor (no checkpoint needed):
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 infer_colour_translation.slurm vae_only "" 0 p1_10_vae_floor
#
#   # Few-step DDIM vs LCM diagnostic (P1-12 follow-up): same checkpoint,
#   # matched low-step/strength/guidance to a p1_10_lcm_sweep run, plain DDIM
#   # scheduler (this script never touches LCMScheduler) -- isolates whether
#   # quality differences come from the scheduler itself vs just step count.
#   # STEPS/STRENGTH/GUIDANCE are env-var overrides (default 50/0.50/2.0,
#   # unchanged from every prior invocation of this script):
#   STEPS=8 STRENGTH=0.80 GUIDANCE=2.0 \
#     sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 infer_colour_translation.slurm correct /datasets/mhoosen/stain-norm/lora/a2h_cond_r8/best 20 p1_10_ddim_vs_lcm_diag
#
# Requires the LoRA/ControlNet checkpoint to already exist
# (train_colour_translation_lora.slurm) unless MODE=vae_only.
#
# Scoring: NOT score_outputs.slurm -- this script's manifest is seed/
# source_mode-shaped, not strength-shaped, and score_outputs.py would KeyError
# on it. Use score_p1_10_ablation.py instead (see the ablation example above).
# =====================================================================

#SBATCH --job-name=infer_p1_10
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
export HF_HOME=/datasets/mhoosen/hf_cache

MODE=${1:?"Usage: sbatch infer_colour_translation.slurm MODE CHECKPOINT_DIR [LIMIT] [OUT_TAG]  (MODE = correct|zero|shuffled|vae_only)"}
CHECKPOINT_DIR=${2:-}
LIMIT=${3:-4}
OUT_TAG=${4:-p1_10_${MODE}}
OUT_DIR=${DATA_ROOT}/eval/${OUT_TAG}
STEPS=${STEPS:-50}
STRENGTH=${STRENGTH:-0.50}
GUIDANCE=${GUIDANCE:-2.0}
# P2-12: this script never passed --direction at all, always defaulting to
# A2H regardless of which checkpoint was loaded. Additive override, unset ->
# identical to every prior invocation.
DIRECTION=${DIRECTION:-A2H}

# PAIRS_DIR/OVERFIT_N (env vars, not positional -- diagnostic mode, mirrors
# train_colour_translation_lora.slurm's OVERFIT_N convention): evaluate a
# checkpoint's own (over)fit training pairs instead of held-out frames.
PAIRS_ARGS=""
if [[ -n "${PAIRS_DIR:-}" ]]; then
    PAIRS_ARGS="--pairs-dir ${PAIRS_DIR}"
    if [[ -n "${OVERFIT_N:-}" ]]; then
        PAIRS_ARGS="${PAIRS_ARGS} --overfit-n ${OVERFIT_N}"
    fi
fi

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import transformers, peft, accelerate, safetensors, diffusers, cv2" 2>/dev/null; then
    pip install --quiet transformers peft accelerate safetensors diffusers opencv-python-headless
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "P1-10 -- Mode=$MODE  Checkpoint=$CHECKPOINT_DIR  Limit=$LIMIT  PairsArgs='${PAIRS_ARGS}'"
echo "Steps=$STEPS  Strength=$STRENGTH  Guidance=$GUIDANCE  Direction=$DIRECTION"
echo "Out=$OUT_DIR"
echo "=================================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo 'ABORT: no GPU on this node -- resubmit.'; exit 1; }
echo "------------------------------------------------------------------"

cd "${CODE_DIR}/src/eval"
if [[ "$MODE" == "vae_only" ]]; then
    if [[ -n "$PAIRS_ARGS" ]]; then
        python infer_colour_translation.py --vae-only ${PAIRS_ARGS} --out "${OUT_DIR}" --limit "${LIMIT}"
    else
        python infer_colour_translation.py \
            --vae-only \
            --root "${MITOS_ROOT}" --heldout "${HELDOUT}" --out "${OUT_DIR}" \
            --limit "${LIMIT}" --max-crops-per-frame 4
    fi
else
    if [[ -z "$CHECKPOINT_DIR" || ! -f "${CHECKPOINT_DIR}/pytorch_lora_weights.safetensors" ]]; then
        echo "ABORT: ${CHECKPOINT_DIR}/pytorch_lora_weights.safetensors not found -- "
        echo "run train_colour_translation_lora.slurm first."
        exit 1
    fi
    if [[ -n "$PAIRS_ARGS" ]]; then
        python infer_colour_translation.py \
            --lora "${CHECKPOINT_DIR}" --controlnet "${CHECKPOINT_DIR}/controlnet" \
            --source-mode "${MODE}" --direction "${DIRECTION}" ${PAIRS_ARGS} --out "${OUT_DIR}" \
            --limit "${LIMIT}" --steps "${STEPS}" --strength "${STRENGTH}" --guidance "${GUIDANCE}"
    else
        python infer_colour_translation.py \
            --lora "${CHECKPOINT_DIR}" --controlnet "${CHECKPOINT_DIR}/controlnet" \
            --source-mode "${MODE}" --direction "${DIRECTION}" \
            --root "${MITOS_ROOT}" --heldout "${HELDOUT}" --out "${OUT_DIR}" \
            --limit "${LIMIT}" --max-crops-per-frame 4 --steps "${STEPS}" --strength "${STRENGTH}" --guidance "${GUIDANCE}"
    fi
fi

echo "------------------------------------------------------------------"
echo "Inference done. Outputs + eval_manifest.csv in ${OUT_DIR}"
echo "Next: score_p1_10_ablation.py (NOT score_outputs.slurm -- manifest schema differs, see header)"
