#!/bin/bash
# =====================================================================
# infer_p1_10_ddim_inversion.slurm -- P1-11: DDIM-inversion inference path
# for the P1-10 checkpoint (lora/a2h_cond_r8/best). See
# tickets/PHASE1-TICKETS.md P1-11.
#
# Submit with:  MODE [INVERSION_FRACTION] [SOURCE_MODE] [LIMIT] [OUT_TAG] [SEEDS]
#   MODE = identity | translate
#   INVERSION_FRACTION = float in (0,1], default 1.0
#   SOURCE_MODE = correct|zero|shuffled (translate only; ignored for identity), default correct
#   LIMIT = max held-out frames, default 20 (matches the P1-10 smoke-gate
#           A06+A08 subset convention -- LIMIT=20 on heldout_frames.csv)
#   OUT_TAG = subdir under eval/p1_10_ddim_inversion/, default derived from MODE+fraction
#   SEEDS = COMMA-separated seed list (e.g. "0,1,2"), default "0". Comma, not
#           space-separated: a space-containing argument does not survive an
#           `ssh host sbatch ... "0 1 2"` round trip intact (ssh rejoins its
#           argv with spaces before the remote shell reparses it, silently
#           splitting "0 1 2" into three separate words and dropping all but
#           the first positionally) -- hit directly, not hypothetical, see
#           the P1-11 ticket log. Converted to space-separated internally
#           before being passed to --seeds.
#
# Task-file-mandated smoke test sequence (task file sec. 23-24) -- run these
# BEFORE the 496-crop full held-out job, per the ticket's own gate. Submit
# each of these directly (no `cd &&` wrapping -- that form reliably triggers
# Claude Code's auto-mode classifier instead of a normal approval prompt):
#   # identity, full inversion:
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 \
#     infer_p1_10_ddim_inversion.slurm identity 1.0 correct 20 identity_inv100 0
#   # identity, partial inversion sweep:
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 \
#     infer_p1_10_ddim_inversion.slurm identity 0.25 correct 20 identity_inv025 0
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 \
#     infer_p1_10_ddim_inversion.slurm identity 0.50 correct 20 identity_inv050 0
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 \
#     infer_p1_10_ddim_inversion.slurm identity 0.75 correct 20 identity_inv075 0
#   # translate, at whichever inversion-fraction looked best above, all three
#   # source-mode ablation controls:
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 \
#     infer_p1_10_ddim_inversion.slurm translate 0.50 correct  20 translate_inv050_correct  0,1,2
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 \
#     infer_p1_10_ddim_inversion.slurm translate 0.50 shuffled 20 translate_inv050_shuffled 0,1,2
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 \
#     infer_p1_10_ddim_inversion.slurm translate 0.50 zero     20 translate_inv050_zero     0,1,2
#
# Then score with the EXISTING scorer (manifest schema is a superset of
# infer_colour_translation.py's, score_p1_10_ablation.py needs no changes):
#   python src/eval/score_p1_10_ablation.py \
#     --eval-dirs eval/p1_10_ddim_inversion/identity_inv025 \
#                 eval/p1_10_ddim_inversion/identity_inv050 \
#                 eval/p1_10_ddim_inversion/identity_inv075 \
#                 eval/p1_10_ddim_inversion/identity_inv100 \
#     --out eval/p1_10_ddim_inversion/identity_summary
#
# Only after the smoke test is validated (task file sec. "E") should LIMIT=0
# (full 496-crop) be submitted, on a fresh OUT_TAG.
#
# Requires the P1-10 checkpoint to already exist (train_colour_translation_
# lora.slurm) -- this script never trains anything.
# =====================================================================

#SBATCH --job-name=p1_10_ddim_inv
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

MODE=${1:?"Usage: sbatch infer_p1_10_ddim_inversion.slurm MODE [INVERSION_FRACTION] [SOURCE_MODE] [LIMIT] [OUT_TAG] [SEEDS]  (MODE = identity|translate)"}
FRACTION=${2:-1.0}
SOURCE_MODE=${3:-correct}
LIMIT=${4:-20}
FRACTION_TAG=$(echo "$FRACTION" | tr -d '.')
DEFAULT_TAG="${MODE}_inv${FRACTION_TAG}"
if [[ "$MODE" == "translate" ]]; then DEFAULT_TAG="${DEFAULT_TAG}_${SOURCE_MODE}"; fi
OUT_TAG=${5:-$DEFAULT_TAG}
# Comma-separated, NOT space-separated: a space-containing single shell
# argument (e.g. "0 1 2") does not survive ssh's argv-rejoin-then-remote-
# reparse round trip intact -- it gets silently split into separate words
# and only the first one reaches $6, shifting nothing else but silently
# dropping seeds 1/2 (hit directly, not hypothetical -- see P1-11 ticket
# log). Comma-separated has no internal whitespace, so it can't be split.
SEEDS_CSV=${6:-0}
SEEDS=${SEEDS_CSV//,/ }
OUT_DIR=${DATA_ROOT}/eval/p1_10_ddim_inversion/${OUT_TAG}

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
python -c "from diffusers import DDIMInverseScheduler" \
    || { echo "ABORT: DDIMInverseScheduler not available in this diffusers install."; exit 1; }

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "P1-11 DDIM inversion -- Mode=$MODE  Fraction=$FRACTION  SourceMode=$SOURCE_MODE  Limit=$LIMIT  Seeds=$SEEDS"
echo "Checkpoint=$CHECKPOINT_DIR"
echo "Out=$OUT_DIR"
echo "=================================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo 'ABORT: no GPU on this node -- resubmit.'; exit 1; }
echo "------------------------------------------------------------------"

cd "${CODE_DIR}/src/eval"
python infer_colour_source_ddim_inversion.py \
    --lora "${CHECKPOINT_DIR}" --controlnet "${CHECKPOINT_DIR}/controlnet" \
    --root "${MITOS_ROOT}" --heldout "${HELDOUT}" --out "${OUT_DIR}" \
    --mode "${MODE}" --source-mode "${SOURCE_MODE}" \
    --inversion-fraction "${FRACTION}" \
    --limit "${LIMIT}" --max-crops-per-frame 4 \
    --seeds ${SEEDS} \
    --debug-inversion

echo "------------------------------------------------------------------"
echo "Inference done. Outputs + eval_manifest.csv + run_metadata.json in ${OUT_DIR}"
echo "Next: python src/eval/score_p1_10_ablation.py --eval-dirs ${OUT_DIR} --out ${OUT_DIR}_summary"
