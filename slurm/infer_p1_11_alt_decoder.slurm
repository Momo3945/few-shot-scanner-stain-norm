#!/bin/bash
# =====================================================================
# infer_p1_11_alt_decoder.slurm -- P1-15 Stage 1: decoder-only swap on top
# of P1-11's validated DDIM-inversion translation path (lora/a2h_cond_r8/
# best). Stock encoder + P1-11's exact trajectory (f=1.00, correct source,
# same guidance/steps) held fixed; only the final decode differs -- every
# run decodes the SAME shared final latent through BOTH the stock VAE and
# stabilityai/sd-vae-ft-mse (the exact-latent A/B, built in, not a mode
# flag). See tickets/P1-15_p1_11_alternate_decoder.md and
# tickets/PHASE1-TICKETS.md P1-15.
#
# Submit with:  SOURCE_MODE [LIMIT] [OUT_TAG] [SEEDS]
#   SOURCE_MODE = correct|zero|shuffled, default correct (Stage 1's fixed
#                 setting is correct; shuffled is the ticket's own
#                 Source-Conditioning Sanity Check)
#   LIMIT       = max held-out frames, default 20 (A06+A08 smoke subset
#                 convention, matches infer_p1_10_ddim_inversion.slurm)
#   OUT_TAG     = subdir under eval/p1_15_alt_decoder/, default derived
#                 from SOURCE_MODE
#   SEEDS       = COMMA-separated seed list (e.g. "0,1,2"), default "0".
#                 Comma, not space-separated -- a space-containing arg does
#                 not survive an `ssh host sbatch ... "0 1 2"` round trip
#                 intact (see P1-11 ticket log for why). Converted to
#                 space-separated internally before being passed to --seeds.
#
# Rollout sequence (submit each separately, smallest first):
#   # 1. Compatibility verification + tiny exact-latent A/B sanity check:
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 \
#     infer_p1_11_alt_decoder.slurm correct 1 sanity_check 0
#   # 2. Smoke test, A06+A08 subset:
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 \
#     infer_p1_11_alt_decoder.slurm correct 20 smoke_correct 0,1,2
#   # 3. Source-conditioning sanity check:
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 \
#     infer_p1_11_alt_decoder.slurm shuffled 20 smoke_shuffled 0,1,2
#   # 4. (only if the smoke gate opens) full 496-crop x 3-seed run:
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 --time=04:00:00 \
#     infer_p1_11_alt_decoder.slurm correct 0 full_correct 0,1,2
#
# Then score with the EXISTING scorer (manifest schema is a superset of
# infer_colour_translation.py's -- score_p1_10_ablation.py needs no changes;
# it will see method=stock_decoder / method=alt_decoder rows separately):
#   python src/eval/score_p1_10_ablation.py \
#     --eval-dirs eval/p1_15_alt_decoder/smoke_correct \
#     --out eval/p1_15_alt_decoder/smoke_correct_summary
#
# Requires the P1-10 checkpoint to already exist -- this script never
# trains anything.
# =====================================================================

#SBATCH --job-name=p1_15_alt_decoder
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

SOURCE_MODE=${1:-correct}
LIMIT=${2:-20}
DEFAULT_TAG="smoke_${SOURCE_MODE}"
OUT_TAG=${3:-$DEFAULT_TAG}
# Comma-separated, NOT space-separated -- see infer_p1_10_ddim_inversion.slurm's
# own comment for why (ssh argv-rejoin silently drops seeds after the first).
SEEDS_CSV=${4:-0}
SEEDS=${SEEDS_CSV//,/ }
OUT_DIR=${DATA_ROOT}/eval/p1_15_alt_decoder/${OUT_TAG}

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
echo "P1-15 Stage 1 alt-decoder -- SourceMode=$SOURCE_MODE  Limit=$LIMIT  Seeds=$SEEDS"
echo "Checkpoint=$CHECKPOINT_DIR"
echo "Out=$OUT_DIR"
echo "=================================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo 'ABORT: no GPU on this node -- resubmit.'; exit 1; }
echo "------------------------------------------------------------------"

cd "${CODE_DIR}/src/eval"
python infer_colour_source_ddim_inversion_alt_decoder.py \
    --lora "${CHECKPOINT_DIR}" --controlnet "${CHECKPOINT_DIR}/controlnet" \
    --root "${MITOS_ROOT}" --heldout "${HELDOUT}" --out "${OUT_DIR}" \
    --source-mode "${SOURCE_MODE}" \
    --inversion-fraction 1.0 \
    --limit "${LIMIT}" --max-crops-per-frame 4 \
    --seeds ${SEEDS} \
    --debug-inversion

echo "------------------------------------------------------------------"
echo "Inference done. Outputs + eval_manifest.csv + run_metadata.json in ${OUT_DIR}"
echo "Next: python src/eval/score_p1_10_ablation.py --eval-dirs ${OUT_DIR} --out ${OUT_DIR}_summary"
