#!/bin/bash
# =====================================================================
# infer_p1_11_ensemble.slurm -- P1-11 future-work idea #2: genuine
# self-ensembling via eta>0 stochastic DDIM reconstruction. Standalone --
# does NOT touch infer_p1_10_ddim_inversion.slurm or its validated script.
# See tickets/PHASE1-TICKETS.md P1-11 "Future work" #2.
#
# Submit with:  ETA [SOURCE_MODE] [LIMIT] [OUT_TAG] [SEEDS]
#   ETA         = DDIM eta for the RECONSTRUCTION pass, 0.0 = deterministic
#                 (should numerically match the original script's output at
#                 the same settings -- the sanity check below verifies this
#                 before spending real compute on eta>0).
#   SOURCE_MODE = correct|zero|shuffled, default correct
#   LIMIT       = max held-out frames, default 20 (A06+A08 smoke subset)
#   OUT_TAG     = subdir under eval/p1_11_ensemble/, default derived from ETA
#   SEEDS       = COMMA-separated ensemble membership (e.g. "0,1,2" for a
#                 3-member ensemble), default "0,1,2". Comma, not
#                 space-separated -- see infer_p1_10_ddim_inversion.slurm's
#                 own comment for why (ssh argv-rejoin silently drops seeds
#                 after the first).
#
# Rollout sequence:
#   # 1. Sanity check: eta=0 should numerically match the validated original
#   #    script's own output at the same crop/seed/settings:
#   sbatch --exclude=<bad-node list> infer_p1_11_ensemble.slurm 0.0 correct 1 eta0_sanity 0
#   # 2. Real ensemble test, smoke scale, a few eta values:
#   sbatch --exclude=<bad-node list> infer_p1_11_ensemble.slurm 0.3 correct 20 eta03_n3 0,1,2
#   sbatch --exclude=<bad-node list> infer_p1_11_ensemble.slurm 0.5 correct 20 eta05_n3 0,1,2
#   sbatch --exclude=<bad-node list> infer_p1_11_ensemble.slurm 1.0 correct 20 eta10_n3 0,1,2
#
# Then score with the EXISTING scorer (manifest's source_mode column carries
# "<mode>_member" / "<mode>_ensemble" -- score_p1_10_ablation.py groups them
# separately and reports a paired win-rate with zero scorer changes):
#   python src/eval/score_p1_10_ablation.py \
#     --eval-dirs eval/p1_11_ensemble/eta05_n3 \
#     --out eval/p1_11_ensemble/eta05_n3_summary
#
# Requires the P1-10 checkpoint to already exist -- this script never
# trains anything.
# =====================================================================

#SBATCH --job-name=p1_11_ensemble
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail

CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_ROOT=/datasets/mhoosen/stain-norm
MITOS_ROOT=${DATA_ROOT}/mitos_heldout
HELDOUT=${DATA_ROOT}/pairs/heldout_frames.csv
export HF_HOME=/datasets/mhoosen/hf_cache

CHECKPOINT_DIR=${CHECKPOINT_DIR:-${DATA_ROOT}/lora/a2h_cond_r8/best}
DIRECTION=${DIRECTION:-A2H}

ETA=${1:?"Usage: sbatch infer_p1_11_ensemble.slurm ETA [SOURCE_MODE] [LIMIT] [OUT_TAG] [SEEDS]"}
SOURCE_MODE=${2:-correct}
LIMIT=${3:-20}
ETA_TAG=$(echo "$ETA" | tr -d '.')
DEFAULT_TAG="eta${ETA_TAG}_${SOURCE_MODE}"
OUT_TAG=${4:-$DEFAULT_TAG}
SEEDS_CSV=${5:-0,1,2}
SEEDS=${SEEDS_CSV//,/ }
OUT_DIR=${DATA_ROOT}/eval/p1_11_ensemble/${OUT_TAG}

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
echo "P1-11 self-ensembling -- Eta=$ETA  SourceMode=$SOURCE_MODE  Limit=$LIMIT  Seeds=$SEEDS  Direction=$DIRECTION"
echo "Checkpoint=$CHECKPOINT_DIR"
echo "Out=$OUT_DIR"
echo "=================================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo 'ABORT: no GPU on this node -- resubmit.'; exit 1; }
echo "------------------------------------------------------------------"

cd "${CODE_DIR}/src/eval"
python infer_colour_source_ddim_inversion_ensemble.py \
    --lora "${CHECKPOINT_DIR}" --controlnet "${CHECKPOINT_DIR}/controlnet" \
    --root "${MITOS_ROOT}" --heldout "${HELDOUT}" --out "${OUT_DIR}" \
    --source-mode "${SOURCE_MODE}" --direction "${DIRECTION}" --eta "${ETA}" \
    --inversion-fraction 1.0 \
    --limit "${LIMIT}" --max-crops-per-frame 4 \
    --seeds ${SEEDS} \
    --debug-inversion

echo "------------------------------------------------------------------"
echo "Inference done. Outputs + eval_manifest.csv + run_metadata.json in ${OUT_DIR}"
echo "Next: python src/eval/score_p1_10_ablation.py --eval-dirs ${OUT_DIR} --out ${OUT_DIR}_summary"
