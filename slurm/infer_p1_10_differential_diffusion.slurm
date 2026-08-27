#!/bin/bash
# =====================================================================
# infer_p1_10_differential_diffusion.slurm -- P1-17: Differential Diffusion
# change-map inference for the P1-10 checkpoint (lora/a2h_cond_r8/best). See
# tickets/PHASE1-TICKETS.md P1-17 and
# tickets/P1-17_differential_diffusion_change_map.md.
#
# Submit with:  RADIUS SIGMA CMIN CMAX [LIMIT] [OUT_TAG] [SEEDS]
#   RADIUS/SIGMA/CMIN/CMAX = change-map parameters (see the ticket's Initial
#           Parameter Grid: radius={2,4,8}, sigma={1,2,4}, cmax={0.50,0.70},
#           cmin={0.0,0.15})
#   LIMIT   = max frames, default 20 (matches the P1-10/P1-11 smoke-gate
#             A06+A08 subset convention on held-out runs)
#   OUT_TAG = subdir under eval/p1_17_diffdiff/, default derived from params
#   SEEDS   = COMMA-separated seed list (e.g. "0,1,2"), default "0". Comma,
#             not space-separated -- a space-containing single shell argument
#             does not survive an `ssh host sbatch ... "0 1 2"` round trip
#             intact (hit directly on P1-11, see its ticket log) -- converted
#             to space-separated internally before being passed to --seeds.
#
# Internal-validation parameter search (MANDATORY before the held-out run,
# per the ticket's "Parameter Selection" section -- never tune radius/sigma/
# c-min/c-max on the held-out slides). Set PAIRS_DIR=1 to run against P1-10's
# training-domain pairs instead of the held-out set:
#   PAIRS_DIR=1 sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 \
#     infer_p1_10_differential_diffusion.slurm 4 2 0.0 0.70 0 r4_s2_cmin0.00_cmax0.70 0
#
# Full held-out run at the frozen, selected config:
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 \
#     infer_p1_10_differential_diffusion.slurm 4 2 0.0 0.70 0 held_out 0,1,2
#
# Then score with the EXISTING scorer (manifest schema matches
# infer_colour_translation.py's exactly -- no scorer changes needed):
#   sbatch score_outputs.slurm  # or invoke score_outputs.py directly against ${OUT_DIR}
#
# Requires the P1-10 checkpoint to already exist (train_colour_translation_
# lora.slurm) -- this script never trains anything.
# =====================================================================

#SBATCH --job-name=p1_17_diffdiff
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
TRAIN_PAIRS_DIR=${DATA_ROOT}/pairs/train
CHECKPOINT_DIR=${DATA_ROOT}/lora/a2h_cond_r8/best
export HF_HOME=/datasets/mhoosen/hf_cache

RADIUS=${1:?"Usage: sbatch infer_p1_10_differential_diffusion.slurm RADIUS SIGMA CMIN CMAX [LIMIT] [OUT_TAG] [SEEDS]"}
SIGMA=${2:?"SIGMA required"}
CMIN=${3:?"CMIN required"}
CMAX=${4:?"CMAX required"}
LIMIT=${5:-20}
DEFAULT_TAG="r${RADIUS}_s${SIGMA}_cmin${CMIN}_cmax${CMAX}"
OUT_TAG=${6:-$DEFAULT_TAG}
SEEDS_CSV=${7:-0}
SEEDS=${SEEDS_CSV//,/ }
OUT_DIR=${DATA_ROOT}/eval/p1_17_diffdiff/${OUT_TAG}

DATA_FLAGS="--root ${MITOS_ROOT} --heldout ${HELDOUT}"
if [[ "${PAIRS_DIR:-0}" == "1" ]]; then
    DATA_FLAGS="--pairs-dir ${TRAIN_PAIRS_DIR}"
fi
SAVE_MAP_FLAG=""
if [[ "${SAVE_MAP:-0}" == "1" ]]; then
    SAVE_MAP_FLAG="--save-change-map"
fi

if [[ ! -f "${CHECKPOINT_DIR}/pytorch_lora_weights.safetensors" ]]; then
    echo "ABORT: ${CHECKPOINT_DIR}/pytorch_lora_weights.safetensors not found -- "
    echo "P1-10 must already be trained (train_colour_translation_lora.slurm)."
    exit 1
fi

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import transformers, peft, accelerate, safetensors, diffusers, cv2" 2>/dev/null; then
    pip install --quiet diffusers==0.39.0 transformers==5.14.1 peft==0.19.1 accelerate==1.14.0 safetensors==0.8.0 opencv-python-headless
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "P1-17 Differential Diffusion -- Radius=$RADIUS Sigma=$SIGMA Cmin=$CMIN Cmax=$CMAX Limit=$LIMIT Seeds=$SEEDS"
echo "Mode=$( [[ "${PAIRS_DIR:-0}" == "1" ]] && echo internal-validation || echo held-out )"
echo "Checkpoint=$CHECKPOINT_DIR"
echo "Out=$OUT_DIR"
echo "=================================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo 'ABORT: no GPU on this node -- resubmit.'; exit 1; }
echo "------------------------------------------------------------------"

cd "${CODE_DIR}/src/eval"
python infer_p1_10_differential_diffusion.py \
    --lora "${CHECKPOINT_DIR}" --controlnet "${CHECKPOINT_DIR}/controlnet" \
    ${DATA_FLAGS} --out "${OUT_DIR}" \
    --radius "${RADIUS}" --sigma "${SIGMA}" --c-min "${CMIN}" --c-max "${CMAX}" \
    --limit "${LIMIT}" --max-crops-per-frame 4 \
    --seeds ${SEEDS} ${SAVE_MAP_FLAG}

echo "------------------------------------------------------------------"
echo "Inference done. Outputs + eval_manifest.csv + run_metadata.json in ${OUT_DIR}"
echo "Next: python src/eval/score_outputs.py --eval-dir ${OUT_DIR} --baseline ${DATA_ROOT}/pairs/baseline_metrics/baseline_summary.csv"
