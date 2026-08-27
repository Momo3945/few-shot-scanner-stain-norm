#!/bin/bash
# =====================================================================
# infer_p1_13_lcm_controls.slurm -- P1-13's post-training controls
# (source ablation, adapter isolation, generic-vs-task-specific LCM), using
# the selected checkpoint (checkpoint-625, chosen via the 5%-SSIM-floor
# rule). See tickets/PHASE1-TICKETS.md P1-13.
#
# One job per (VARIANT, SOURCE_MODE) combination -- submit each separately:
#
#   # Control 2 -- source ablation (task-specific adapter), 3 jobs:
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 \
#     infer_p1_13_lcm_controls.slurm task_specific correct 20
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 \
#     infer_p1_13_lcm_controls.slurm task_specific zero 20
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 \
#     infer_p1_13_lcm_controls.slurm task_specific shuffled 20
#
#   # Control 3 -- adapter isolation:
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 \
#     infer_p1_13_lcm_controls.slurm adapter_disabled correct 20
#
#   # Control 1 comparison arm -- generic pretrained LCM-LoRA:
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 \
#     infer_p1_13_lcm_controls.slurm generic_lcm correct 20
#
# Then score everything together (existing, unmodified scorer):
#   sbatch score_p1_10_ablation.slurm  # or invoke score_p1_10_ablation.py directly, see below
#
# Guidance-corrected re-run (2026-08-25): set LCM_ADAPTER_DIR/GUIDANCE to
# override the checkpoint/guidance for task_specific, after the guidance
# sweep found guidance_scale=2.0 (Control 1's original setting) mismatched
# w~U[1,2]'s training-time CFG semantics (G=w+1) and checkpoint-500 @
# guidance=1.0 selected as the internally-frozen operating point:
#   LCM_ADAPTER_DIR=/datasets/mhoosen/stain-norm/lora/a2h_cond_r8_lcm_distilled_lr0.00003_ga4/checkpoint-500 \
#   GUIDANCE=1.0 sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 -J p1_13_ctrl_guidance_fix \
#     infer_p1_13_lcm_controls.slurm task_specific correct 20
# =====================================================================

#SBATCH --job-name=p1_13_ctrl
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
MITOS_ROOT=${DATA_ROOT}/mitos_heldout
HELDOUT=${DATA_ROOT}/pairs/heldout_frames.csv
LORA_DIR=${DATA_ROOT}/lora/a2h_cond_r8/best
LCM_ADAPTER=${LCM_ADAPTER_DIR:-${DATA_ROOT}/lora/a2h_cond_r8_lcm_distilled_lr0.00003_ga4/best_by_validation_b}
GUIDANCE_OVERRIDE=${GUIDANCE:-}
export HF_HOME=/datasets/mhoosen/hf_cache

VARIANT=${1:?"Usage: sbatch infer_p1_13_lcm_controls.slurm VARIANT SOURCE_MODE [LIMIT] [OUT_TAG]  (VARIANT = task_specific | adapter_disabled | generic_lcm)"}
SOURCE_MODE=${2:?"SOURCE_MODE required (correct | zero | shuffled)"}
LIMIT=${3:-20}
OUT_TAG=${4:-p1_13_ctrl_${VARIANT}_${SOURCE_MODE}}
OUT_DIR=${DATA_ROOT}/eval/${OUT_TAG}

LCM_FLAG=""
if [[ "${VARIANT}" == "task_specific" || "${VARIANT}" == "adapter_disabled" ]]; then
    if [[ ! -f "${LCM_ADAPTER}/pytorch_lora_weights.safetensors" ]]; then
        echo "ABORT: ${LCM_ADAPTER}/pytorch_lora_weights.safetensors not found."
        exit 1
    fi
    LCM_FLAG="--lcm-adapter ${LCM_ADAPTER}"
fi
GUIDANCE_FLAG=""
if [[ -n "${GUIDANCE_OVERRIDE}" ]]; then
    GUIDANCE_FLAG="--guidance ${GUIDANCE_OVERRIDE}"
fi

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import transformers, peft, accelerate, safetensors, diffusers, cv2" 2>/dev/null; then
    pip install --quiet diffusers==0.39.0 transformers==5.14.1 peft==0.19.1 accelerate==1.14.0 safetensors==0.8.0 opencv-python-headless
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "P1-13 controls -- Variant=$VARIANT  SourceMode=$SOURCE_MODE  Limit=$LIMIT"
echo "LoRA=$LORA_DIR  LcmAdapter=${LCM_ADAPTER:-N/A}"
echo "Out=$OUT_DIR"
echo "=================================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo 'ABORT: no GPU on this node -- resubmit.'; exit 1; }
echo "------------------------------------------------------------------"

cd "${CODE_DIR}/src/eval"
python infer_p1_13_lcm_controls.py \
    --variant "${VARIANT}" --source-mode "${SOURCE_MODE}" \
    --lora "${LORA_DIR}" ${LCM_FLAG} ${GUIDANCE_FLAG} \
    --root "${MITOS_ROOT}" --heldout "${HELDOUT}" --out "${OUT_DIR}" \
    --limit "${LIMIT}" --max-crops-per-frame 4 --seeds 0 1 2

echo "------------------------------------------------------------------"
echo "Inference done. Outputs + eval_manifest.csv in ${OUT_DIR}"
echo "Next (once all control arms are done): python score_p1_10_ablation.py --eval-dirs ${DATA_ROOT}/eval/p1_13_ctrl_* --out ${DATA_ROOT}/eval/p1_13_ctrl_summary"
