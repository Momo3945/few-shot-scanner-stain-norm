#!/bin/bash
# =====================================================================
# d3_infer_condition_split_p3_07.slurm -- P3-07 diagnostic D3: split the
# trained 6-channel source condition [source RGB | source Canny] into
# rgb_only / canny_only / rgb_canny (unmodified) via
# infer_colour_translation_sdxl.py's new --condition-mode flag, on the SAME
# A06+A08 held-out subset D2 used, same operating point (strength=0.50,
# steps=50, guidance=2.0, crop=1024, source-mode=correct), WITH the trained
# colour LoRA enabled (D2 already isolated the LoRA; D3 asks a different
# question -- which half of the conditioning the deployed model actually
# uses to preserve source-scanner colour). Never touches the trained
# weights. On bigbatch (GPU). See tickets/PHASE3-TICKETS.md P3-07, D3.
#
# Submit with:  CONDITION_MODE  (rgb_canny | rgb_only | canny_only)
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44,mscluster61 \
#     -J d3_rgb_canny_p3_07 d3_infer_condition_split_p3_07.slurm rgb_canny
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44,mscluster61 \
#     -J d3_rgb_only_p3_07 d3_infer_condition_split_p3_07.slurm rgb_only
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44,mscluster61 \
#     -J d3_canny_only_p3_07 d3_infer_condition_split_p3_07.slurm canny_only
# =====================================================================

#SBATCH --job-name=d3_p3_07
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail

CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_ROOT=/datasets/mhoosen/stain-norm
MITOS_ROOT=${DATA_ROOT}/mitos_heldout
HELDOUT_FULL=${DATA_ROOT}/pairs/heldout_frames.csv
CHECKPOINT_DIR=${DATA_ROOT}/lora/a2h_cond_r8_sdxl_1024/final
export HF_HOME=/datasets/mhoosen/hf_cache

CONDITION_MODE=${1:?"Usage: sbatch d3_infer_condition_split_p3_07.slurm CONDITION_MODE (rgb_canny|rgb_only|canny_only)"}
case "$CONDITION_MODE" in
    rgb_canny|rgb_only|canny_only) ;;
    *) echo "ABORT: CONDITION_MODE must be rgb_canny, rgb_only, or canny_only (got '${CONDITION_MODE}')."; exit 1 ;;
esac
OUT_DIR=${DATA_ROOT}/eval/p3_07_d3_${CONDITION_MODE}

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

# Balanced A06+A08 diagnostic subset -- same one D2 used (regenerated here,
# on the compute node, if not already present; identical filter so it's
# byte-identical either way).
HELDOUT_SUBSET=${OUT_DIR}/heldout_frames_a06_a08.csv
awk -F, 'NR==1 || $1=="A06" || $1=="A08"' "${HELDOUT_FULL}" > "${HELDOUT_SUBSET}"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import transformers, peft, accelerate, safetensors, diffusers, cv2" 2>/dev/null; then
    pip install --quiet transformers peft accelerate safetensors diffusers opencv-python-headless
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "P3-07 D3 -- condition_mode=${CONDITION_MODE}, ControlNet=${CHECKPOINT_DIR}/controlnet, LoRA enabled"
echo "Heldout subset: ${HELDOUT_SUBSET} ($(wc -l < "${HELDOUT_SUBSET}") lines incl. header)"
echo "Out: ${OUT_DIR}"
echo "=================================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo 'ABORT: no GPU on this node -- resubmit.'; exit 1; }
echo "------------------------------------------------------------------"

cd "${CODE_DIR}/src/eval"
if [[ ! -f "${CHECKPOINT_DIR}/pytorch_lora_weights.safetensors" ]]; then
    echo "ABORT: ${CHECKPOINT_DIR}/pytorch_lora_weights.safetensors not found."
    exit 1
fi
python infer_colour_translation_sdxl.py \
    --lora "${CHECKPOINT_DIR}" --controlnet "${CHECKPOINT_DIR}/controlnet" \
    --source-mode correct --condition-mode "${CONDITION_MODE}" \
    --root "${MITOS_ROOT}" --heldout "${HELDOUT_SUBSET}" --out "${OUT_DIR}" \
    --limit 0 --max-crops-per-frame 4 --steps 50 --strength 0.50 --guidance 2.0 --crop 1024

echo "------------------------------------------------------------------"
echo "Inference done. Outputs + eval_manifest.csv in ${OUT_DIR}"
echo "Next: score_p1_10_ablation.py --eval-dirs ${OUT_DIR} --out eval/p3_07_d3_${CONDITION_MODE}_summary"
