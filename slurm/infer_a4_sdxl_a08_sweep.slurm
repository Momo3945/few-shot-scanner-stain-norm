#!/bin/bash
# =====================================================================
# infer_a4_sdxl_a08_sweep.slurm -- P3-04 follow-up: SDXL A4 strength sweep on
# A08 (a TYPICAL slide, not the A06 colour-gap outlier).
#
# The original sweep (job 44369) accidentally only hit A06 because LIMIT=8
# grabbed the first 8 heldout_frames.csv rows, which are all A06. This
# repeats the exact same sweep (same strengths, steps, guidance, seed) on
# A08 instead, using a filtered heldout CSV (pairs/heldout_frames_a08.csv,
# a NEW file containing only A08's 28 rows -- the canonical
# heldout_frames.csv is untouched).
#
# Purpose: determine whether the A06 sweep's positive-recovery-at-higher-
# strength trend generalises to a typical slide, or whether SDXL's negative
# colour drift at strength 0.20 (the pooled 496-crop finding, Δlab -5.60)
# just gets worse with strength on typical slides -- mirroring SD1.5's own
# known pattern of A06 recovering more at high strength while typical
# slides recover less (see P1-09/RESULTS_SUMMARY.md).
#
# SSIM caveat (already known before running): the A06 sweep showed SSIM
# falling monotonically as strength rises (0.399 -> 0.299, strength 0.20 ->
# 0.70) -- strength 0.20 is likely SDXL's structural ceiling already, and
# it's already below all three classical baselines' pooled SSIM (Macenko
# 0.628, Reinhard 0.681, Histogram Matching 0.651). This sweep is expected
# to close out the open ticket question on colour-recovery generalisation,
# not to find a strength that beats baseline SSIM.
#
# Submit with (no args -- settings fixed to match the original A06 sweep):
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 infer_a4_sdxl_a08_sweep.slurm
# =====================================================================

#SBATCH --job-name=p3_04_sweep_a08
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
HELDOUT=${DATA_ROOT}/pairs/heldout_frames_a08.csv
LORA_DIR=${DATA_ROOT}/lora/a2h_r8_sdxl/final
CONTROLNET=diffusers/controlnet-canny-sdxl-1.0
OUT_DIR=${DATA_ROOT}/eval/a4_sdxl_sweep_a08
export HF_HOME=/datasets/mhoosen/hf_cache

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import transformers, peft, accelerate, safetensors, diffusers, cv2" 2>/dev/null; then
    pip install --quiet transformers peft accelerate safetensors diffusers opencv-python-headless
fi

if [[ ! -f "${LORA_DIR}/pytorch_lora_weights.safetensors" ]]; then
    echo "ABORT: ${LORA_DIR}/pytorch_lora_weights.safetensors not found."
    exit 1
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "A4-SDXL A08 sweep -- LoRA=$LORA_DIR  ControlNet=$CONTROLNET  +LCM-LoRA(sdxl)"
echo "Strengths='0.20 0.30 0.40 0.50 0.70'  Heldout=$HELDOUT (A08 only, 28 frames)  Steps=8"
echo "Out=$OUT_DIR"
echo "=================================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo 'ABORT: no GPU on this node -- resubmit.'; exit 1; }
echo "------------------------------------------------------------------"

cd "${CODE_DIR}"
python src/eval/infer_colour_lora_sdxl.py \
    --lora       "${LORA_DIR}" \
    --root       "${MITOS_ROOT}" \
    --heldout    "${HELDOUT}" \
    --out        "${OUT_DIR}" \
    --controlnet "${CONTROLNET}" \
    --lcm \
    --direction A2H \
    --strengths 0.20 0.30 0.40 0.50 0.70 \
    --steps 8 \
    --guidance 1.5 \
    --limit 0 \
    --max-crops-per-frame 4 \
    --seed 0

echo "------------------------------------------------------------------"
echo "Inference done. Outputs + eval_manifest.csv in ${OUT_DIR}"
echo "Next: sbatch slurm/score_outputs.slurm a4_sdxl_sweep_a08"
