#!/bin/bash
# =====================================================================
# infer_p1_10_target_only_baseline.slurm -- P1-10 smoke gate: the ORIGINAL
# target-only colour LoRA (lora/a2h_r8/final, no source conditioning at all)
# run at settings matched to infer_colour_translation.py's own defaults
# (strength 0.50, 50-step DDIM, guidance 2.0), on the SAME A06+A08 subset
# used for the P1-10 correct/shuffled smoke runs (LIMIT=20).
#
# This is the "target-only LoRA" arm of the ticket's go/no-go smoke gate:
#   "Continue to the full held-out run only if the correct source condition
#   beats both the target-only LoRA and its own shuffled-source ablation on
#   windowed colour recovery without reducing structural fidelity."
#
# IMPORTANT: does NOT reuse infer_colour_lora.slurm as-is -- that launcher's
# default OUT_DIR is eval/a2h_r8/, which is the already-committed official
# P2-04 evaluation output. Writing smoke-gate results there would silently
# overwrite it. This script writes to a separate, disposable path instead.
#
# Scoring: score_outputs.py (this manifest is strength-shaped, matching its
# native schema -- NOT score_p1_10_ablation.py, which expects source_mode/seed).
#
# Submit with (no args -- everything fixed to match the smoke gate spec):
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 infer_p1_10_target_only_baseline.slurm
# =====================================================================

#SBATCH --job-name=p1_10_smoke_target_only
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail

CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_ROOT=/datasets/mhoosen/stain-norm
MITOS_ROOT=${DATA_ROOT}/mitos_heldout
HELDOUT=${DATA_ROOT}/pairs/heldout_frames.csv
LORA_DIR=${DATA_ROOT}/lora/a2h_r8/final
OUT_DIR=${DATA_ROOT}/eval/p1_10_smoke_target_only
export HF_HOME=/datasets/mhoosen/hf_cache

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import transformers, peft, accelerate, safetensors, diffusers, cv2" 2>/dev/null; then
    pip install --quiet transformers peft accelerate safetensors diffusers opencv-python-headless
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "P1-10 smoke gate -- target-only LoRA baseline (lora/a2h_r8/final)"
echo "strength=0.50 steps=50 guidance=2.0 limit=20 (A06+A08 mix)"
echo "Out=$OUT_DIR"
echo "=================================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo 'ABORT: no GPU on this node -- resubmit.'; exit 1; }
if [[ ! -f "${LORA_DIR}/pytorch_lora_weights.safetensors" ]]; then
    echo "ABORT: ${LORA_DIR}/pytorch_lora_weights.safetensors not found."
    exit 1
fi
echo "------------------------------------------------------------------"

cd "${CODE_DIR}/src/eval"
python infer_colour_lora.py \
    --lora "${LORA_DIR}" \
    --root "${MITOS_ROOT}" --heldout "${HELDOUT}" --out "${OUT_DIR}" \
    --direction A2H \
    --strengths 0.50 \
    --steps 50 \
    --guidance 2.0 \
    --limit 20 \
    --max-crops-per-frame 4 \
    --seed 0

echo "------------------------------------------------------------------"
echo "Done. Outputs + eval_manifest.csv in ${OUT_DIR}"
echo "Next: score_outputs.py --eval-dir ${OUT_DIR}"
