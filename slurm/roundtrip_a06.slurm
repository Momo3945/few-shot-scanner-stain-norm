#!/bin/bash
# =====================================================================
# roundtrip_a06.slurm -- P2-07 cycle consistency: round-trip reconstruction
# A06 -> LoRA(A->H) -> Hhat06 -> LoRA(H->A) -> Ahat06 <-> original A06 (GPU).
#
# Proposal-explicit (sec:experiments "Round Trip Reconstruction"): deterministic
# 50-step DDIM, trained colour LoRA weights only -- NOT LCM (stochastic drift on
# the return pass would contaminate the structural-deviation measurement; LCM is
# reserved strictly for the unidirectional deployment pipeline) and, per the
# proposal's own equation, no ControlNet by default (roundtrip_a06.py's
# --controlnet is available as an opt-in follow-up against the full deployed A3
# stack, not the literal test). Metrics: grayscale SSIM, PSNR, MAE (+ bonus LAB/
# windowed-LAB/CIEDE2000 for free via metrics.py's score_aligned_pair()).
#
# Default strength 0.20 (P1-09's best general-purpose point, and the point this
# experiment's own axis -- structural preservation -- most rewards), not a full
# sweep -- keeps the default run's compute volume sane (16 A06 frames x up to 4
# crops x 2 full 50-step DDIM passes already, before multiplying by strengths).
# Pass extra --strengths values for a supplementary sweep if wanted later.
#
# Submit with:  [STRENGTHS] [LIMIT] [STEPS]
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 roundtrip_a06.slurm 0.2 2 20   # smoke test: 2 frames, 20 steps
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 roundtrip_a06.slurm             # full A06 (16 frames), strength 0.20, 50-step DDIM
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 roundtrip_a06.slurm "0.2 0.3 0.4 0.5" 0 50  # supplementary strength sweep
#
# Uses the plain stainnorm env (main diffusion pipeline).
# =====================================================================

#SBATCH --job-name=roundtrip_a06
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail

CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_ROOT=/datasets/mhoosen/stain-norm
MITOS_ROOT=/datasets/mhoosen/stain-norm/mitos_heldout
HELDOUT=${DATA_ROOT}/pairs/heldout_frames.csv
A2H_LORA=${DATA_ROOT}/lora/a2h_r8/final
H2A_LORA=${DATA_ROOT}/lora/h2a_r8/final
OUT_DIR=${DATA_ROOT}/eval/roundtrip_a06
export HF_HOME=/datasets/mhoosen/hf_cache

STRENGTHS=${1:-"0.2"}
LIMIT=${2:-0}
STEPS=${3:-50}

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import transformers, peft, accelerate, safetensors, diffusers, cv2" 2>/dev/null; then
    pip install --quiet transformers peft accelerate safetensors diffusers opencv-python-headless
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "A2H_LoRA=$A2H_LORA  H2A_LoRA=$H2A_LORA  (no ControlNet, DDIM only)"
echo "Strengths='$STRENGTHS'  Limit=$LIMIT  Steps=$STEPS"
echo "Out=$OUT_DIR"
echo "=================================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import torch; print('cuda available:', torch.cuda.is_available())"
# fail fast on a GPU-less node instead of crawling on CPU
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo 'ABORT: no GPU on this node -- resubmit.'; exit 1; }
echo "------------------------------------------------------------------"

cd "${CODE_DIR}/src/eval"
python roundtrip_a06.py \
    --a2h-lora "${A2H_LORA}" \
    --h2a-lora "${H2A_LORA}" \
    --root     "${MITOS_ROOT}" \
    --heldout  "${HELDOUT}" \
    --out      "${OUT_DIR}" \
    --slide A06 \
    --strengths ${STRENGTHS} \
    --steps "${STEPS}" \
    --limit "${LIMIT}" \
    --max-crops-per-frame 4 \
    --seed 0

echo "------------------------------------------------------------------"
echo "Round-trip done. original/ + hhat/ + ahat/ + per_crop.csv + summary.csv in ${OUT_DIR}"
