#!/bin/bash
# =====================================================================
# infer_p1_10_strength_sweep.slurm -- P1-10 follow-up: strength sweep on the
# trained source-conditioned checkpoint (lora/a2h_cond_r8/best).
#
# Motivation: the full held-out run (job 44542, all 496 crops) used the
# ticket's mandated staged-rollout settings (plain 50-step DDIM, strength
# 0.50, no LCM/ControlNet acceleration) -- a deliberately conservative first
# test, not the tuned operating point. It found P1-10 clearly beats SD1.5's
# own prior best (A4/A5@0.20) on colour recovery but is roughly tied on SSIM
# (marginally behind, ~1%). This sweep checks whether a different strength
# (mirroring A4/A5's own finding that 0.20 beats 0.50 on SD1.5's target-only
# pipeline) does better on BOTH axes at once for the source-conditioned model.
#
# Diagnostic scope (not the full 496-crop set -- that's ~5x the compute of
# the already-long full run at a single strength): same A06+A08 subset used
# for the smoke gate (LIMIT=20 on the canonical heldout_frames.csv -> all 16
# A06 frames + 4 A08 frames), single seed (not the default 3) to keep this
# affordable. If a strength looks clearly better here, a full 496-crop run
# at that strength is the natural follow-up (separate, gated job).
#
# Submit once per strength value (STRENGTH positional, no default -- always
# state explicitly):
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 infer_p1_10_strength_sweep.slurm 0.20
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 infer_p1_10_strength_sweep.slurm 0.30
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 infer_p1_10_strength_sweep.slurm 0.40
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 infer_p1_10_strength_sweep.slurm 0.70
#   (0.50 already covered by the full held-out run, job 44542 -- not repeated here)
# Each writes to its own eval/p1_10_sweep_s<NN>/ dir, no collisions.
# =====================================================================

#SBATCH --job-name=p1_10_sweep
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
CHECKPOINT_DIR=${DATA_ROOT}/lora/a2h_cond_r8/best
export HF_HOME=/datasets/mhoosen/hf_cache

STRENGTH=${1:?"Usage: sbatch infer_p1_10_strength_sweep.slurm STRENGTH  (e.g. 0.20)"}
STRENGTH_TAG=$(echo "$STRENGTH" | tr -d '.')
OUT_DIR=${DATA_ROOT}/eval/p1_10_sweep_s${STRENGTH_TAG}

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import transformers, peft, accelerate, safetensors, diffusers, cv2" 2>/dev/null; then
    pip install --quiet transformers peft accelerate safetensors diffusers opencv-python-headless
fi

if [[ ! -f "${CHECKPOINT_DIR}/pytorch_lora_weights.safetensors" ]]; then
    echo "ABORT: ${CHECKPOINT_DIR}/pytorch_lora_weights.safetensors not found."
    exit 1
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "P1-10 strength sweep -- strength=${STRENGTH}  Checkpoint=${CHECKPOINT_DIR}"
echo "A06+A08 subset (LIMIT=20), single seed, 50-step DDIM"
echo "Out=$OUT_DIR"
echo "=================================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo 'ABORT: no GPU on this node -- resubmit.'; exit 1; }
echo "------------------------------------------------------------------"

cd "${CODE_DIR}/src/eval"
python infer_colour_translation.py \
    --lora "${CHECKPOINT_DIR}" --controlnet "${CHECKPOINT_DIR}/controlnet" \
    --source-mode correct \
    --root "${MITOS_ROOT}" --heldout "${HELDOUT}" --out "${OUT_DIR}" \
    --strength "${STRENGTH}" \
    --steps 50 \
    --limit 20 --max-crops-per-frame 4 \
    --seeds 0

echo "------------------------------------------------------------------"
echo "Inference done. Outputs + eval_manifest.csv in ${OUT_DIR}"
echo "Next: score_p1_10_ablation.py --eval-dirs ${OUT_DIR} --out ${OUT_DIR}_summary"
