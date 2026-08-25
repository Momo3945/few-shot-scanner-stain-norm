#!/bin/bash
# =====================================================================
# score_p1_13_checkpoint_sweep.slurm -- P1-13 inference-only guidance sweep:
# score every saved checkpoint of the real 39-pair run x student
# guidance_scale in {1.0, 1.5, 2.0}, against BOTH the frozen P1-10 teacher
# (teacher-fidelity) and the real paired Hamamatsu ground truth (real-H) --
# internal validation split only, no held-out slides, no new training.
# Motivated by a train/inference guidance-semantics mismatch: this run's
# w~U[1.0,2.0] training range corresponds to diffusers-style guidance
# G~U[2.0,3.0] (G = w + 1, LCM paper CFG form vs diffusers CFG form), so
# Control 1's guidance_scale=2.0 result should not be trusted until this
# sweep is done. See tickets/PHASE1-TICKETS.md P1-13.
#
# Submit with no args:
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 score_p1_13_checkpoint_sweep.slurm
# =====================================================================

#SBATCH --job-name=p1_13_gsweep
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
DATA_DIR=/datasets/mhoosen/stain-norm/pairs/train
TEACHER_DIR=/datasets/mhoosen/stain-norm/lora/a2h_cond_r8/best
RUN_DIR=/datasets/mhoosen/stain-norm/lora/a2h_cond_r8_lcm_distilled_lr0.00003_ga4
VAL_FRAMES_JSON=${RUN_DIR}/training_config.json
OUT_JSON=${RUN_DIR}/guidance_sweep.json
export HF_HOME=/datasets/mhoosen/hf_cache

mkdir -p "${CODE_DIR}/logs"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import transformers, peft, accelerate, safetensors, diffusers, cv2, skimage" 2>/dev/null; then
    echo "Installing missing Python deps..."
    pip install --quiet transformers peft accelerate safetensors diffusers opencv-python-headless scikit-image
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "P1-13 guidance sweep -- RunDir=$RUN_DIR"
echo "ValFramesJson=$VAL_FRAMES_JSON"
echo "Out=$OUT_JSON"
echo "=================================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed -- no GPU visible?"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo 'ABORT: no GPU on this node -- resubmit.'; exit 1; }
echo "------------------------------------------------------------------"

cd "${CODE_DIR}/src/eval"
python score_p1_13_checkpoint_sweep.py \
    --pairs-dir "${DATA_DIR}" --val-frames-json "${VAL_FRAMES_JSON}" \
    --teacher-dir "${TEACHER_DIR}" --run-dir "${RUN_DIR}" \
    --student-guidances 1.0 1.5 2.0 \
    --out "${OUT_JSON}"

echo "------------------------------------------------------------------"
echo "Job $SLURM_JOB_ID finished. Results: ${OUT_JSON}"
