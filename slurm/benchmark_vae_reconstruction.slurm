#!/bin/bash
# =====================================================================
# benchmark_vae_reconstruction.slurm -- P1-14: pure VAE self-reconstruction
# benchmark, stock SD1.5 VAE vs stabilityai/sd-vae-ft-mse. No UNet, no
# ControlNet, no LoRA, no noise, no DDIM -- isolates decoder/autoencoder
# reconstruction quality only. See tickets/P1-14_vae_reconstruction_benchmark.md
# and tickets/PHASE1-TICKETS.md P1-14.
#
# Submit with positional args:  STAGE [VAE] [OUT_TAG]
#   STAGE = internal | heldout
#   VAE   = stock | ft_mse | both (default both)
#
#   # Stage A -- internal validation (both scanner domains, non-held-out pairs):
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 \
#     benchmark_vae_reconstruction.slurm internal both p1_14_stage_a
#
#   # Stage B -- canonical 496-crop held-out confirmation (only if Stage A gates open):
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 --time=03:00:00 \
#     benchmark_vae_reconstruction.slurm heldout both p1_14_stage_b
#
#   # Qualitative panel only (set QUALITATIVE_SLIDES, not a positional arg --
#   # skips the expensive scoring pass, e.g. after Stage B already ran):
#   QUALITATIVE_SLIDES="A06 A08" sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44 \
#     benchmark_vae_reconstruction.slurm heldout both p1_14_stage_b_qualitative
# =====================================================================

#SBATCH --job-name=p1_14_vae_bench
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
export HF_HOME=/datasets/mhoosen/hf_cache

STAGE=${1:?"Usage: sbatch benchmark_vae_reconstruction.slurm STAGE [VAE] [OUT_TAG]  (STAGE = internal | heldout)"}
VAE=${2:-both}
OUT_TAG=${3:-p1_14_${STAGE}}
OUT_DIR=${DATA_ROOT}/eval/${OUT_TAG}
QUAL_FLAG=""
if [[ -n "${QUALITATIVE_SLIDES:-}" ]]; then
    QUAL_FLAG="--qualitative-only --qualitative-slides ${QUALITATIVE_SLIDES}"
fi

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import transformers, diffusers, cv2, skimage" 2>/dev/null; then
    echo "Installing missing Python deps..."
    pip install --quiet transformers diffusers opencv-python-headless scikit-image
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "P1-14 -- Stage=$STAGE  VAE=$VAE  Out=$OUT_DIR"
echo "=================================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed -- no GPU visible?"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo 'ABORT: no GPU on this node -- resubmit.'; exit 1; }
echo "------------------------------------------------------------------"

cd "${CODE_DIR}/src/eval"
if [[ "${STAGE}" == "internal" ]]; then
    python benchmark_vae_reconstruction.py \
        --stage internal --vae "${VAE}" \
        --pairs-dir "${DATA_ROOT}/pairs/train" \
        --out "${OUT_DIR}"
else
    python benchmark_vae_reconstruction.py \
        --stage heldout --vae "${VAE}" \
        --root "${DATA_ROOT}/mitos_heldout" \
        --heldout "${DATA_ROOT}/pairs/heldout_frames.csv" \
        --max-crops-per-frame 4 \
        --out "${OUT_DIR}" ${QUAL_FLAG}
fi

echo "------------------------------------------------------------------"
echo "Job $SLURM_JOB_ID finished. Results in ${OUT_DIR}"
