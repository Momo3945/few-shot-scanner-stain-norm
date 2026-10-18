#!/bin/bash
# =====================================================================
# vae_only_newdecoder.slurm -- P1-18 G1.4 prerequisite: stock-encoder + NEW-decoder VAE-only
# reconstruction (A_V) of the held-out source crops, in the identity-manifest schema that
# fuse_source_detail.py --identity-manifest reads. Fast (no registration, no diffusion).
# See src/eval/vae_only_newdecoder.py and tickets/P1-18_vae_decoder_finetune.md (G1.4).
#
# Submit with:  DECODER_DIR OUT_TAG [IDENTITY_MANIFEST_REL]
#   DECODER_DIR = absolute path of a saved AutoencoderKL dir (e.g. .../vae/d1_pn03/best_val)
#   OUT_TAG     = subdir under eval/p1_18_g14/
#   IDENTITY_MANIFEST_REL = relative to eval/, default
#                 p1_10_ddim_inversion/identity_full_heldout/eval_manifest.csv (checksum-verified
#                 full-496 --mode identity run that P1-16's held-out fusion used)
#   sbatch --exclude=<BAD_NODES> --time=00:30:00 -J p1_18_g14_av vae_only_newdecoder.slurm \
#       /datasets/mhoosen/stain-norm/vae/d1_pn03/best_val vae_only_d1_pn03
# =====================================================================

#SBATCH --job-name=p1_18_vae_only
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail

CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_ROOT=/datasets/mhoosen/stain-norm
export HF_HOME=/datasets/mhoosen/hf_cache

DECODER_DIR=${1:?"Usage: sbatch vae_only_newdecoder.slurm DECODER_DIR OUT_TAG [IDENTITY_MANIFEST_REL]"}
OUT_TAG=${2:?"OUT_TAG required"}
IDENTITY_REL=${3:-p1_10_ddim_inversion/identity_full_heldout/eval_manifest.csv}
OUT_DIR=${DATA_ROOT}/eval/p1_18_g14/${OUT_TAG}

if [[ ! -f "${DECODER_DIR}/config.json" ]]; then
    echo "ABORT: ${DECODER_DIR} has no config.json (not a saved AutoencoderKL dir)."; exit 1
fi
if [[ ! -f "${DATA_ROOT}/eval/${IDENTITY_REL}" ]]; then
    echo "ABORT: identity manifest ${DATA_ROOT}/eval/${IDENTITY_REL} not found."; exit 1
fi
mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "P1-18 G1.4 A_V -- decoder=${DECODER_DIR}  identity=${IDENTITY_REL}  out=${OUT_DIR}"
echo "=================================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed -- no GPU visible?"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo 'ABORT: no GPU on this node -- resubmit.'; exit 1; }
echo "------------------------------------------------------------------"

cd "${CODE_DIR}/src/eval"
python vae_only_newdecoder.py \
    --identity-manifest "${DATA_ROOT}/eval/${IDENTITY_REL}" \
    --decoder-path "${DECODER_DIR}" \
    --out "${OUT_DIR}"

echo "------------------------------------------------------------------"
echo "Done. Manifest + reconstructions in ${OUT_DIR}"
