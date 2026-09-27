#!/bin/bash
# =====================================================================
# floor_check_pixcell.slurm -- PXC-02 Part B: zero-shot (no fine-tuning)
# PixCell generation floor vs real registered held-out targets. See
# tickets/PROBE-PIXCELL-TICKETS.md, src/eval/floor_check_pixcell.py.
#
# NOT part of the graded proposal scope -- see that ticket file's caveat.
#
# Submit with:  [LIMIT] [OUT_TAG]
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44,mscluster75,mscluster74,mscluster51,mscluster83,mscluster76,mscluster50,mscluster45,mscluster57,mscluster59,mscluster62 \
#     -J pxc_02b_floor floor_check_pixcell.slurm 2 pxc_02b_floor_check
# =====================================================================

#SBATCH --job-name=pxc_02b_floor
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -uo pipefail

CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_ROOT=/datasets/mhoosen/stain-norm
export HF_HOME=/datasets/mhoosen/hf_cache

LIMIT=${1:-2}
OUT_TAG=${2:-pxc_02b_floor_check}
OUT_DIR=${DATA_ROOT}/eval/${OUT_TAG}

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import timm" 2>/dev/null; then
    pip install --quiet timm
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Limit=$LIMIT  Out=$OUT_DIR"
echo "=================================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed -- no GPU visible?"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo 'ABORT: no GPU on this node -- resubmit.'; exit 1; }
echo "------------------------------------------------------------------"

cd "${CODE_DIR}/src/eval"
# --online: trust_remote_code re-checks the Hub for the pipeline's commit
# hash on every load, confirmed empirically (job 61122 failed offline even
# though the pipeline code was already cached from PXC-01a's own run) --
# same reasoning as probe_pixcell_pipeline.slurm's own --online flag.
python floor_check_pixcell.py \
    --root "${DATA_ROOT}/mitos_heldout" \
    --heldout "${DATA_ROOT}/pairs/heldout_frames.csv" \
    --limit "${LIMIT}" \
    --out "${OUT_DIR}" --online

echo "------------------------------------------------------------------"
echo "Floor check finished. Results in ${OUT_DIR}"
