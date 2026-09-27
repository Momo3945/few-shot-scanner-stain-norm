#!/bin/bash
# =====================================================================
# probe_pixcell_pipeline.slurm -- PXC-01a: the mandatory wiring/smoke
# check before any PixCell LoRA training or held-out scoring is written.
# See tickets/PROBE-PIXCELL-TICKETS.md, src/eval/probe_pixcell_pipeline.py's
# own docstring for the real unknowns this resolves.
#
# NOT part of the graded proposal scope -- see that ticket file's caveat.
#
# Submit with no args (real crop path is the only thing that might need
# adjusting if pairs/train/ layout differs):
#   sbatch --exclude=mscluster48,mscluster65,mscluster46,mscluster44,mscluster75,mscluster74,mscluster51,mscluster83,mscluster76,mscluster50,mscluster45,mscluster57,mscluster59,mscluster62 \
#     -J pxc_01a_probe probe_pixcell_pipeline.slurm
# =====================================================================

#SBATCH --job-name=pxc_01a_probe
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -uo pipefail

CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_ROOT=/datasets/mhoosen/stain-norm
SOURCE_CROP=${SOURCE_CROP:-${DATA_ROOT}/pairs/train/A03_00A_c000_aperio.png}
OUT_DIR=${DATA_ROOT}/eval/pxc_01a_probe
export HF_HOME=/datasets/mhoosen/hf_cache

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import timm" 2>/dev/null; then
    echo "Installing missing Python dep: timm ..."
    pip install --quiet timm
else
    echo "timm present."
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "SourceCrop=$SOURCE_CROP  Out=$OUT_DIR"
echo "=================================================================="
nvidia-smi || echo "WARNING: nvidia-smi failed -- no GPU visible?"
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo 'ABORT: no GPU on this node -- resubmit.'; exit 1; }
echo "------------------------------------------------------------------"

cd "${CODE_DIR}/src/eval"
# --online: trust_remote_code's custom pipeline (Python code, not just
# weights) has never been fetched on this cluster before -- needs real
# network access on this first run, same reasoning as fetch_models.slurm's
# own "first run downloads via the compute node's outbound access" note.
python probe_pixcell_pipeline.py \
    --source-crop "${SOURCE_CROP}" \
    --out "${OUT_DIR}" \
    --seed 0 --online

echo "------------------------------------------------------------------"
echo "Probe finished (check exit code above). Report + output PNG in ${OUT_DIR}"
