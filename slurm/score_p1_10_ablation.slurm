#!/bin/bash
# =====================================================================
# score_p1_10_ablation.slurm -- score infer_colour_translation.py-shaped
# outputs (seed/source_mode-shaped manifest -- NOT score_outputs.slurm's
# strength-shaped schema) vs held-out/training-pair ground truth (CPU).
#
#   sbatch score_p1_10_ablation.slurm "eval/p1_10_ablation_correct eval/p1_10_ablation_zero" p1_10_ablation_summary
#
# Space-separated EVAL_DIRS in one quoted arg (see infer_p1_10_ddim_inversion.
# slurm's header for why: ssh's argv-rejoin-then-remote-reparse round trip can
# silently split an unquoted multi-word arg). CPU-only (no model), runs on
# stampede.
# =====================================================================

#SBATCH --job-name=score_p1_10
#SBATCH --partition=stampede
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail

CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_ROOT=/datasets/mhoosen/stain-norm

EVAL_DIRS_ARG=${1:?"Usage: sbatch score_p1_10_ablation.slurm \"EVAL_DIR1 EVAL_DIR2 ...\" OUT_TAG"}
OUT_TAG=${2:?"OUT_TAG required"}
OUT_DIR=${DATA_ROOT}/eval/${OUT_TAG}

mkdir -p "${CODE_DIR}/logs"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import cv2, scipy, skimage" 2>/dev/null; then
    pip install --quiet opencv-python-headless scipy scikit-image tifffile
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Eval dirs=$EVAL_DIRS_ARG"
echo "Out=$OUT_DIR"
echo "=================================================================="

cd "${CODE_DIR}"
RESOLVED_DIRS=""
for d in ${EVAL_DIRS_ARG}; do
    RESOLVED_DIRS="${RESOLVED_DIRS} ${DATA_ROOT}/eval/${d}"
done

python src/eval/score_p1_10_ablation.py --eval-dirs ${RESOLVED_DIRS} --out "${OUT_DIR}"

echo "------------------------------------------------------------------"
echo "Scoring done. Output in ${OUT_DIR}"
