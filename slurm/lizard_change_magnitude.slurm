#!/bin/bash
# =====================================================================
# lizard_change_magnitude.slurm -- per-image change size of every Lizard normaliser (CPU,
# stampede) + its relation to the HoVer-Net Dice drop. P2-11b structure-safety arm.
# Run AFTER all seven normalised-image dirs and their score_lizard.slurm per_image.csv exist:
# diffusion (P2-08), StainNet/ParamNet/StainGAN, Macenko/Reinhard/Histogram Matching.
#
#   sbatch lizard_change_magnitude.slurm
# Output: eval/lizard_change_magnitude/{change_magnitude_per_image.csv,change_magnitude_summary.csv}
# =====================================================================

#SBATCH --job-name=lizard_change_mag
#SBATCH --partition=stampede
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_ROOT=/datasets/mhoosen/stain-norm
EVAL=${DATA_ROOT}/eval
OUT_DIR=${EVAL}/lizard_change_magnitude

mkdir -p "${CODE_DIR}/logs" "${OUT_DIR}"

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
if ! python -c "import numpy, scipy, skimage" 2>/dev/null; then
    pip install --quiet numpy scipy scikit-image
fi

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)   Out=$OUT_DIR"
echo "=================================================================="

cd "${CODE_DIR}/src/eval"
python lizard_change_magnitude.py \
    --images-dir "${DATA_ROOT}/lizard_heldout/images" \
    --eval-root  "${EVAL}" \
    --out        "${OUT_DIR}" \
    --run "Diffusion (P2-08)=lizard_normalised_images:lizard_normalised" \
    --run "StainNet=lizard_normalised_images_stainnet:lizard_normalised_stainnet" \
    --run "ParamNet=lizard_normalised_images_paramnet:lizard_normalised_paramnet" \
    --run "StainGAN=lizard_normalised_images_staingan:lizard_normalised_staingan" \
    --run "Macenko=lizard_normalised_images_macenko:lizard_normalised_macenko" \
    --run "Reinhard=lizard_normalised_images_reinhard:lizard_normalised_reinhard" \
    --run "Histogram Matching=lizard_normalised_images_histogram_matching:lizard_normalised_histogram_matching"

echo "------------------------------------------------------------------"
echo "Done."
