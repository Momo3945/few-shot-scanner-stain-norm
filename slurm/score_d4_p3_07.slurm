#!/bin/bash
# =====================================================================
# score_d4_p3_07.slurm -- P3-07 diagnostic D4: score all four
# controlnet_conditioning_scale runs (each independently -- all write
# source_mode=correct, so scoring together in one score_p1_10_ablation.py
# call would pool them indistinguishably), then run d4_verify_p3_07.py to
# compute the internal-validation raw baseline and print the recovery-vs-
# scale table + verdict. On stampede (CPU). Requires
# d4_infer_scale_sweep_p3_07.slurm to have COMPLETED first.
# See tickets/PHASE3-TICKETS.md P3-07, D4.
#
# Submit with no args:
#   sbatch -J score_d4_p3_07 score_d4_p3_07.slurm
# =====================================================================

#SBATCH --job-name=score_d4_p3_07
#SBATCH --partition=stampede
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail

CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_ROOT=/datasets/mhoosen/stain-norm
PAIRS_DIR=${DATA_ROOT}/pairs/train_1024
VAL_FRAMES_JSON=${DATA_ROOT}/lora/a2h_cond_r8_sdxl_1024/pair_manifest.json
OUT_DIR=${DATA_ROOT}/eval/p3_07_d4_summary

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "=================================================================="

cd "${CODE_DIR}/src/eval"
for SCALE in 0.25 0.50 0.75 1.00; do
    echo "-- scoring scale=${SCALE} --"
    python score_p1_10_ablation.py \
        --eval-dirs "${DATA_ROOT}/eval/p3_07_d4_scale_${SCALE}" \
        --out "${DATA_ROOT}/eval/p3_07_d4_scale_${SCALE}_summary"
done

echo "------------------------------------------------------------------"
python d4_verify_p3_07.py \
    --pairs-dir "${PAIRS_DIR}" --val-frames-json "${VAL_FRAMES_JSON}" \
    --scale-summaries \
        0.25="${DATA_ROOT}/eval/p3_07_d4_scale_0.25_summary/per_crop.csv" \
        0.50="${DATA_ROOT}/eval/p3_07_d4_scale_0.50_summary/per_crop.csv" \
        0.75="${DATA_ROOT}/eval/p3_07_d4_scale_0.75_summary/per_crop.csv" \
        1.00="${DATA_ROOT}/eval/p3_07_d4_scale_1.00_summary/per_crop.csv" \
    --out "${OUT_DIR}/d4_verdict.csv"

echo "------------------------------------------------------------------"
echo "D4 scoring + verdict done. See ${OUT_DIR}"
