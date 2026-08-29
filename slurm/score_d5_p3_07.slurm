#!/bin/bash
# =====================================================================
# score_d5_p3_07.slurm -- P3-07 diagnostic D5: score all four
# colour-LoRA-scale runs (each independently -- all write
# source_mode=correct), then run d5_verify_p3_07.py to compute the
# internal-validation raw baseline and apply the selection rule (positive
# recovery required, highest SSIM among positive configs; else stop and
# recommend P3-07b). On stampede (CPU). Requires
# d5_infer_lora_scale_sweep_p3_07.slurm to have COMPLETED first.
# See tickets/PHASE3-TICKETS.md P3-07, D5.
#
# Submit with no args:
#   sbatch -J score_d5_p3_07 score_d5_p3_07.slurm
# =====================================================================

#SBATCH --job-name=score_d5_p3_07
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
OUT_DIR=${DATA_ROOT}/eval/p3_07_d5_summary

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "=================================================================="

cd "${CODE_DIR}/src/eval"
for LORA_SCALE in 1.0 1.25 1.5 2.0; do
    echo "-- scoring lora_scale=${LORA_SCALE} --"
    python score_p1_10_ablation.py \
        --eval-dirs "${DATA_ROOT}/eval/p3_07_d5_lora_${LORA_SCALE}" \
        --out "${DATA_ROOT}/eval/p3_07_d5_lora_${LORA_SCALE}_summary"
done

echo "------------------------------------------------------------------"
python d5_verify_p3_07.py \
    --pairs-dir "${PAIRS_DIR}" --val-frames-json "${VAL_FRAMES_JSON}" \
    --scale-summaries \
        1.0="${DATA_ROOT}/eval/p3_07_d5_lora_1.0_summary/per_crop.csv" \
        1.25="${DATA_ROOT}/eval/p3_07_d5_lora_1.25_summary/per_crop.csv" \
        1.5="${DATA_ROOT}/eval/p3_07_d5_lora_1.5_summary/per_crop.csv" \
        2.0="${DATA_ROOT}/eval/p3_07_d5_lora_2.0_summary/per_crop.csv" \
    --out "${OUT_DIR}/d5_verdict.csv"

echo "------------------------------------------------------------------"
echo "D5 scoring + verdict done. See ${OUT_DIR}"
