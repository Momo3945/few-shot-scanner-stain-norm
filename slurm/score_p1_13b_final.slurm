#!/bin/bash
# =====================================================================
# score_p1_13b_final.slurm -- P1-13b final smoke comparison: score
# checkpoint-250 @ guidance=1.0 (full-UNet scope) against the EXISTING
# (unchanged) generic_lcm_correct arm only -- deliberately does NOT pool
# with either of the two prior task_specific_correct dirs (g=2.0 original,
# g=1.0 attention-only), since all three share the identical
# "task_specific_correct" source_mode bucket and would get conflated by
# score_p1_10_ablation.py's grouping. See tickets/PHASE1-TICKETS.md P1-13b.
#
# Submit with no args:
#   sbatch score_p1_13b_final.slurm
# =====================================================================

#SBATCH --job-name=score_p1_13b_final
#SBATCH --partition=stampede
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail

DATA_ROOT=/datasets/mhoosen/stain-norm
OUT_DIR=${DATA_ROOT}/eval/p1_13b_ctrl_final_summary

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Out: $OUT_DIR"
echo "=================================================================="

cd /home-mscluster/mhoosen/stain-norm/src/eval
python score_p1_10_ablation.py \
    --eval-dirs "${DATA_ROOT}"/eval/p1_13b_ctrl_task_specific_correct_g1 \
                "${DATA_ROOT}"/eval/p1_13_ctrl_generic_lcm_correct \
    --out "${OUT_DIR}"

echo "------------------------------------------------------------------"
echo "Scoring done. Summary in ${OUT_DIR}"
