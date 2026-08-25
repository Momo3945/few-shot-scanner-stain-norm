#!/bin/bash
# =====================================================================
# score_p1_13_lcm_controls.slurm -- P1-13 post-training controls: score
# all 5 control arms (task_specific x correct/zero/shuffled,
# adapter_disabled x correct, generic_lcm x correct) together via the
# EXISTING, unmodified score_p1_10_ablation.py -- its per-mode aggregation
# groups by the manifest's "source_mode" column, which
# infer_p1_13_lcm_controls.py writes as "<variant>_<source_mode>", so every
# control arm lands in its own bucket automatically. On stampede (CPU).
# See tickets/PHASE1-TICKETS.md P1-13.
#
# Submit with no args (after all 5 infer_p1_13_lcm_controls.slurm jobs
# have completed):
#   sbatch score_p1_13_lcm_controls.slurm
# =====================================================================

#SBATCH --job-name=score_p1_13_ctrl
#SBATCH --partition=stampede
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:30:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail

CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_ROOT=/datasets/mhoosen/stain-norm
OUT_DIR=${DATA_ROOT}/eval/p1_13_ctrl_summary

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Out: $OUT_DIR"
echo "=================================================================="

cd "${CODE_DIR}/src/eval"
python score_p1_10_ablation.py \
    --eval-dirs "${DATA_ROOT}"/eval/p1_13_ctrl_task_specific_correct \
                "${DATA_ROOT}"/eval/p1_13_ctrl_task_specific_zero \
                "${DATA_ROOT}"/eval/p1_13_ctrl_task_specific_shuffled \
                "${DATA_ROOT}"/eval/p1_13_ctrl_adapter_disabled_correct \
                "${DATA_ROOT}"/eval/p1_13_ctrl_generic_lcm_correct \
    --out "${OUT_DIR}"

echo "------------------------------------------------------------------"
echo "Scoring done. Summary in ${OUT_DIR}"
