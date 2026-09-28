#!/bin/bash
# =====================================================================
# diagnose_direction_colour_bias.slurm -- one-off diagnostic (CPU, stampede):
# why is A2H/H2A colour recovery asymmetric and backbone-dependent?
# See src/eval/diagnose_direction_colour_bias.py's docstring and
# tickets/PHASE3-TICKETS.md P3-06b/P3-07 H2A's closing finding.
#
# Submit with no args:
#   sbatch diagnose_direction_colour_bias.slurm
#
# MODE (env var, not positional -- additive, defaults to "zero" so the
# original invocation above is unchanged): the text2img follow-up (D2)
# compares diagnose_text2img_colour_default.py's samples instead:
#   MODE=text2img sbatch diagnose_direction_colour_bias.slurm
# =====================================================================

#SBATCH --job-name=diagnose_direction_bias
#SBATCH --partition=stampede
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail

CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_ROOT=/datasets/mhoosen/stain-norm
MODE=${MODE:-zero}

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm

echo "=================================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Mode=$MODE"
echo "=================================================================="

cd "${CODE_DIR}/src/eval"
python diagnose_direction_colour_bias.py --root "${DATA_ROOT}" --mode "${MODE}"

echo "------------------------------------------------------------------"
echo "Diagnostic done."
