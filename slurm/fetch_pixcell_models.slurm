#!/bin/bash
# =====================================================================
# fetch_pixcell_models.slurm -- PXC-00: fetch PixCell + its UNI2-h
# embedding-model dependency. See tickets/PROBE-PIXCELL-TICKETS.md.
#
# NOT part of the graded proposal scope -- see that ticket file's own
# caveat before running this for real. Kept as a separate script from
# fetch_models.slurm deliberately: that script uses `set -euo pipefail`
# and already has one permanently-failing line (gated SD3.5) baked in
# until manual approval lands, so appending more gated/manual-approval
# downloads there would make an already-fragile script harder to reason
# about. This script is a self-contained, isolated fetch -- failing here
# never affects fetch_models.slurm's own SD1.5/SDXL/SD3.5/VAE downloads.
#
# TWO MANUAL PREREQUISITES, neither scriptable (same shape as PR-00's
# SD3.5 gating in fetch_models.slurm):
#   1. UNI2-h is GATED and specifically requires an institutional email
#      as your HF account's PRIMARY email -- a personal Gmail-type
#      address is auto-denied per MahmoodLab's own stated policy. Request
#      access at https://huggingface.co/MahmoodLab/UNI2-h with that email
#      set first.
#   2. This cluster has no HF token configured at all as of 2026-09-27
#      (`hf auth whoami` -> "Not logged in", confirmed same as the SD3.5
#      probe's own PR-00 finding). Generate a token at
#      https://huggingface.co/settings/tokens and run `hf auth login`
#      (or set HF_TOKEN) on the cluster before this script can do anything
#      useful for UNI2-h. PixCell's own weights are NOT gated, so those
#      two lines below will succeed even without a token/login -- only
#      the UNI2-h line needs both prerequisites done.
#
# Submit with no args:
#   sbatch -J fetch_pixcell fetch_pixcell_models.slurm
# =====================================================================

#SBATCH --job-name=fetch_pixcell
#SBATCH --partition=stampede
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -uo pipefail

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm

export HF_HOME=/datasets/mhoosen/hf_cache
mkdir -p "$HF_HOME"
echo "HF_HOME = $HF_HOME"

if ! curl -sI https://huggingface.co | head -1 | grep -q HTTP; then
    echo "ERROR: this compute node cannot reach huggingface.co. Aborting." >&2
    exit 1
fi
echo "huggingface.co reachable -- starting downloads."

echo "== PixCell-256 (ungated) =="
hf download StonyBrook-CVLab/PixCell-256
PIXCELL_256_STATUS=$?

echo "== PixCell-1024 (ungated) =="
hf download StonyBrook-CVLab/PixCell-1024
PIXCELL_1024_STATUS=$?

echo "== UNI2-h (GATED -- requires institutional-email approval + hf auth login, see header) =="
hf download MahmoodLab/UNI2-h
UNI2H_STATUS=$?

echo "=================================================================="
echo "PixCell-256 exit: $PIXCELL_256_STATUS  PixCell-1024 exit: $PIXCELL_1024_STATUS  UNI2-h exit: $UNI2H_STATUS"
if [[ $UNI2H_STATUS -ne 0 ]]; then
    echo "UNI2-h download failed -- expected until both manual prerequisites in this"
    echo "script's header are done. PixCell itself may still have succeeded above."
fi
echo "Verify with real byte sizes (hf cache scan / du -sh), never trust exit codes alone."
