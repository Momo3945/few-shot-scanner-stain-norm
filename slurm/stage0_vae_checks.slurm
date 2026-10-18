#!/bin/bash
# =====================================================================
# stage0_vae_checks.slurm -- P1-18 Stage 0.1 + 0.4 (CPU, stampede).
#   0.1 weight diff: stock SD1.5 VAE vs sd-vae-ft-mse, per module.
#   0.4 LPIPS: `pip install --no-deps lpips` if missing (torch is NEVER touched --
#       torch version is printed before/after and the job aborts if it changed),
#       fetch the VGG16 backbone into TORCH_HOME, verify real byte size + a
#       sanity forward pass. Needs outbound internet from the compute node
#       (same assumption as fetch_sd_vae_ft_mse.slurm).
# Outputs: /datasets/mhoosen/stain-norm/vae/stage0/{weight_diff,lpips_check}.json
#
# Submit with no args:
#   sbatch slurm/stage0_vae_checks.slurm
# =====================================================================

#SBATCH --job-name=p1_18_stage0
#SBATCH --partition=stampede
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail

CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_ROOT=/datasets/mhoosen/stain-norm
OUT_DIR=${DATA_ROOT}/vae/stage0

source /home-mscluster/mhoosen/miniconda3/etc/profile.d/conda.sh
conda activate stainnorm
export HF_HOME=/datasets/mhoosen/hf_cache
export TORCH_HOME=/datasets/mhoosen/torch_cache
export HF_HUB_OFFLINE=1          # both VAEs are already cached; fail rather than silently re-download
export PYTHONUNBUFFERED=1
mkdir -p "$OUT_DIR" "$TORCH_HOME"

TORCH_BEFORE=$(python -c "import torch; print(torch.__version__)")
echo "torch before: $TORCH_BEFORE"

echo "=== 0.1 weight diff ==="
python "${CODE_DIR}/src/eval/vae_stage0_checks.py" --weight-diff --out "$OUT_DIR"

echo "=== 0.4 lpips ==="
if ! python -c "import lpips" 2>/dev/null; then
    echo "lpips missing -> pip install --no-deps lpips"
    pip install --no-deps lpips
fi
TORCH_AFTER=$(python -c "import torch; print(torch.__version__)")
echo "torch after:  $TORCH_AFTER"
if [[ "$TORCH_BEFORE" != "$TORCH_AFTER" ]]; then
    echo "ABORT: torch version changed (pinned build must not change)."
    exit 1
fi
# Backbone download needs the network; the HF offline flag is irrelevant to torchvision's
# download but unset it anyway so nothing in the lpips import path is blocked.
HF_HUB_OFFLINE=0 python "${CODE_DIR}/src/eval/vae_stage0_checks.py" --lpips --out "$OUT_DIR"

echo "STAGE0 CHECKS COMPLETE"
ls -la "$OUT_DIR" "$TORCH_HOME/hub/checkpoints"
