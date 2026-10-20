#!/bin/bash
# =====================================================================
# benchmark_inference_speed.slurm -- P2-16: per-crop inference time of every method on one GPU node
# (final SD1.5 pipeline, generic LCM at 4 real steps, SDXL @1024, learned and classical baselines).
# Times the per-crop call only (no loading, registration or scoring); writes
# eval/p2_16_inference_speed{_smoke}/{timings,summary}.csv. See tickets/PHASE2-TICKETS.md P2-16.
#
# Smoke (2 timed crops, SD1.5 + one baseline):
#   PYTHONUNBUFFERED=1 sbatch --exclude=<BAD_NODES> --time=00:30:00 -J p2_16_smoke \
#     slurm/benchmark_inference_speed.slurm smoke
# Full (16 timed crops, all methods):
#   PYTHONUNBUFFERED=1 sbatch --exclude=<BAD_NODES> --time=01:30:00 -J p2_16_speed \
#     slurm/benchmark_inference_speed.slurm full
# =====================================================================

#SBATCH --job-name=p2_16_speed
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.out
#SBATCH --error=/home-mscluster/mhoosen/stain-norm/logs/%x.%j.err

set -euo pipefail

CODE_DIR=/home-mscluster/mhoosen/stain-norm
DATA_ROOT=/datasets/mhoosen/stain-norm
export HF_HOME=/datasets/mhoosen/hf_cache

RUN=${1:?"Usage: sbatch benchmark_inference_speed.slurm smoke|full"}
case "${RUN}" in
    smoke) OUT=${DATA_ROOT}/eval/p2_16_inference_speed_smoke; N=2; METHODS=(sd15_inv_f3 sd15_lcm_4step stainnet macenko) ;;
    full)  OUT=${DATA_ROOT}/eval/p2_16_inference_speed;       N=16; METHODS=() ;;
    *) echo "unknown run '${RUN}'"; exit 1 ;;
esac
METHOD_ARGS=()
if [[ ${#METHODS[@]} -gt 0 ]]; then METHOD_ARGS=(--methods "${METHODS[@]}"); fi

echo "Job ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-?}  run=${RUN}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

source ~/miniconda3/etc/profile.d/conda.sh && conda activate stainnorm
python -c "import torch,sys; ok=torch.cuda.is_available(); print('cuda', ok); sys.exit(0 if ok else 3)" \
    || { echo "ABORT: no GPU on ${SLURMD_NODENAME:-?}"; exit 3; }

cd "${CODE_DIR}/src/eval"
python benchmark_inference_speed.py \
    --root "${DATA_ROOT}/mitos_heldout" --heldout "${DATA_ROOT}/pairs/heldout_frames.csv" \
    --sd15-ckpt "${DATA_ROOT}/lora/a2h_cond_r8/best" \
    --sdxl-ckpt "${DATA_ROOT}/lora/a2h_cond_r8_sdxl_1024_overlap144/best" \
    --baselines-dir "${DATA_ROOT}/baselines" \
    --out "${OUT}" --n "${N}" --warmup 2 ${METHOD_ARGS[@]+"${METHOD_ARGS[@]}"}

echo "------------------------------------------------------------------"
echo "Done. Summary: ${OUT}/summary.csv"
