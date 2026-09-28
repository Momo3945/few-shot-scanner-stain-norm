# Reproducibility

All training and inference for this project runs on Slurm-managed HPC
compute (the University of the Witwatersrand's `mscluster`), never locally —
the local repo checkout is source-of-truth for code, not a place to run
GPU jobs.

## Environment

- Python 3.10
- PyTorch 2.5.1 (CUDA 12.1 build)
- See [`requirements.txt`](../requirements.txt) for the full pinned
  dependency list (`diffusers`, `transformers`, `accelerate`, `peft`,
  `opencv-python`, `scikit-image`, `scipy`, `numpy`, `Pillow`, `tifffile`).

```bash
conda create -n stainnorm python=3.10
conda activate stainnorm
pip install -r requirements.txt
```

Model weights (Stable Diffusion 1.5, SDXL base, SD3.5-large, the matching
LCM-LoRA and ControlNet-Canny adapters) are pulled from the Hugging Face Hub
into a shared cache and are not committed to this repo.

## Code layout

| Path | Contents |
|---|---|
| `src/data/` | Dataset extraction/inspection (`extract_pairs.py`, `inspect_mitos.py`, `sample_hist_mitos.py`) |
| `src/train/` | LoRA/ControlNet training scripts |
| `src/eval/` | Registration, metrics, inference, and scoring (these import each other as flat siblings — kept co-located deliberately) |
| `slurm/` | One `.slurm` launcher per training/inference/scoring script — each script's header comment documents its exact argument usage |
| `tickets/` | Per-phase task board — the authoritative "what's actually done" record, with Slurm job IDs as evidence |
| `docs/results/` | Archived per-experiment CSVs and the consolidated [`RESULTS_SUMMARY.md`](results/RESULTS_SUMMARY.md) |
| `docs/phases/` | Per-phase reference docs — what each phase consumes/produces and where |

## Running a training/inference/scoring cycle

Sync code to the cluster, then submit jobs via `sbatch` (GPU partitions are
selected by partition name, not `--gres`):

```bash
rsync -av ./src ./slurm <cluster-user>@<cluster-host>:<remote-code-root>/
ssh <cluster-user>@<cluster-host> 'sbatch slurm/train_colour_lora.slurm A2H 8'
ssh <cluster-user>@<cluster-host> 'sbatch slurm/infer_colour_lora.slurm a2h_r8 "0.3 0.4 0.5" 0'
ssh <cluster-user>@<cluster-host> 'sbatch slurm/score_outputs.slurm a2h_r8'
```

A job leaving the Slurm queue is not proof of success — always check the
job's `.out`/`.err` log under `logs/` before treating a run as complete.

## Scoring outputs

`src/eval/score_outputs.py` computes LAB-Wasserstein colour distance, SSIM,
PSNR, and MAE against the registered held-out ground truth, reporting both a
pooled aggregate and a per-slide breakdown (one held-out slide, A06, is a
confirmed colour-gap outlier and is never allowed to stand in for the pooled
number alone).

## Cluster-specific detail

Partition names, known bad/slow compute nodes, storage layout, and the full
list of hard-won cluster gotchas are maintained in
[`CLAUDE.md`](../CLAUDE.md) — that file is written for AI-agent-assisted
development on this specific cluster and isn't duplicated here, but it's the
canonical reference if you're setting up your own run.
