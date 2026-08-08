# CLAUDE.md — Histopathology Stain-Normalisation (BSc Honours)

## Project
Few-shot H&E **scanner** stain-normalisation (Aperio↔Hamamatsu) via a frozen
Stable Diffusion 1.5 base + colour LoRA + ControlNet + LCM-LoRA. Ablation ladder
A0–A5 on SD 1.5 (Phase 1), evaluation (Phase 2), SDXL portability transfer
(Phase 3), SD 3.5 feasibility probe (auxiliary, descopable). Supervisor: Richard
Klein. Public datasets only (MITOS-ATYPIA-14, CAMELYON17, TCGA-BRCA, PanNuke, Lizard).

## Where things run (READ FIRST)
- This repo is edited **locally on Windows**. It is the source of truth.
- ALL compute runs on the Wits `mscluster` HPC. Never train/infer locally.
- Drive the cluster over SSH, e.g.: `ssh mhoosen@146.141.21.100 'squeue --me'`.
- Sync code up before submitting jobs:
  `rsync -av ./src ./slurm mhoosen@146.141.21.100:/home-mscluster/mhoosen/stain-norm/`

## Cluster facts (hard-won — DO NOT re-derive or assume otherwise)
- Login: `146.141.21.100`, user `mhoosen`. Home `/home-mscluster/mhoosen` ≈ 50 GB, CODE ONLY.
- Large data, model cache, and all job OUTPUTS live in `/datasets/mhoosen/...` (201 TB).
- **`/gluster` DOES NOT EXIST on this cluster.** Use `/datasets` for outputs/checkpoints.
- Conda env `stainnorm` (Python 3.10, torch 2.5.1+cu121). In jobs, activate with:
  `source ~/miniconda3/etc/profile.d/conda.sh && conda activate stainnorm`
- Model cache: `export HF_HOME=/datasets/mhoosen/hf_cache` (SD1.5+adapters, SDXL, SD3.5 present).
- **GPUs are selected by PARTITION ONLY. `#SBATCH --gres=gpu:1` is REJECTED** ("Invalid gres").
  Partitions: `stampede` (CPU/weak GPU, use for CPU jobs), `bigbatch` (RTX 3090 24GB — default
  for GPU work), `biggpu` (mature jobs only).
- **Never run heavy work on the login node** — always `sbatch`, or a short `srun --pty` for tests.
- Some `bigbatch` nodes come up GPU-less; GPU jobs must fail-fast if
  `torch.cuda.is_available()` is False (do not CPU-crawl).
- Load shedding is real: prefer checkpointed/resumable jobs; run long downloads via `sbatch`.
- A job leaving `squeue` is NOT proof of success — always read the `.out`/`.err` logs.

## Layout
This is the **canonical** code layout — the only one with real, working files. Do not
follow README.md's older phase-based structure for code placement; see "Layout
convention" below.
- `src/data/`  extract_pairs.py, sample_hist_mitos.py, inspect_mitos.py
- `src/train/` train_colour_lora.py
- `src/eval/`  registration.py, metrics.py, progress.py, infer_colour_lora.py, score_outputs.py
              (these import each other as siblings — keep co-located)
- `slurm/`     *.slurm launchers          `logs/` job logs

### Layout convention (resolved 2026-08-08)
`phase1_ablation/`, `phase2_evaluation/`, `phase3_sdxl/`, `probe_sd35/` are
**documentation-only** — each contains a single `INPUTS.md` describing what that phase
consumes/produces and pointing at the real paths above (and under
`/datasets/mhoosen/stain-norm/` on the cluster). They do NOT hold code, configs,
checkpoints, or results — those formerly-empty subdirectories were removed. Never
recreate `checkpoints/`/`configs/`/`outputs/`/`results/` subfolders under a phase
folder; real artifacts live under `src/`, `slurm/`, and the cluster `/datasets` paths.

## Methodology guardrails (protect experiment validity — treat as invariants)
- Training pairs are **coordinate-corresponding, NOT pixel-exact**. Affine registration is for
  held-out PIXEL metrics only — never applied to training data.
- Held-out slides **A06/A08/A09/A13/A16 are NEVER used in any training**. Keep leak checks.
- **A06 is a genuine colour-gap outlier** (LAB Wasserstein ~95 vs ~25 for others, registration
  confirmed clean). Always report PER-SLIDE plus an outlier-excluded aggregate; never let the
  pooled ALL number stand alone. Outlier flagging is robust (median/MAD), not mean/std.
- Colour LoRA objective = target-domain denoising loss; direction comes from img2img at inference.
- Report both clean-slide and all-slide aggregates.
- MITOS training archives use a DOUBLED dir layout (`A03/A03/frames/x20`); testing does not.
  Frame IDs match across scanners (A03_00A ↔ H03_00A) = the coordinate correspondence.

## Do NOT touch
- Raw datasets under `/datasets` (read-only in spirit). Held-out frames. `heldout_frames.csv`.
- **Never reinstall or upgrade `torch` inside a job** — the CUDA build is pinned and correct.

## Workflow expectations
- Use **plan mode** for anything multi-file; I review the plan before you apply it.
- Commit to git before large changes. Never `git push`/force without asking.
- Never run destructive cluster commands (`scancel -u`, `rm -rf` on /datasets) without confirming.
- After any architecture decision or newly discovered cluster gotcha, UPDATE THIS FILE.

## Reference docs (in repo)
- `docs/proposal.pdf` — the graded research proposal (methodology source of truth).
- `docs/wits_mscluster_guide.md` — cluster user guide. MOTD/live system overrides it.
