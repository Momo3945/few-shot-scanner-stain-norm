# CLAUDE.md — Histopathology Stain-Normalisation (BSc Honours)

## Project
Few-shot H&E **scanner** stain-normalisation (Aperio↔Hamamatsu) via a frozen
Stable Diffusion 1.5 base + colour LoRA + ControlNet + LCM-LoRA. Ablation ladder
A0–A5 on SD 1.5 (Phase 1), evaluation (Phase 2), SDXL portability transfer
(Phase 3), SD 3.5 feasibility probe (auxiliary, descopable). Supervisor: Richard
Klein. Public datasets only (MITOS-ATYPIA-14, CAMELYON17, TCGA-BRCA, PanNuke, Lizard).

## Identity — git commits (READ FIRST)
- Author: **Muhammad Hoosen** — cluster username `mhoosen`.
- **ALL Git commits and Git attribution in this repository must be in my name,
  Muhammad Hoosen.** Never author or attribute a commit to "Claude", "Anthropic",
  an assistant/bot account, or any other identity. This applies to both the Git
  author and committer identity.
- Before **every commit** (not only the first commit of a session), verify the
  repository-local identity with:
    `git config --local user.name`
    `git config --local user.email`
  The name MUST resolve to `Muhammad Hoosen`. If it does not, set it with:
    `git config --local user.name "Muhammad Hoosen"`
- Use the repository-local Git identity, not a global override. To diagnose where
  a value is coming from when needed, check:
    `git config --show-origin --get user.name`
    `git config --show-origin --get user.email`
- If `user.email` is not already configured for this repo, ASK me for it — never
  invent, guess, or substitute an email address. Once supplied, set it locally with:
    `git config --local user.email "<my email>"`
- If GitHub authentication is available through the GitHub MCP server/token (or
  `gh`), Claude may verify which GitHub account is authenticated. For example, with
  `gh`: `gh api user --jq '.login'` or `gh auth status`. The GitHub username is
  separate from `git config user.name` and must never replace `Muhammad Hoosen` as
  the commit author name.
- Before committing, also verify the staged commit will use my identity (for example
  by checking the local config above). If an accidental commit is ever created with
  the wrong author/committer identity, STOP and tell me before rewriting history.
- Commit messages should be factual and specific (what changed, not "updates").
- Do not add a `Co-Authored-By: Claude`, `Co-Authored-By: Anthropic`, bot trailer,
  assistant signature, or similar metadata. Commits represent **my authorship** of
  the research and should appear as mine on GitHub.

## Where things run
- This repo is edited **locally on Windows**. It is the source of truth.
- ALL compute runs on the Wits `mscluster` HPC. Never train/infer locally.
- Drive the cluster over SSH, e.g.: `ssh mhoosen@146.141.21.100 'squeue --me'`.
- Sync code up before submitting jobs:
  `rsync -av ./src ./slurm mhoosen@146.141.21.100:/home-mscluster/mhoosen/stain-norm/`

## Permission rule — Slurm job control (explicit, matches .claude/settings.json)
- **Always ask for my explicit confirmation before running `sbatch` or `scancel`**,
  even for jobs I've run before, even for "just a re-run". State exactly which
  script and arguments you intend to submit and wait for my go-ahead.
- Read-only cluster queries (`squeue`, `sinfo`, `cat`/`tail` on logs, `du`, `find`,
  `ls`) do not need permission — check freely to answer status questions.
- Never run `scancel -u` (cancels ALL my jobs) without explicit confirmation of
  which jobs and why.

## Cluster facts (hard-won — DO NOT re-derive or assume otherwise)
- Login: `146.141.21.100`, user `mhoosen`. Home `/home-mscluster/mhoosen` ≈ 50 GB, CODE ONLY.
- Large data, model cache, and all job OUTPUTS live in `/datasets/mhoosen/...` (201 TB).
- **`/gluster` DOES NOT EXIST on this cluster.** Use `/datasets` for outputs/checkpoints.
- Conda env `stainnorm` (Python 3.10, torch 2.5.1+cu121). In jobs, activate with:
  `source ~/miniconda3/etc/profile.d/conda.sh && conda activate stainnorm`
- Model cache: `export HF_HOME=/datasets/mhoosen/hf_cache`.
  SD1.5 + LCM-LoRA + ControlNet-Canny: full weights, healthy.
  **SDXL base and SD3.5-large: KNOWN BROKEN — only config/tokenizer files cached
  (~1–5 MB), no .safetensors.** A prior `fetch_models.slurm` run logged "✓ Downloaded"
  and exited 0 despite this — job success/log message is NOT proof weights are complete.
  Must re-run `hf download` for these two WITHOUT the `--include` filter before Phase 3
  or the SD3.5 probe can start. Always verify with `hf cache scan` or checked file sizes,
  never trust an exit code alone for model downloads.
- **GPUs are selected by PARTITION ONLY. `#SBATCH --gres=gpu:1` is REJECTED** ("Invalid gres").
  Partitions confirmed via `sinfo`: `stampede` (CPU/weak GPU), `bigbatch` (RTX 3090 24GB —
  default for GPU work), `biggpu` (mature jobs only), plus a default `batch*` and a
  `gpuexpress` partition seen in `sinfo` but not yet characterised — investigate before
  relying on either.
- **Never run heavy work on the login node** — always `sbatch`, or a short `srun --pty` for tests.
- Some `bigbatch` nodes come up GPU-less; GPU jobs must fail-fast if
  `torch.cuda.is_available()` is False (do not CPU-crawl — this has happened before
  and burned ~1 hour on job 3809).
- Load shedding is real: prefer checkpointed/resumable jobs; run long downloads via `sbatch`.
- A job leaving `squeue` is NOT proof of success — always read the `.out`/`.err` logs.

## Layout (CANONICAL — this section is authoritative; README.md defers to this)
- `src/data/`  extract_pairs.py, inspect_mitos.py, sample_hist_mitos.py
- `src/train/` train_colour_lora.py
- `src/eval/`  registration.py, metrics.py, progress.py, infer_colour_lora.py,
              score_outputs.py (these import each other as siblings — keep co-located)
- `slurm/`     train_colour_lora.slurm, fetch_models.slurm, infer_colour_lora.slurm,
              score_outputs.slurm       `logs/` job logs
- `tickets/`   task breakdown per phase, derived from proposal.tex (see below)
- The flat layout above is where ALL real code, checkpoints, and results actually
  live (confirmed by direct cluster audit). Any `phase1_ablation/`, `phase2_evaluation/`,
  `phase3_sdxl/`, `probe_sd35/` folders are **documentation-only** (an INPUTS.md per
  phase describing what that phase consumes/produces and pointing at the real flat
  paths above) — they are NOT real working directories and nothing should be written
  into subfolders under them.

## Tickets — task tracking against the proposal
- `tickets/` contains one file per phase (PHASE1-TICKETS.md, PHASE2-TICKETS.md,
  PHASE3-TICKETS.md, PROBE-SD35-TICKETS.md) plus a README.md index.
- **`docs/proposal.tex` (the graded research proposal) is the SOURCE OF TRUTH for
  scope.** Every ticket cites the proposal section/label it comes from. When scope
  is ambiguous or a ticket needs writing/updating, consult the proposal directly —
  do not invent scope not grounded in it.
- If the proposal file is still named `proposal_draft(6).tex` rather than
  `proposal.tex`, treat them as the same file; renaming to `proposal.tex` is fine
  but not required.
- Keep ticket status current: when a job completes (verified via cluster logs, not
  assumed from a submission), update the relevant ticket's Status field and note
  the Slurm job ID as evidence. Cross-check against the cluster before marking
  anything DONE — a job leaving the queue is not proof of success (see above).
- When starting new work, check `tickets/` first for whether it's already scoped;
  if not, propose a new ticket (with a proposal citation) before writing code.

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
- Diffusion training loss (the per-step MSE printed during LoRA training) is noisy and
  plateaus near a floor regardless of whether the adapter is useful — it is NOT a success
  signal. The only valid success signal is held-out LAB Wasserstein / SSIM recovery delta
  from `score_outputs.py`, never the training loss curve alone.

## Do NOT touch
- Raw datasets under `/datasets` (read-only in spirit). Held-out frames. `heldout_frames.csv`.
- **Never reinstall or upgrade `torch` inside a job** — the CUDA build is pinned and correct.

## Workflow expectations
- Use **plan mode** for anything multi-file; I review the plan before you apply it.
- Commit to git before large changes (see Identity section for commit authorship rules).
- Never `git push`/force without asking.
- Never run destructive cluster commands (`scancel -u`, `rm -rf` on /datasets) without confirming.
- After any architecture decision or newly discovered cluster gotcha, UPDATE THIS FILE.

## Reference docs (in repo)
- `docs/proposal.tex` (or `docs/proposal_draft(6).tex`) — the graded research proposal.
  SOURCE OF TRUTH for scope and the basis for all tickets.
- `docs/wits_mscluster_comprehensive_guide.md` — cluster user guide. MOTD/live system overrides it.
