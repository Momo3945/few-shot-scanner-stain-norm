# Probe — SD3.5 Feasibility

**Documentation only.** This folder holds no code, configs, or results — see
CLAUDE.md's "Layout" section for the canonical code location.

**Auxiliary. Not a phase.** A feasibility check on whether SD3.5 is a viable backbone
at all. Results are kept out of the main Phase 1–3 comparison and should not be
presented alongside them.

## Status (2026-08-08)

**Blocked.** `/datasets/mhoosen/hf_cache/hub/models--stabilityai--stable-diffusion-3.5-large`
contains only config/tokenizer files (~4.9MB), no `.safetensors` weights — same
`fetch_models.slurm` `--include` filter bug as Phase 3. Not started.

## Consumes

| Path | What |
|---|---|
| `pairs/train/` | A small subset of the MITOS paired crops — enough to smoke-test |
| `data/mitos/` | Source frames, if larger samples are needed |

Deliberately minimal. This probe does not consume CAMELYON17, Lizard, PanNuke, or
TCGA-BRCA, and does not run the Phase-2 evaluation battery.

## Produces (aspirational — nothing produced yet)

Not yet run, so no cluster path convention is established. The probe is not expected
to produce weights worth keeping long-term; record whatever path is used here once it
runs.
