# Phase 3 — Architecture Transfer to SDXL

Source of truth: `docs/proposal.tex`, §"Phase 3: Architecture Transfer to SDXL"
(`sec:phase3_sdxl`, `sec:sdxl_a5`).

**Blocked overall** on the Phase 1 decision gate — do not start these until Phase 1's
A0–A5 results are in and reviewed (see PHASE1-TICKETS.md footer).

---

## P3-01 — Architecture decision gate review
**Status:** TODO — blocked on Phase 1 completion
**Source:** proposal §"Time Plan", "Architecture decision gate" paragraph
**Description:** Formally review SD1.5 A0–A5 results plus SDXL compute-contingency
evidence. Three possible outcomes per the proposal:
  (i) results clean + SDXL feasible → proceed as planned (P3-03);
  (ii) SDXL expected slow → descope to smaller rank / fewer eval slides;
  (iii) Phase 1 reveals a fundamental architecture issue → run deeper SD1.5
       diagnostics instead of transferring; SD1.5 ablation ladder remains the
       primary contribution regardless.

## P3-02 — Fix SDXL model weight download
**Status:** TODO — BLOCKING; known bug
**Source:** infrastructure prerequisite for all of Phase 3
**Description:** `hf_cache` currently holds only config/tokenizer files for
`stabilityai/stable-diffusion-xl-base-1.0` (~1.6 MB) — no `.safetensors` weights,
despite `fetch_models.slurm` logging "✓ Downloaded" and exiting 0. The `--include`
filter pattern used in that job does not match SDXL's actual sharded filenames.
**Fix:** re-run `hf download stabilityai/stable-diffusion-xl-base-1.0` WITHOUT the
`--include` filter (disk is not a constraint — 181 TB+ free on `/datasets`). Verify
with `hf cache scan` or explicit file-size check, not just job exit code, before
trusting it.
**Also needed:** `latent-consistency/lcm-lora-sdxl` and an SDXL Canny ControlNet
(e.g. `diffusers/controlnet-canny-sdxl-1.0`) — not yet downloaded at all.

## P3-03 — Transfer best Phase 1 configuration to SDXL
**Status:** TODO — blocked on P3-01, P3-02
**Source:** `sec:phase3_sdxl`
**Description:** Transfer ONLY the best-performing SD1.5 configuration (from A0–A5).
The proposal is explicit: **do not** repeat the full ablation ladder on SDXL — that
would multiply compute cost without proportional scientific value, since Phase 1
already isolates each component's contribution.
**Tooling:** same diffusers training ecosystem; `lcm-lora-sdxl` for 2–8 step inference.

## P3-04 — SDXL vs SD1.5 comparison
**Status:** TODO — blocked on P3-03
**Source:** `sec:phase3_sdxl`
**Description:** Compare the transferred SDXL configuration against its SD1.5
counterpart using the same colour, structure, and speed metrics from Phase 2's
harness (reuse `score_outputs.py`).

## P3-05 — A5 warm-start variant on SDXL (contingent)
**Status:** TODO — CONTINGENT, do not start until P1-07 (A5 vs A4 on SD1.5) resolves
**Source:** `sec:sdxl_a5`
**Description:** Only transfer the histopathology warm-start variant to SDXL if A5
first shows measurable benefit over A4 on SD1.5 (beyond the inter-run noise floor).
If A5 does NOT improve on A4, SDXL transfer uses the vanilla SDXL base only, and the
negative A5 result is reported as evidence that ControlNet + colour LoRA is sufficient
without a domain warm-start — this is a valid, reportable outcome either way.

---

**Compute note:** proposal states SDXL is compute-contingent — if training time or
memory is excessive on `bigbatch`, descope to smaller rank, fewer eval slides, or
partial transfer, while SD1.5 A0–A5 remains the core contribution regardless of how
Phase 3 goes.
