# Phase 3 — Architecture Transfer to SDXL

Source of truth: `docs/proposal.tex`, §"Phase 3: Architecture Transfer to SDXL"
(`sec:phase3_sdxl`, `sec:sdxl_a5`).

**Unblocked (2026-08-10)** — Phase 1 (A0–A5) is complete and the decision gate
(P3-01) has been reviewed. Outcome (i): proceed as planned.

---

## P3-01 — Architecture decision gate review
**Status:** ✅ DONE (2026-08-10) — outcome (i): results are clean, SDXL is
feasible (compute confirmed, see P3-02). Phase 3 proceeds as planned.
**Source:** proposal §"Time Plan", "Architecture decision gate" paragraph
**Description:** Formally review SD1.5 A0–A5 results plus SDXL compute-contingency
evidence. Three possible outcomes per the proposal:
  (i) results clean + SDXL feasible → proceed as planned (P3-03);
  (ii) SDXL expected slow → descope to smaller rank / fewer eval slides;
  (iii) Phase 1 reveals a fundamental architecture issue → run deeper SD1.5
       diagnostics instead of transferring; SD1.5 ablation ladder remains the
       primary contribution regardless.
**Review (2026-08-10):**
- **A0–A5 results are clean** — every rung produced valid, verified, non-crashing
  full held-out results (`docs/results/RESULTS_SUMMARY.md`); no fundamental
  architecture issue found, so outcome (iii) does not apply.
- **SDXL compute is confirmed feasible** — SDXL base and SD3.5-large weights are
  both fully cached and verified with real byte sizes (SDXL base 6.94GB,
  SD3.5-large 16.46GB; see CLAUDE.md, corrected from a stale "KNOWN BROKEN" note).
  No compute blocker found, so outcome (ii)'s descope path is not currently needed
  (revisit if actual SDXL training on `bigbatch` turns out slow in practice).
- **Base config recommendation for P3-03: A4** (ControlNet + colour LoRA + LCM-LoRA),
  not A3. The proposal's own abstract frames the contribution as a unified pipeline
  combining all three components with fast inference as a co-equal requirement
  (not optional acceleration on top of a "real" A3 result) — A4 meets the proposal's
  own bar for a clean result (SSIM flat-to-better than the 50-step DDIM reference at
  every scope; the pre-committed 20-step-DDIM fallback exists for structural/artefact
  failure, which did not occur).
- **P3-05 (A5 warm-start transfer) is triggered, not just contingent-possible** —
  `sec:sdxl_a5`'s literal test is measurable A5 vs A4 improvement "beyond the
  inter-run noise floor." P2-05 established that floor at ~0.1–0.3 LAB units
  (rank-change noise). A5 beats A4 on A06 by +2.6 / +5.8 / +8.7 LAB units across
  strengths 0.30/0.40/0.50 — one to two orders of magnitude past the floor. See
  P3-05 below, now unblocked.
- **New Phase 3 blocker, not previously on this list:** `latent-consistency/
  lcm-lora-sdxl` and an SDXL Canny ControlNet checkpoint are not in the cache at
  all (confirmed via cluster search, zero matches) — already flagged under P3-02
  below but restated here since it directly blocks P3-03's LCM/ControlNet legs.

## P3-02 — Fix SDXL model weight download
**Status:** ✅ DONE (2026-08-08) — job 37073 COMPLETED 16:49; verified with real file
checks, not exit code: `models--stabilityai--stable-diffusion-xl-base-1.0/` = 33G,
13 `.safetensors` files present.
**Source:** infrastructure prerequisite for all of Phase 3
**Root cause (confirmed via `--dry-run`, not guessed):** `hf download`'s `--include`
is a single-value option (Click-based CLI: `hf download [OPTIONS] REPO_ID
[FILENAMES]...`). The old `--include "*.safetensors" "*.json" "*.txt" "*.model"`
bound only `*.safetensors` to `--include`; the other three patterns became
positional `FILENAMES`, which switches `hf download` to a code path that ignores
`--include` entirely. Reproduced the exact broken result via dry-run: 17
config/tokenizer files, 3.2M, zero `.safetensors` — matches the actual cache exactly.
**Fix applied:** `slurm/fetch_models.slurm` now uses repeated `--include` flags (the
CLI's own documented syntax), verified via dry-run to actually select the
`.safetensors` files this time.
**Next step:** run `sbatch slurm/fetch_models.slurm` (stampede, CPU-only) and verify
with real file-size checks after, not job exit code.
**Also needed:** `latent-consistency/lcm-lora-sdxl` and an SDXL Canny ControlNet
(e.g. `diffusers/controlnet-canny-sdxl-1.0`) — not yet downloaded at all, not yet
added to `fetch_models.slurm`.

## P3-03 — Transfer best Phase 1 configuration to SDXL
**Status:** TODO — unblocked (P3-01 done). Still needs `lcm-lora-sdxl` + SDXL
Canny ControlNet fetched (see P3-02's "Also needed" note) before it can run.
Recommended base config: **A4** (ControlNet + colour LoRA + LCM-LoRA), per P3-01.
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
**Status:** TODO — UNBLOCKED (2026-08-10). P1-07 resolved: A5 shows measurable
benefit over A4 on SD1.5, well beyond the inter-run noise floor (see P3-01's
review). Still blocked on the same SDXL model-weight prerequisites as P3-03.
**Source:** `sec:sdxl_a5`
**Description:** Only transfer the histopathology warm-start variant to SDXL if A5
first shows measurable benefit over A4 on SD1.5 (beyond the inter-run noise floor).
**Trigger condition MET (2026-08-10):** A5 beats A4 on A06 recovery delta by
+2.6/+5.8/+8.7 LAB units at strengths 0.30/0.40/0.50 (`docs/results/RESULTS_SUMMARY.md`,
`tickets/PHASE1-TICKETS.md` P1-07) — the established noise floor for adapter-level
changes is ~0.1–0.3 LAB units (P2-05, rank 4 vs 8), so this is 10-30x past it, not a
borderline call. Note A5's benefit is slide-dependent (worse than A4 on typical
slides A08/A16 at low strength, though that gap narrows or flips positive at higher
strength per the Phase 1 follow-up experiments) — this nuance should carry over into
how the SDXL transfer is scoped and reported, not just "A5 wins."

---

**Compute note:** proposal states SDXL is compute-contingent — if training time or
memory is excessive on `bigbatch`, descope to smaller rank, fewer eval slides, or
partial transfer, while SD1.5 A0–A5 remains the core contribution regardless of how
Phase 3 goes.
