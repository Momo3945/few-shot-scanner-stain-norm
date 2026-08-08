# Tickets — index

Task breakdown for the stain-normalisation project, derived directly from
`docs/proposal.tex` (the graded research proposal — the source of truth for scope).
Each ticket cites the proposal section/label it comes from, so scope stays
traceable back to the graded document rather than drifting over time.

## Files
- `PHASE1-TICKETS.md` — SD1.5 ablation ladder A0–A5 (training)
- `PHASE2-TICKETS.md` — evaluation (colour, structural, clinical, cycle, CAMELYON17)
- `PHASE3-TICKETS.md` — SDXL portability transfer (blocked on Phase 1 decision gate)
- `PROBE-SD35-TICKETS.md` — auxiliary SD3.5 feasibility probe (lowest priority, descopable)

## Status legend
- `TODO` — not started
- `🔄 IN PROGRESS` — actively running or partially done
- `✅ DONE` — complete, with evidence cited (Slurm job ID, file path, or both)
- `BLOCKED` — cannot start; blocking dependency named explicitly

## Ticket ID scheme
`P<phase>-<number><letter>`, e.g. `P1-03a`. Letters (a/b/c/d) are used where the
proposal defines a small matrix of the same task (e.g. rank × direction for the
colour LoRA) rather than genuinely separate work items.

## Rules for keeping this current
1. Never mark a ticket DONE from a job *submission* alone — a job leaving `squeue`
   is not proof of success (see CLAUDE.md). Verify via the job's `.out` log or the
   actual output file before flipping status.
2. Every new ticket must cite a proposal section/label. If something doesn't map to
   the proposal, it may be legitimate infrastructure work (fine) but should be noted
   as such rather than given a `P#-##` ID implying it's proposal-scoped.
3. When a ticket's status changes, update it in place — don't create a duplicate.
4. This index and the four ticket files are meant to be read by both Muhammad and
   Claude Code sessions as the current task board. Keep it accurate rather than
   aspirational.
