# P1-15 — P1-11 with the Winning Alternate Decoder

**What this tests:** keeps P1-11's DDIM inversion, conditioning, and
translated latent completely unchanged, and swaps only the final decoding
stage to P1-14's winning VAE (`sd-vae-ft-mse`) — a decoder-only isolation,
so any SSIM change is attributable to the decoder alone. Gated on P1-14
finding a meaningful reconstruction gain (it has, Stage A).

**Status:** ⚠️ **Ahead of what's written up.** `RESULTS_SUMMARY.md` and
`tickets/PHASE1-TICKETS.md` still list this as "blocked, not started," but
smoke-test data (sanity check + correct/shuffled source-mode arms) now
exists on the cluster (pulled below, dated 2026-08-27/28). **No numbers
from this folder should be cited or presented until this is properly
reviewed and written up** — this README intentionally states no results.

**Files:** `eval/sanity_check/`, `eval/smoke_correct/`,
`eval/smoke_correct_summary/`, `eval/smoke_shuffled/`,
`eval/smoke_all_summary/` (`eval_manifest.csv`, `per_crop.csv`,
`summary.csv`, `paired_win_rate.csv`, `run_metadata.json` per arm).

**Next step:** review this data, write up Stage-appropriate findings in
`RESULTS_SUMMARY.md`, then update this README to match — flagged to the
user as a follow-up, not done as part of this reorg.

**Full narrative:** `tickets/P1-15_p1_11_alternate_decoder.md`,
`tickets/PHASE1-TICKETS.md` P1-15.
