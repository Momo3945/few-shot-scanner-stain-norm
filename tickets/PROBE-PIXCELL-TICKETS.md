# Auxiliary Probe — PixCell Feasibility (proposed, NOT yet in scope)

**⚠️ This is not in `docs/proposal.tex` at all — unlike the SD3.5 probe (which
has its own dedicated, explicitly descopable proposal section,
`sec:sd35_probe`), PixCell has no proposal citation whatsoever.** Per this
project's own ticket rule ("every ticket cites the proposal section it comes
from... if something doesn't map to the proposal, it should be noted as such
rather than given an ID implying it's proposal-scoped"), this file is
**extracurricular exploration, not graded scope**, until/unless: (a) the
proposal is formally amended to add it (the same way `sec:sd35_probe` was
presumably added), or (b) you decide to run it purely as an informal
side-investigation not claimed as thesis evidence. Do not report anything
from this file as answering the project's main research questions or as
part of the Phase 1–3 success criteria without that decision being made
explicitly first. Ticket IDs below use `PXC-` (not `P#-`) specifically to
avoid implying proposal-scoped status.

**Recommendation, stated up front:** technically feasible as a bounded
probe, same shape as the SD3.5 one (PR-00→PR-05) — but with one materially
bigger unknown than SD3.5 had (see PXC-02 below) and two licensing/access
hurdles to clear first (PXC-00). Worth doing only if you've decided this is
worth the extra time on top of the already-substantial completed Phase 1–3
work and the in-progress SD3.5 probe/H2A follow-ups — not a quick add-on.

---

## What PixCell actually is (verified 2026-09-27, not assumed from the pasted table)

Sources: [paper](https://arxiv.org/abs/2506.05127), [GitHub](https://github.com/cvlab-stonybrook/PixCell), [PixCell-256](https://huggingface.co/StonyBrook-CVLab/PixCell-256), [PixCell-1024](https://huggingface.co/StonyBrook-CVLab/PixCell-1024), [PixCell-256-Cell-ControlNet](https://huggingface.co/StonyBrook-CVLab/PixCell-256-Cell-ControlNet), [UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h).

- **Architecture:** DiT (PixArt-Σ-family transformer), **~0.6B params** —
  notably *smaller* than SD3.5's ~8B transformer, in the same ballpark as
  SD1.5's ~860M UNet. Good news for VRAM/compute (SD3.5's probe needed
  17.77GB peak on a 24GB 3090; PixCell should need considerably less).
- **VAE: confirmed 16-channel, literally `stabilityai/stable-diffusion-3.5-large`'s
  VAE component, loaded separately** — matches the pasted table exactly.
  Not SD1.5/SDXL's 4-channel VAE family at all.
- **Conditioning — the single biggest difference the pasted table
  understates: no text encoder, no text prompt at all.** PixCell conditions
  purely on **UNI2-h SSL image embeddings** (a 681M-param pathology
  foundation vision transformer from MahmoodLab, 1536-dim output) extracted
  from a reference image. There is nothing equivalent to this project's
  fixed prompt `"H&E stained histopathology tissue"` — "what to generate"
  is entirely determined by which image's embedding you feed in.
- **No native img2img/strength dial, confirmed.** Generation is
  embedding-conditioned sampling from noise (DPM-Solver, ~20 steps
  default), not "encode a real image to a latent, add noise at a chosen
  strength, partially denoise" the way SD1.5/SDXL/SD3.5 img2img works here.
  "Virtual staining" (their own closest analog to this project's A→H task,
  demonstrated for H&E→IHC) works by conditioning on the *source* image's
  UNI2-h embedding and generating a *new* image — it is not verified
  whether/how well this preserves exact pixel-level structure the way this
  project's registered-pair SSIM/PSNR/MAE metrics need. **This is the real
  open question — see PXC-02.**
- **LoRA fine-tuning: confirmed supported**, with an existing example for
  a task conceptually close to this project's own (`virtual_staining/
  train_lora.py`, H&E→IHC stain transfer via LoRA + reference embeddings)
  — the closest thing to precedent code for an A→H-style colour-domain LoRA
  on this backbone.
- **ControlNet: one released variant exists** (`PixCell-256-Cell-ControlNet`),
  but it conditions on a **cell-segmentation mask**, not Canny edges or
  HoVer-Net boundaries directly, and it requires the UNI2-h embedding
  **simultaneously** (dual conditioning: mask for spatial layout, embedding
  for content/style) — not a drop-in replacement for this project's
  Canny-ControlNet convention, but real, working proof that a ControlNet
  branch can be trained on this DiT at all (mock training code is in the
  repo) — meaningfully lowers the risk of building a fresh
  source-RGB-conditioned branch (the same idea P1-10/P3-06/P3-07 each
  built from scratch for their own backbone), since there's now a working
  reference implementation to fork from instead of zero precedent.
- **No LCM-equivalent distillation** — confirmed, matches the pasted
  table. Same caveat this project's own SD3.5 probe ticket already states
  for that model: does not answer the LCM-LoRA research question. Default
  20-step DPM-Solver is already fast on a 0.6B model, partially offsetting
  this.
- **Resolutions released: 256×256 and 1024×1024 only** — no 512, unlike
  the pasted table's framing this matters for consistency with this
  project's existing 512 crop grid (P3-06/P1-10 etc.) — would need to
  either downsample 1024 outputs or accept a resolution mismatch when
  comparing against existing results.
- **Diffusers integration is via a *custom remote-code* pipeline**
  (`trust_remote_code=True`, `custom_pipeline="StonyBrook-CVLab/PixCell-pipeline"`),
  not a class in mainline `diffusers` — same general risk category as any
  community pipeline (less battle-tested, needs the pipeline's Python file
  fetched from the Hub at least once, which matters for this cluster's
  usual `HF_HUB_OFFLINE=1` convention after first fetch).
- **Licensing — a real, concrete difference from every other backbone in
  this project (SD1.5/SDXL: permissive Apache/OpenRAIL-ish; SD3.5: gated
  but standard commercial-ish EULA):**
  - PixCell model weights: **CC BY-NC-ND 4.0** ("No Derivatives") —
    non-commercial academic use is clearly fine for a thesis, but the "ND"
    clause is a real question mark around publishing/redistributing a
    *fine-tuned* checkpoint publicly (e.g. a GitHub release or public HF
    upload attached to the thesis). Keeping any fine-tuned checkpoint
    private/local for internal evaluation only avoids the ambiguity;
    worth a one-line disclosure in the thesis either way, and worth
    checking with your supervisor before publishing any derivative weights.
  - The PixCell *code* repo itself is separately licensed CC BY-NC 4.0
    (derivatives permitted, non-commercial) — the code/training-script
    license is fine; it's specifically the released *weights* license that
    carries the ND restriction.
  - UNI2-h (the embedding model PixCell depends on for every image, both
    training-data prep and generation) is **also CC BY-NC-ND 4.0**, same
    caveat.
- **Access: PixCell's own weights are NOT gated** (confirmed — no
  "agree to access" banner on either HF repo) — but **UNI2-h IS gated, and
  the request specifically requires an institutional email** (a personal
  Gmail-type address is auto-denied per MahmoodLab's own stated policy) —
  a real prerequisite blocker, same shape as PR-00's SD3.5 gated-repo
  hurdle, needing manual account-level action before any code can run.

---

## PXC-00 — Access, licensing disclosure, and model fetch

**Status:** ✅ DONE (2026-09-27, job 61028). All three models fetched and
verified with real byte sizes, not exit codes (`slurm/fetch_pixcell_models.slurm`):
`StonyBrook-CVLab/PixCell-256` (2.3G), `StonyBrook-CVLab/PixCell-1024`
(2.3G), `MahmoodLab/UNI2-h` (2.6G) — all under
`/datasets/mhoosen/hf_cache/hub/`. **UNI2-h succeeded on the first attempt,
no institutional-email gating hit encountered** — a valid HF token already
existed at `/datasets/mhoosen/hf_cache/token` (dated 2026-08-21, from
earlier SD3.5-probe gated-access work, per `PROBE-SD35-TICKETS.md` PR-00)
and evidently already carries UNI2-h approval. (An earlier `hf auth
whoami` check that session reported "Not logged in" — that was checking
the *default* `~/.cache/huggingface/token` location without `HF_HOME`
exported; the real token lives under this project's canonical
`HF_HOME=/datasets/mhoosen/hf_cache`, which every actual download script
already exports correctly — a false alarm, not a real gap.) Licensing
caveat (CC BY-NC-ND on both PixCell and UNI2-h) still applies as documented
above regardless of access already working — non-commercial thesis use is
fine, don't publish derivative weights without checking with your
supervisor first.

**Description:**
1. Confirm you have (or can register) an institutional email eligible for
   UNI2-h's gated access — request access at
   `https://huggingface.co/MahmoodLab/UNI2-h` with that email set as your
   HF account's primary email (this is an account-level action, same as
   PR-00's SD3.5 approval — cannot be scripted).
2. Once approved, fetch (via `sbatch`, per this project's standing rule —
   never a bare-`ssh` download): `MahmoodLab/UNI2-h`,
   `StonyBrook-CVLab/PixCell-1024` (or `-256` for a cheaper first probe),
   and — if PXC-02 below looks promising — `StonyBrook-CVLab/PixCell-256-Cell-ControlNet`
   for reference. Verify with real byte sizes, not exit codes, per this
   project's own hard-won `hf cache scan` discipline.
3. Note the CC BY-NC-ND terms (both PixCell and UNI2-h) explicitly in the
   thesis if this probe is reported at all — non-commercial academic use
   is fine, but flag it rather than silently omit it, and don't publish
   any fine-tuned PixCell/UNI2-h-derived checkpoint publicly without
   checking the ND clause with your supervisor first.

## PXC-01 — LoRA training time + peak VRAM measurement (mirrors PR-01)

**Status:** 🔄 IN PROGRESS — PXC-01a (wiring probe) ✅ DONE (2026-09-27,
job 61121, `src/eval/probe_pixcell_pipeline.py`), PXC-01b (real LoRA
training) not yet started.

**PXC-01a result — the documented API works, with two real bugs found
and fixed en route (both via direct primary-source reads, not guessing):**
1. `timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True)` with no
   extra kwargs crashes inside timm's own position-embedding resampler
   (`shape '[1, 15, 15, -1]' is invalid for input of size 391680`) — this
   cluster's timm (1.0.30) does not auto-populate UNI2-h's required
   architecture kwargs from `hf-hub:` loading alone. Fixed by passing
   UNI2-h's own documented `timm_kwargs` explicitly (`reg_tokens=8`,
   `dynamic_img_size=True`, `mlp_layer=timm.layers.SwiGLUPacked`,
   `act_layer=torch.nn.SiLU`, plus the standard ViT-H/14 dims) — confirmed
   correct via UNI2-h's own HF README, not trial-and-error.
2. `guidance_scale=1.5` (PixCell's own documented default) requires an
   explicit `negative_uni_embeds` — the pipeline's own `check_inputs()`
   refuses to silently default one. Fixed via the pipeline's own
   documented `get_unconditional_embedding()` helper (confirmed by
   directly reading the cached pipeline source on the cluster, not
   guessing a zero-tensor substitute).

**Resolves this project's real unknowns going in:** UNI2-h embedding for a
single 224×224 image is `(1, 1536)`, reshaped to `(1, 1, 1536)` — confirmed
to match `caption_num_tokens=1` read directly from the loaded transformer's
own config (not assumed). Full pipeline (UNI2-h → PixCell-256, 20 steps,
guidance 1.5) completes in **3.34s**, peak VRAM **4.03 GB** — dramatically
cheaper than SD3.5's own probe result (17.77 GB), confirming the
cheap-compute expectation stated below.

**Qualitative sanity check (same-domain, zero fine-tuning — conditioning
on the source Aperio crop's own embedding, no A→H translation intent
yet):** the generated 256×256 output visually preserves the source
crop's tissue architecture, gland/nuclear positions, and overall colour
strikingly well for a purely embedding-conditioned generation — more
structure-preserving than expected going in for a "lossy 1536-dim
summary." Purely qualitative; PXC-02 Part B (below) is the real
quantitative measurement against a registered target.

**Description:** Train an A→H LoRA on the same ≤50 A03/H03 crop pairs,
using UNI2-h embeddings extracted from the *source* (Aperio) crop as the
conditioning input, target = the Hamamatsu crop, following the
`virtual_staining/train_lora.py` pattern as the starting point (adapt, not
copy verbatim — that script targets IHC-from-H&E, not scanner colour).
Record wall-clock + peak VRAM, same measurement discipline as PR-01.
**Expectation, not yet verified:** given PixCell's 0.6B params (smaller
than SD3.5's 8B), this should be markedly cheaper than SD3.5's 17.77GB/
13.3min-per-1000-steps result — a genuine reason for optimism on pure
compute cost, independent of the harder question below.

## PXC-02 — Structural-fidelity floor check (the real open question — do this before anything else past PXC-01)

**Status:** 🔄 IN PROGRESS — Part A done (2026-09-27, job 61089), Part B
blocked on PXC-01a's wiring probe.

**Part A result — PixCell's VAE-only self-reconstruction floor is
dramatically higher than every other backbone in this project's own
VAE floor:** extended `src/eval/benchmark_vae_reconstruction.py` with a
`sd35_pixcell` VAE entry (PixCell's own VAE, verbatim
`stabilityai/stable-diffusion-3.5-large`, bf16 — additive, `stock`/`ft_mse`
untouched), ran `--stage heldout --limit 2` (8 real held-out Aperio crops,
A06 slide, zero denoising, deterministic `.mode()` encode):

| VAE | Self-reconstruction SSIM |
|---|---|
| SD1.5 stock (P1-10's own floor) | 0.5393 |
| SD1.5 `sd-vae-ft-mse` (P1-14) | ~0.618 (pooled, Stage B) |
| **SD3.5 / PixCell's VAE (this check)** | **0.7732** |

This is the single most encouraging PixCell number so far: the 4-channel
VAE ceiling that has capped *every* SD1.5/SDXL structural-fidelity result
in this project (P1-10 established 0.5393 as a hard architectural floor
no post-hoc fix could exceed) simply may not apply the same way here — a
16-channel latent reconstructs real histopathology crops far more
faithfully to begin with. This is necessary, not sufficient: it says
nothing yet about whether embedding-conditioned *generation* (Part B, not
just passive encode/decode) can get anywhere near this new, higher
ceiling — that is exactly what Part B measures next.

**Part B result — decisive, and negative: embedding-only conditioning does
not preserve enough structure for this project's registered-pair metrics.**
Ran `src/eval/floor_check_pixcell.py` (job 61123) — 8 real held-out crops
(A06, 256×256, zero fine-tuning), generated from each crop's own UNI2-h
embedding, scored against the real registered Hamamatsu target:

| Comparison | SSIM | LAB total |
|---|---|---|
| vs. real registered Hamamatsu target | **0.0451** | 78.22 |
| vs. raw Aperio source (sanity) | **0.0363** | 15.94 |

**Both SSIM values are noise-level** — far below even SD1.5's own
already-low img2img floor (0.27–0.46), and nowhere near the 0.7732 VAE-only
ceiling Part A just measured for this same backbone. The generation is not
even closer to its own source (0.0363) than to the target (0.0451) —
consistent with a single 1536-dim global embedding carrying no
pixel/nucleus-position information at all, only a coarse semantic/style
summary. **This matches this ticket's own pre-stated decisive criterion
exactly** ("if this comes back catastrophically low... that's a decisive,
cheap answer the embedding-only paradigm doesn't fit this project's
evaluation methodology at all"). The earlier PXC-01a qualitative
single-crop check *looked* structurally similar by eye (generic
"pink H&E tissue, similar density" resemblance) — this quantitative,
registered-pair result shows that visual impression was not real
pixel-level structural correspondence.

**Consequence for PXC-03:** a source-conditioning ControlNet build (feeding
the actual source RGB into the denoising process, not just its embedding)
is now a *precondition* for any usable structural number from this
backbone, not an optional refinement — exactly the scope decision this
ticket flagged in advance. This is the real go/no-go point for the whole
probe: PXC-03 is a genuinely large new build (a first-of-its-kind
source-conditioning branch for a DiT, this project's third such build
after SD1.5/SDXL) — worth an explicit decision before starting, not an
automatic next step.

**Description:** Before investing in any source-conditioning ControlNet
build, check the cheapest possible thing first, mirroring P1-10's own
"VAE-only floor" methodology and P1-14's VAE-swap benchmark: **encode a
held-out registered crop through PixCell's VAE (= SD3.5's VAE) and decode
it immediately with zero denoising** — what SSIM ceiling does that alone
impose? Then, separately, run PixCell's actual embedding-conditioned
generation (condition on the source crop's own UNI2-h embedding, generate
fresh, no source-RGB path at all) and measure SSIM/PSNR/MAE against the
registered target with **zero fine-tuning** — this tells you, before
building any custom machinery, whether embedding-only conditioning is even
in the right ballpark for this project's pixel-exact metrics, or whether
it structurally cannot preserve nucleus-level positions the way img2img
does (a real possibility, since a 1536-dim embedding is a lossy
content summary, not the source pixels). **If this comes back catastrophically
low** (materially worse than SD1.5's own already-low img2img SSIM floor,
~0.27–0.46 range), that's a decisive, cheap answer that the embedding-only
paradigm doesn't fit this project's evaluation methodology at all, and the
source-conditioning ControlNet build (PXC-03) would need to happen before
any usable number exists — worth knowing before, not after, that build.

## PXC-03 — Source-conditioning ControlNet build (only if PXC-02 motivates it)

**Status:** BLOCKED on PXC-02's result.
**Description:** If PXC-02 shows embedding-only conditioning can't hold
structure, build a fresh source-RGB-conditioned branch for PixCell's DiT —
conceptually the same idea as P1-10 (SD1.5) and P3-06/P3-07 (SDXL), but
this would be the **third from-scratch implementation of that idea, for a
third backbone family (DiT, not UNet)** — real new engineering, not a
port. `PixCell-256-Cell-ControlNet`'s existence (and its repo's mock
training code) is a working reference to fork from, which meaningfully
de-risks this vs. building from zero, but it's still the single largest
scope item in this whole probe. Canny/HoVer-Net boundary maps already
generated for P2-08 are directly reusable as the condition image, per the
pasted table's own (correct) observation.

## PXC-04 — Report outcome

**Status:** BLOCKED on PXC-01/02(/03).
**Description:** Same discipline as PR-05 — write up feasible/infeasible
as a short, clearly-scoped note. Given PXC-00's proposal-scope caveat
above, this explicitly **cannot** be framed as a fourth ablation condition
or as evidence for/against the main research questions unless the
proposal itself is amended first — report it as informal exploratory
follow-up work at most, distinct from the graded Phase 1–3 results.
