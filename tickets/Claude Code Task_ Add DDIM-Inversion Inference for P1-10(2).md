# Task: Add a DDIM-Inversion Inference Path for P1-10

We have already trained and validated the **P1-10 source-conditioned A→H model**. Do **not retrain anything** and do **not modify the existing validated inference path**. I want a new inference implementation that replaces the current random img2img corruption/strength initialization with **DDIM inversion of the actual source image**, while preserving the existing source-conditioning mechanism and model checkpoint.

The purpose of this experiment is to test whether DDIM inversion preserves more source structure than the current:

[\
\text{VAE encode} \rightarrow \text{add random noise at strength} \rightarrow \text{DDIM denoise}\
]

inference path.

The new path should instead perform:

[\
\text{VAE encode source}\
\rightarrow\
\text{DDIM inversion}\
\rightarrow\
\text{source-specific noisy latent}\
\rightarrow\
\text{conditional DDIM translation}\
\rightarrow\
\text{VAE decode}\
]

This is **inference-only**. The already-trained P1-10 weights must be used exactly as-is.

---

# 1. FIRST: inspect the existing implementation

Before writing code, inspect the repository and identify:

1. The current P1-10 inference script used for the successful full held-out evaluation.
2. The P1-10 training script.
3. How the source-conditioning branch is constructed.
4. How the trained checkpoint `lora/a2h_cond_r8/best` is loaded.
5. The current scheduler.
6. Whether P1-10 was trained with:
   - epsilon prediction, or
   - v-prediction.
7. The exact VAE scaling convention.
8. How source-condition tensors are produced.
9. How the source-conditioning residuals are injected into the frozen U-Net.
10. How seeds are handled.
11. How the current `strength=0.50` path selects the starting timestep.
12. How output filenames and evaluation metadata are currently written.

Do not assume any of these details from memory.

**The existing P1-10 implementation is the source of truth.**

The new script must remain compatible with the exact trained model.

---

# 2. Reference implementation

Use the official HistDiST implementation as the conceptual reference for DDIM inversion:

- repository: `ErikGro/HistDiST`
- file: `inference/inference.py`

Important: do **not** blindly copy its scheduler settings.

HistDiST uses configuration choices such as v-prediction, trailing timesteps and zero-terminal-SNR because its model was trained accordingly.

Our P1-10 model must use **the same prediction formulation and scheduler assumptions it was trained with**.

What should be borrowed from HistDiST is the high-level procedure:

```text
source image
    ↓
VAE encode
    ↓
source latent z_A
    ↓
DDIM inverse scheduler
    ↓
source-specific inverted latent z_t
    ↓
switch to normal DDIM scheduler
    ↓
conditional reverse diffusion
    ↓
decoded translated image
```

HistDiST also keeps the clean source conditioning available during the forward translation pass. Our P1-10 source-conditioning branch must likewise remain active exactly as in the successful existing `correct-source` inference mode.

---

# 3. DO NOT replace or edit the existing inference script

Create a new script.

Suggested name:

```text
src/eval/infer_colour_source_ddim_inversion.py
```

If the repository has a more appropriate naming convention, follow it.

The original P1-10 inference script must continue to function unchanged.

I need to be able to run both:

```text
old random-noise/img2img inference
```

and

```text
new DDIM-inversion inference
```

for direct comparison.

---

# 4. The current inference path we are replacing

The existing P1-10 inference is approximately:

[\
A\
\xrightarrow{VAE}\
z\_A\
]

then:

[\
z\_t=\
\sqrt{\bar{\alpha}\_t}z\_A+\
\sqrt{1-\bar{\alpha}\_t}\epsilon\
]

with:

[\
\epsilon\sim\mathcal{N}(0,I)\
]

and the timestep determined by `strength=0.50`.

Then:

[\
z\_t\
\xrightarrow[\text{correct source conditioning}]{DDIM}\
\hat z\_H\
]

followed by VAE decode.

The important problem is that the noise added to `z_A` is arbitrary Gaussian noise.

The new script should **not call the current random-noise initialization path** when inversion mode is enabled.

---

# 5. New full-inversion inference path

Implement this procedure.

## Step A — load source

Load the Aperio source patch exactly as the existing P1-10 inference script does.

Use identical:

- resizing,
- normalization,
- tensor shape,
- dtype,
- device handling.

No new preprocessing.

---

## Step B — VAE encode source

Encode the source using the exact VAE already associated with P1-10.

Conceptually:

[\
z\_A=E(A)\
]

Be extremely careful about the SD1.5 latent scale factor.

The script must verify the existing implementation's convention rather than assuming whether scaling occurs before or after storing the source conditioning latent.

The inversion latent needs to use the scale expected by the scheduler/U-Net.

---

## Step C — construct the source conditioning

Construct the **exact same ****`correct-source`**** condition** that passed the P1-10 smoke test and full evaluation.

Do not introduce a new source encoder.

Do not replace the trained ControlNet/source branch.

Do not change conditioning scale unless necessary for compatibility.

The only intended experimental variable is:

```text
random img2img initialization
            ↓
DDIM inversion initialization
```

Everything else should remain fixed.

---

# 6. DDIM inversion

Use Diffusers' `DDIMInverseScheduler` where compatible with the installed Diffusers version and existing scheduler configuration.

The inverse scheduler must inherit/match the relevant settings from the existing DDIM scheduler/model.

At minimum verify:

```text
num_train_timesteps
beta_start
beta_end
beta_schedule
prediction_type
clip_sample
set_alpha_to_one
steps_offset
timestep_spacing
rescale_betas_zero_snr
```

Do not blindly hard-code HistDiST values.

Build the inverse scheduler from the existing scheduler config wherever possible.

For example, conceptually:

```python
inverse_scheduler = DDIMInverseScheduler.from_config(
    existing_scheduler.config
)
```

but confirm this works correctly with the installed library version and P1-10 configuration.

---

# 7. Important: what the model should see during inversion

For the first implementation, inversion should answer the simplest possible question:

> What diffusion trajectory corresponds to this exact Aperio source image?

Therefore inversion should reconstruct the **source domain**, not perform the A→H transformation during the inverse pass.

Do not accidentally apply the A→H stain transformation while moving from:

[\
z\_A \rightarrow z\_t.\
]

The source conditioning used during inversion must be selected deliberately.

Implement an explicit CLI option so we can test inversion conditioning modes rather than hiding the choice.

For example:

```text
--inversion-condition none
--inversion-condition source
```

Default initially to the most conservative identity-preserving choice based on the existing model architecture.

Document exactly what each mode means.

If `none` means zeroing the source conditioning branch during inversion, make that explicit.

If model conditioning cannot simply be disabled because of the trained architecture, reproduce the appropriate source-domain/identity conditioning behavior.

Do not silently guess.

---

# 8. Perform inversion iteratively

Starting from:

[\
z\_0=z\_A,\
]

run DDIM inversion through the selected inversion timesteps:

[\
z\_0\
\rightarrow\
z\_1\
\rightarrow\
z\_2\
\rightarrow\
\cdots\
\rightarrow\
z\_T.\
]

The implementation should follow the inverse scheduler's mathematically correct stepping direction.

Do not simulate inversion by merely calling:

```python
scheduler.add_noise(...)
```

That would defeat the experiment.

The resulting latent must have been obtained through iterative model-guided inversion.

---

# 9. Support PARTIAL inversion

Do not implement only a fixed full inversion.

I need both:

```text
full inversion
```

and

```text
partial inversion
```

because partial inversion is the inversion equivalent of img2img strength and may yield a better colour/structure trade-off.

Add a CLI parameter such as:

```text
--inversion-fraction
```

where:

```text
1.00 = full inversion
0.80 = invert through 80% of trajectory
0.60 = 60%
0.40 = 40%
0.20 = 20%
```

Alternatively use:

```text
--inversion-start-step
```

if that maps more naturally onto Diffusers.

Prefer the fraction interface for usability, but internally map it cleanly onto scheduler timesteps.

Validate bounds:

[\
0 < f \leq 1.\
]

Log both:

```text
requested inversion fraction
actual selected timestep
actual number of inverse steps
```

because discretization may mean they do not map perfectly.

---

# 10. Translation / reconstruction pass

Once the inverted latent has been obtained, switch to a **normal DDIM scheduler** configured consistently with the inverse scheduler and the P1-10 training setup.

Start reverse denoising from the corresponding timestep.

During translation:

- use the trained A→H P1-10 weights,
- enable the normal **correct source conditioning**,
- use the original clean Aperio source condition at every step exactly as the existing inference path does,
- preserve all existing conditioning scales,
- preserve prompt/text behavior,
- preserve guidance configuration,
- preserve deterministic seed handling where applicable.

Conceptually:

[\
z\_T^A\
\xrightarrow[\text{source}=A]{P1-10}\
z\_H\
]

then:

[\
\hat H=D(z\_H).\
]

---

# 11. Do NOT use `strength` in inversion mode

The existing:

```text
--strength 0.50
```

parameter belongs to random img2img corruption.

DDIM inversion should have its own parameter:

```text
--inversion-fraction
```

or equivalent.

Do not internally convert inversion back into `add_noise(strength)`.

If maintaining CLI compatibility is useful, allow `--strength` to exist but:

- reject it in inversion mode, or
- clearly ignore it with a warning.

I prefer an explicit error if both are passed:

```text
ERROR: --strength is not used with DDIM inversion.
Use --inversion-fraction instead.
```

---

# 12. Number of steps

Default initially to:

```text
--inversion-steps 50
--translation-steps 50
```

because the existing P1-10 full held-out evaluation used 50-step DDIM.

Expose them separately:

```text
--inversion-steps
--translation-steps
```

This allows us later to test:

```text
50 → 50
100 → 50
100 → 100
```

without code changes.

Do not assume the inverse and forward counts must always be equal.

---

# 13. Identity reconstruction mode — MANDATORY

Before using inversion for A→H translation, the new script must support an **identity reconstruction test**.

Add something like:

```text
--mode identity
```

and:

```text
--mode translate
```

### `identity` mode

Do:

[\
A\
\rightarrow\
VAE\
\rightarrow\
DDIM\ inversion\
\rightarrow\
DDIM\ reconstruction\
\rightarrow\
A'\
]

with **no A→H stain transformation**.

The purpose is to determine how much structure is lost purely by the inversion/reconstruction path.

The exact source-domain/identity conditioning setup must be chosen consistently with the model architecture.

Save:

```text
source
reconstruction
```

and calculate or make them available for:

```text
SSIM(source, reconstruction)
PSNR(source, reconstruction)
MAE(source, reconstruction)
```

This is the first mandatory sanity test.

---

# 14. Baseline identity comparison — MANDATORY

The identity experiment needs a direct baseline against the old random-corruption path.

For the exact same source crops, seeds and DDIM step count, produce:

### Existing pathway

[\
A\
\rightarrow\
VAE\
\rightarrow\
\text{random noise at strength 0.50}\
\rightarrow\
DDIM\
\rightarrow\
A'\_{\text{img2img}}\
]

### New pathway

[\
A\
\rightarrow\
VAE\
\rightarrow\
DDIM^{-1}\
\rightarrow\
DDIM\
\rightarrow\
A'\_{\text{inv}}\
]

Score:

```text
SSIM
PSNR
MAE
```

between each reconstruction and the original source.

The primary identity question is:

[\
SSIM(A,A'\_{\text{inv}})

>

SSIM(A,A'\_{\text{img2img}})?\
]

Do not proceed to making scientific claims from translation results until this identity check has been produced.

---

# 15. Keep the source-ablation controls available

The existing P1-10 test used:

```text
correct source
shuffled source
zero source
```

Preserve those modes in the new inference script if possible:

```text
--source-mode correct
--source-mode shuffled
--source-mode zero
```

They should apply during the **translation** pass.

This is important because if inversion dramatically increases SSIM while:

```text
correct ≈ shuffled
```

then inversion may simply be reconstructing the source and bypassing the learned conditional mapping.

We still need:

[\
\text{correct}>\text{shuffled}\
]

to show the learned A→H source relationship remains causally useful.

---

# 16. Seed handling

Use deterministic seeds exactly as the current held-out evaluation does.

Important distinction:

DDIM inversion itself should ideally be deterministic when:

```text
eta = 0
```

and the scheduler/configuration permits it.

Translation should also default to deterministic DDIM:

```text
eta = 0
```

unless the current P1-10 implementation explicitly requires otherwise.

Still retain the existing three-seed evaluation framework so comparisons remain compatible with prior results.

Log seeds in metadata.

---

# 17. Preserve all existing output/evaluation conventions

Outputs must remain compatible with the current evaluation tooling.

Do not create a completely different directory structure if that would require rewriting all scoring code.

Use something like:

```text
eval/p1_10_ddim_inversion/
```

with subdirectories/metadata matching existing P1-10 output conventions.

Suggested separation:

```text
eval/p1_10_ddim_inversion/
    identity/
    translate/
        inv020/
        inv040/
        inv060/
        inv080/
        inv100/
```

or whatever is most compatible with the current scorer.

Do not overwrite:

```text
eval/p1_10_full_heldout
```

or any prior experiment.

---

# 18. Metadata

For every run save enough metadata to reproduce it exactly.

At minimum:

```text
checkpoint
direction
source mode
inversion conditioning mode
prediction type
scheduler config
VAE
inversion steps
translation steps
inversion fraction
actual starting timestep
guidance scale
conditioning scale
seed
dtype
model revision/path
git commit if available
```

Also explicitly record:

```text
initialization = ddim_inversion
```

so it cannot be confused with old img2img output.

---

# 19. Add diagnostics

For debugging, add optional logging:

```text
--debug-inversion
```

When enabled, print/save:

1. Source latent mean/std.
2. Inverted latent mean/std.
3. Reconstruction latent mean/std.
4. Timesteps used for inversion.
5. Timesteps used for translation.
6. First and last selected timestep.
7. Whether source conditioning is enabled during inversion.
8. Source conditioning mode during translation.
9. Scheduler prediction type.
10. VAE scaling factor.
11. NaN/Inf checks.

Do not dump huge tensors.

---

# 20. Optional trajectory saving

Add:

```text
--save-inversion-trajectory
```

When enabled, save a small number of decoded snapshots, e.g.:

```text
0%
25%
50%
75%
100%
```

during inversion and translation.

This is only for debugging and visual confirmation.

Do not enable by default because it will slow evaluation and consume disk space.

---

# 21. Important scheduler correctness checks

Before trusting results, add assertions/checks that:

### Prediction type matches

If training used:

```text
epsilon
```

then inversion and translation must use:

```text
epsilon
```

unless there is a mathematically justified conversion.

Do not switch to HistDiST's `v_prediction`.

### Scheduler parameters match

The inverse and forward schedulers should describe the same diffusion process.

### Timesteps align

For partial inversion:

if the inverse pass ends at timestep (t\_k),

the translation pass must start from the corresponding (t\_k).

Do not invert to one timestep and start reconstruction from an unrelated timestep.

### Scaling matches

Verify whether the VAE latent is multiplied by:

```text
vae.config.scaling_factor
```

before inversion.

Use the same latent convention throughout.

---

# 22. DO NOT introduce these changes

This experiment is intended to isolate **inference initialization**.

Therefore do not:

- retrain P1-10,
- change LoRA rank,
- change the trained source branch,
- change the VAE,
- add the MSE VAE,
- add Consistency Decoder,
- add ControlNet variants,
- add reconstruction losses,
- change source preprocessing,
- change target preprocessing,
- change evaluation registration,
- change prompts,
- change the training dataset,
- change checkpoint selection,
- change colour post-processing,
- add residual colour fusion.

Those are separate future experiments.

For this ticket:

[\
\boxed{\
\text{ONLY change random-noise initialization to DDIM inversion}\
}\
]

as far as technically possible.

---

# 23. Smoke-test procedure

Do not immediately run all 496 crops.

Start with a small deterministic smoke test.

Use the same A06 + A08 type of subset used for previous P1-10 gating if possible.

First run:

### Test 1 — VAE-only identity

Already exists conceptually, but verify compatibility.

[\
A\rightarrow VAE\rightarrow A'\
]

### Test 2 — random img2img identity

[\
A\rightarrow strength=0.50\rightarrow A'\
]

### Test 3 — full inversion identity

[\
A\rightarrow inversion(1.0)\rightarrow reconstruction\rightarrow A'\
]

### Test 4 — partial inversion identity

At minimum:

```text
0.25
0.50
0.75
1.00
```

Then report:

```text
SSIM
PSNR
MAE
```

for each.

---

# 24. Translation smoke test

If inversion identity is working correctly, run P1-10 translation on the same small subset.

Test:

```text
inversion fraction = 0.25
inversion fraction = 0.50
inversion fraction = 0.75
inversion fraction = 1.00
```

Score against real Hamamatsu:

```text
SSIM ↑
windowed LAB ↓
LAB total ↓
PSNR ↑
MAE ↓
dE2000 ↓
recovery Δlab ↑
```

Use exactly the same scorer as the existing P1-10 evaluation.

Do not create a separate metric implementation.

---

# 25. Also run conditioning ablation at the best inversion setting

Once the best inversion fraction is identified on validation/smoke data, run:

```text
correct
shuffled
zero
```

at that same inversion setting.

We want to verify:

[\
SSIM\_{\text{correct}}

>

SSIM\_{\text{shuffled}}\
]

while also checking colour recovery.

This ensures inversion hasn't caused the model to ignore the trained source-conditioning branch.

---

# 26. Acceptance criteria

The implementation itself passes when:

1. It can perform DDIM inversion without random `add_noise()` initialization.
2. Inversion→reconstruction works deterministically.
3. Partial inversion works.
4. Correct P1-10 source conditioning remains active during translation.
5. Existing checkpoints load with no retraining.
6. Outputs are compatible with the existing scorer.
7. Old inference remains untouched.
8. No NaNs/Infs occur.
9. Timesteps are correctly paired between inverse and forward passes.
10. Identity reconstruction results can be compared directly with current img2img.

The scientific experiment is promising if:

[\
SSIM\_{\text{inversion translation}}

>

SSIM\_{\text{current P1-10}}\
]

while:

[\
LAB\_{\text{inversion translation}}\
\leq\
LAB\_{\text{current P1-10}}\
]

or there is a clearly superior colour/structure Pareto point.

Current P1-10 full held-out reference:

```text
SSIM:          0.4485
windowed LAB: 31.60
PSNR:         16.94
MAE:          27.81
recovery Δlab:+2.89
```

These values are the benchmark to beat.

---

# 27. Very important interpretation

Do not assume full inversion is automatically best.

The likely optimum may be a **partial inversion**.

Full inversion gives diffusion maximum freedom to perform the target-domain transformation but may introduce more reconstruction drift.

A shallow inversion may preserve much more source structure while still allowing the learned colour transformation to act.

So think of:

```text
--inversion-fraction
```

as the new analogue of the old:

```text
--strength
```

but obtained through a source-specific diffusion trajectory rather than arbitrary Gaussian corruption.

The experiment should search for the **best colour/structure operating point**, not merely compare `fraction=1.0`.

---

# 28. Deliverables

When finished, give me:

## A. Files created/changed

List every changed path.

There should preferably be no changes to the old validated P1-10 inference script.

## B. Technical explanatio

Explain:

1. How the current random img2img path works.
2. How the new inversion path works.
3. How source conditioning is handled during inversion.
4. How source conditioning is handled during translation.
5. How partial inversion maps onto timesteps.
6. Why scheduler compatibility with the trained checkpoint is preserved.

## C. Smoke-test commands

Give exact commands for:

```text
identity / full inversion
identity / partial inversion
translation / correct
translation / shuffled
translation / zero
```

## D. Smoke-test results

Report:

```text
SSIM
PSNR
MAE
```

for identity tests and the full existing metric set for translation.

## E. Do not launch the 496-crop full evaluation until the smoke test is validated

First show me the identity and small translation results.

Only after the inversion pathway is proven correct should the full held-out job be submitted.

---

# Final goal

The scientific question is:

> **Is part of P1-10's remaining structural-fidelity loss caused by the random img2img noise initialization, and can source-specific DDIM inversion recover that fidelity without sacrificing the improved Aperio→Hamamatsu colour transformation?**

Keep the implementation narrowly focused on answering that question.

Do not redesign P1-10.

Do not retrain anything.

Use the existing trained checkpoint and replace only the inference initialization path with a properly implemented, source-specific DDIM inversion trajectory.
