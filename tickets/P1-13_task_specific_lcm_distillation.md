# P1-13 — Task-Specific LCM-LoRA Distillation from the Frozen P1-10 Teacher

**Status:** ✅ CLOSED (2026-08-26) — negative result. Full narrative,
prerequisite-gate evidence, training ablations, checkpoint-selection
methodology, post-training controls, the guidance-semantics correction, and
the full-UNet scope-matched replication (P1-13b) are all recorded in
`tickets/PHASE1-TICKETS.md`'s P1-13/P1-13b sections — this file is the
original task spec and is kept for reference, not updated further.

**Final outcome:** neither the attention-only adapter nor the full-UNet
scope-matched replication (P1-13b) beat the generic pretrained
`lcm-lora-sdv1-5` on the held-out A06+A08 evaluation, even after correcting
a train/inference guidance-semantics mismatch (`w` uses the LCM paper's CFG
form, where diffusers `guidance_scale` G = w + 1). Full-UNet scope brought
LAB approximately level with generic LCM within seed variation but did not
close the SSIM gap. Closed under the precommitted hard-stop rule — no
further rank sweeps, LR zoos, w-range sweeps, additional training slides,
new losses, or held-out tuning.

## Motivation

P1-12 tests the generic pretrained `latent-consistency/lcm-lora-sdv1-5` on top of the already-trained P1-10 source-conditioned A→H translator. The current evidence indicates that generic LCM acceleration preserves useful structure but attenuates the specialised scanner-colour transformation learned by P1-10.

This ticket asks whether the problem is specifically that the generic LCM-LoRA was distilled from ordinary SD1.5 behaviour rather than from the actual P1-10 deployment model.

The intended fix is to distil a new LCM-LoRA **from the frozen P1-10 conditional translator itself**.

This is consistent with the project methodology's intended training order: train/freeze the scanner-specific model first, then perform consistency distillation against the final frozen deployment configuration.

## Scientific Question

> Can an LCM-LoRA distilled directly from the trained P1-10 source-conditioned A→H translator preserve substantially more of the teacher's scanner-colour transformation than the generic SD1.5 LCM-LoRA while retaining few-step inference?

## Prerequisite Gate

Before implementing this ticket, complete a matched low-step diagnostic:

1. P1-10 + plain DDIM at a low step count.
2. P1-10 + generic LCM-LoRA at a comparable low step count.
3. Same source crops.
4. Same P1-10 checkpoint.
5. Same correct-source conditioning.
6. Same starting regime as closely as scheduler semantics permit.
7. Same evaluation metrics.

Proceed only if the evidence indicates that the generic LCM adapter itself materially attenuates the A→H mapping, rather than the degradation being explained entirely by having too few denoising updates.

If few-step DDIM and few-step LCM deteriorate similarly, document that result and do not assume custom LCM distillation will solve the issue.

## Teacher Definition

The teacher is the fully frozen P1-10 model:

```text
Frozen SD1.5 U-Net
        +
Frozen P1-10 colour LoRA
        +
Frozen P1-10 6-channel source-conditioning branch
(source RGB + source Canny)
        +
Frozen text encoder
        +
Frozen stock SD1.5 VAE
```

Checkpoint:

```text
lora/a2h_cond_r8/best
```

The teacher must use the exact same correct-source conditioning pathway that passed the P1-10 correct/zero/shuffled ablation.

Do **not** use the old target-only `a2h_r8` LoRA as the teacher.

## Scope Boundary

This ticket distils the **P1-10 conditional denoising model**.

It does **not** distil the P1-11 DDIM-inversion initialization mechanism.

Primary teacher reference:

```text
P1-10:
50-step DDIM
correct source conditioning
standard img2img initialization
```

P1-11 remains the high-quality reference for comparison, not the distillation teacher.

Do not silently introduce DDIM inversion into this training ticket.

## New Code Path

Do not modify the existing P1-10 training script.

Create a dedicated training path such as:

```text
src/train/train_p1_10_lcm_lora.py
slurm/train_p1_10_lcm_lora.slurm
```

Save to a fresh checkpoint namespace:

```text
lora/a2h_cond_r8_lcm_distilled/
```

Never overwrite:

```text
lora/a2h_cond_r8/best
```

or the generic LCM checkpoint.

## Implementation Preparation

Before writing the training loop:

1. Inspect the actual Diffusers version installed on the cluster.
2. Inspect the matching official LCM / LCM-LoRA consistency-distillation example for that version.
3. Inspect the current P1-10 training and inference code.
4. Confirm P1-10's prediction type.
5. Confirm the source-conditioning forward pass.
6. Confirm the modules targeted by the generic SD1.5 LCM-LoRA.
7. Confirm scheduler configuration and timestep conventions.
8. Confirm latent scaling.

Do not guess or blindly copy settings from another project.

P1-10 is known to use:

```text
prediction_type = epsilon
```

The custom distillation must respect the actual teacher formulation.

## Training Data

Use only the existing P1-10 training-domain paired data.

Reuse the existing frame-grouped train/validation split where practical:

```text
~39 train pairs
~11 validation pairs
```

or recreate the exact split deterministically from saved training metadata.

Never use:

```text
A06
A08
A09
A13
A16
```

for training or checkpoint selection.

## Conditional Distillation Requirement

Teacher and student must receive the **same correct source condition**.

Conceptually:

```text
source Aperio A
       │
       ├── RGB + Canny ──────► source-conditioning branch
       │
target Hamamatsu H
       │
       ▼
target latent + noise
       │
       ├────────► frozen P1-10 teacher
       │
       └────────► student + trainable LCM-LoRA
```

The student must not be allowed to learn an unconditional target-domain shortcut.

For matched teacher/student comparisons preserve:

- source identity;
- target identity;
- noise;
- timestep;
- prompt conditioning;
- source-conditioning tensor.

## Trainable Parameters

Only the **new LCM-LoRA** may receive gradients.

Freeze:

```text
base UNet
P1-10 colour LoRA
P1-10 source-conditioning branch
VAE
text encoder
```

At startup log:

```text
total parameters
trainable parameters
trainable parameter names
```

Abort training if any P1-10 colour-LoRA or source-conditioning parameter unexpectedly has `requires_grad=True`.

## Initial Training Budget

Initial maximum:

```text
4000 optimization steps
```

Save at least every:

```text
250 steps
```

This is a feasibility horizon, not an assumption that step 4000 is optimal.

Checkpoint selection must use the internal training-domain validation split.

Do not automatically select the final checkpoint.

## Validation

### A. Distillation validation

Track the consistency-distillation objective.

Purpose:

> Is the student learning to reproduce the frozen P1-10 teacher trajectory?

### B. Teacher-matching validation

On internal validation frames generate:

```text
P1-10 teacher output
task-specific LCM student output
```

for the same source and compare directly.

Report:

```text
SSIM(student, teacher)
PSNR(student, teacher)
MAE(student, teacher)
LAB distance(student, teacher)
```

This is a teacher-fidelity diagnostic, not the final biological evaluation.

## Mandatory Controls

### Control 1 — Generic vs task-specific LCM

Compare:

```text
generic pretrained LCM-LoRA
vs
P1-13 task-specific LCM-LoRA
```

under matched P1-10 inference settings.

### Control 2 — Source ablation

At the selected P1-13 checkpoint:

```text
correct
shuffled
zero
```

The correct source must remain structurally superior.

### Control 3 — Adapter isolation

Disabling the new P1-13 LCM-LoRA must return the model to the ordinary P1-10 behavior.

### Control 4 — Few-step DDIM reference

Keep the matched low-step DDIM control in the final analysis to distinguish:

```text
few-step limitation
```

from:

```text
generic LCM-adapter mismatch
```

## Few-Step Operating-Point Search

After checkpoint selection, test only a narrow grid.

Suggested first grid:

```text
steps    = {4, 6, 8}
guidance = {1.5, 2.0}
```

Use only a small strength range informed by P1-12.

For every configuration log:

```text
requested strength
requested inference steps
actual initial timestep
actual reverse-step count
```

Do not count byte-identical strength configurations as separate experiments.

## Held-Out Evaluation

Only after all training-domain choices are frozen, run the canonical:

```text
496 held-out crops × 3 seeds
```

on:

```text
A06
A08
A09
A13
A16
```

Report:

```text
SSIM
PSNR
MAE
windowed LAB
LAB total
dE2000
recovery Δlab
```

with:

```text
per-slide
ALL
ALL excluding A06
```

## Required Comparison Table

Include at minimum:

```text
P1-10 50-step DDIM
P1-11 DDIM inversion
P1-12 best generic LCM
P1-13 task-specific LCM
Raw
Macenko
Reinhard
Histogram Matching
```

## Runtime / Acceleration Measurement

Because this ticket concerns LCM acceleration, measure:

```text
mean inference time per crop
median inference time per crop
speedup vs P1-10 50-step DDIM
speedup vs P1-11
```

Use the same GPU class where practical.

## Acceptance Criteria

P1-13 is a scientific pass if it:

1. clearly beats the best P1-12 generic-LCM point on colour recovery;
2. does not materially reduce SSIM relative to that point;
3. retains correct-source dependence;
4. remains substantially faster than the 50-step quality path;
5. matches the P1-10 teacher more closely than the generic LCM adapter does.

A particularly strong outcome would be:

```text
task-specific LCM colour recovery >> generic LCM
```

with similar structural fidelity and few-step runtime.

It does **not** need to beat P1-11 outright; P1-11 remains the high-quality non-accelerated reference.

## Failure Interpretation

If P1-13 still cannot retain P1-10's colour transformation:

> Few-step consistency sampling is not fidelity-equivalent for this specialised scanner translation task under the current data and compute regime.

Do not continue indefinitely tuning strength or CFG after that result.

That is a valid RQ3 outcome: LCM provides major speed gains but the scanner-specific conditional transformation requires a longer trajectory for maximum fidelity.
