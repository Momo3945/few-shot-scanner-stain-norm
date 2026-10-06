#!/usr/bin/env python3
"""
train_learned_baseline.py -- train a learned stain-normalisation baseline (P2-11b).

  --method stainnet   paired L1, SGD lr 0.01, cosine, bs 10, random 256 crops (upstream recipe)
  --method paramnet   CycleGAN-style ParamNet recipe (upstream train.py): ParamNet + ResnetGenerator
                      per direction, LSGAN + cycle(10) + diff(10) + identity(2), Adam 2e-4, bs 1
  --method staingan   standard CycleGAN: 2 x ResnetGenerator(9 blocks) + 2 x PatchGAN, LSGAN,
                      cycle(10) + identity(5), Adam 2e-4, linear decay over the second half

Data = the SAME coordinate-corresponding pairs the colour LoRA trains on
(pairs/train/*_aperio.png, *_hamamatsu.png). Methodology guardrails:
  * pairs are coordinate-corresponding, NOT pixel-exact -> no registration; StainNet's
    per-pixel colour map takes same-location crops, the GAN methods sample A and B
    UNPAIRED (as their papers do);
  * held-out slides A06/A08/A09/A13/A16 never enter training (checked below);
  * NO checkpoint selection against held-out data -- we keep the final-step weights only.
    Training loss is not a success signal; only score_outputs.py deltas are.

Direction: A2H maps Aperio -> Hamamatsu (domain A = aperio), H2A the reverse.
Inference weights are written to <out>/<method>_<direction>.pt (+ last.pt for resume).

Usage
-----
    python train_learned_baseline.py --method stainnet --train-dir pairs/train \
        --out baselines/a2h_stainnet --direction A2H [--smoke] [--steps N]
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ -> `baselines` package
from baselines.models import (ImagePool, NLayerDiscriminator, ParamNet, ResnetGenerator,  # noqa: E402
                              StainNet, gan_loss, init_weights)

HELDOUT_SLIDES = ("A06", "A08", "A09", "A13", "A16", "H06", "H08", "H09", "H13", "H16")
CROP = 256


def parse_args():
    ap = argparse.ArgumentParser(description="Train a learned stain-norm baseline (P2-11b).")
    ap.add_argument("--method", choices=["stainnet", "paramnet", "staingan"], required=True)
    ap.add_argument("--train-dir", required=True, help="pairs/train (contains *_aperio.png/*_hamamatsu.png).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--direction", choices=["A2H", "H2A"], default="A2H")
    ap.add_argument("--steps", type=int, default=0,
                    help="Total optimiser steps. 0 = method default (stainnet: 300 epochs of the "
                         "pair set x --repeat; paramnet 20000; staingan 20000).")
    ap.add_argument("--repeat", type=int, default=1,
                    help="stainnet: passes over the pair set per 'epoch' (fresh random crops each).")
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--smoke", action="store_true", help="5-step sanity run.")
    ap.add_argument("--resume", action="store_true", help="Resume from <out>/last.pt if present.")
    return ap.parse_args()


def load_pairs(train_dir: str):
    a_files = sorted(glob.glob(os.path.join(train_dir, "*_aperio.png")))
    if not a_files:
        raise SystemExit(f"No *_aperio.png in {train_dir}")
    a_imgs, h_imgs = [], []
    for fa in a_files:
        base = os.path.basename(fa)
        if base.startswith(HELDOUT_SLIDES):
            raise SystemExit(f"LEAK: held-out slide in training dir: {base}")
        fh = fa.replace("_aperio.png", "_hamamatsu.png")
        if not os.path.exists(fh):
            raise SystemExit(f"Missing Hamamatsu partner for {base}")
        a_imgs.append(np.asarray(Image.open(fa).convert("RGB")))
        h_imgs.append(np.asarray(Image.open(fh).convert("RGB")))
    a = torch.from_numpy(np.stack(a_imgs)).permute(0, 3, 1, 2).contiguous()  # uint8 N,3,H,W
    h = torch.from_numpy(np.stack(h_imgs)).permute(0, 3, 1, 2).contiguous()
    print(f"Loaded {len(a_files)} pairs, {tuple(a.shape[1:])}; leak check passed "
          f"(no {HELDOUT_SLIDES[:5]} frames).", flush=True)
    return a, h


def to_unit(x: torch.Tensor, device) -> torch.Tensor:
    """uint8 -> float in [-1, 1]."""
    return (x.to(device).float() - 127.5) / 127.5


def rand_crop(img: torch.Tensor, y: int, x: int) -> torch.Tensor:
    return img[..., y:y + CROP, x:x + CROP]


def sample_paired(src, tgt, bs, device):
    """Same-location crops from a source/target pair (StainNet)."""
    idx = torch.randint(0, src.shape[0], (bs,))
    H, W = src.shape[-2:]
    xs, ys = [], []
    for i in idx.tolist():
        y, x = random.randint(0, H - CROP), random.randint(0, W - CROP)
        xs.append(rand_crop(src[i], y, x))
        ys.append(rand_crop(tgt[i], y, x))
    return to_unit(torch.stack(xs), device), to_unit(torch.stack(ys), device)


def sample_unpaired(a, b, device):
    """Independent random crops from independently chosen images (GAN methods, bs 1)."""
    H, W = a.shape[-2:]
    ia, ib = random.randrange(a.shape[0]), random.randrange(b.shape[0])
    ca = rand_crop(a[ia], random.randint(0, H - CROP), random.randint(0, W - CROP))
    cb = rand_crop(b[ib], random.randint(0, H - CROP), random.randint(0, W - CROP))
    return to_unit(ca.unsqueeze(0), device), to_unit(cb.unsqueeze(0), device)


def set_requires_grad(nets, flag):
    for n in nets:
        for p in n.parameters():
            p.requires_grad = flag


# ----------------------------------------------------------------------
# StainNet
# ----------------------------------------------------------------------
def train_stainnet(args, src, tgt, out_dir, device):
    n_pairs, bs = src.shape[0], 10
    iters_per_epoch = max(1, -(-n_pairs // bs)) * args.repeat
    epochs = 300
    total = args.steps or epochs * iters_per_epoch
    model = StainNet(3, 3, 3, 32).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    # upstream: CosineAnnealingLR(T_max=epochs), stepped once per epoch -> per-step equivalent below
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total))
    step0 = 0
    last = out_dir / "last.pt"
    if args.resume and last.exists():
        ck = torch.load(last, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        step0 = ck["step"]
        print(f"Resumed at step {step0}", flush=True)
    l1 = nn.L1Loss()
    # Reference: L1 of the do-nothing mapping on the same crop distribution. A trained net whose
    # training L1 is not clearly below this has not converged (found 2026-10-05: 1,500 steps was
    # worse than identity, because 'epoch' = 5 steps on 50 pairs, unlike upstream's thousands).
    with torch.no_grad():
        ident = float(np.mean([l1(*sample_paired(src, tgt, bs, device)).item() for _ in range(50)]))
    print(f"identity-mapping L1 on training crops = {ident:.4f} (trained net must beat this)", flush=True)
    t0 = time.time()
    for step in range(step0 + 1, total + 1):
        x, y = sample_paired(src, tgt, bs, device)
        loss = l1(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if not torch.isfinite(loss):
            raise SystemExit(f"Non-finite loss at step {step}")
        if step % 50 == 0 or step == total or step <= 5:
            print(f"step {step}/{total}  l1={loss.item():.4f}  lr={sched.get_last_lr()[0]:.5f}  "
                  f"t={time.time() - t0:.0f}s", flush=True)
        if step % args.ckpt_every == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "step": step}, last)
    return {"model": model.state_dict()}, total, {"n_layer": 3, "n_channel": 32}


# ----------------------------------------------------------------------
# CycleGAN-style trainers (ParamNet, StainGAN). Domain A = source, B = target.
# ----------------------------------------------------------------------
def train_cyclegan(args, a_dom, b_dom, out_dir, device, method):
    total = args.steps or 20000
    warmup = min(1000, total // 10) if method == "paramnet" else 0
    lr = 2e-4
    inn = nn.InstanceNorm2d

    if method == "paramnet":
        gen_ab, gen_ba = ParamNet().to(device), ParamNet().to(device)       # net_G_A, net_G_B
        aux_ab = ResnetGenerator(3, 3, 64, inn, n_blocks=6).to(device)       # net_G_AA (refines fake_b)
        aux_ba = ResnetGenerator(3, 3, 64, inn, n_blocks=6).to(device)       # net_G_BB
        gen_nets = [gen_ab, gen_ba, aux_ab, aux_ba]
        init_gain = 0.002
    else:
        gen_ab = ResnetGenerator(3, 3, 64, inn, n_blocks=9).to(device)
        gen_ba = ResnetGenerator(3, 3, 64, inn, n_blocks=9).to(device)
        gen_nets = [gen_ab, gen_ba]
        init_gain = 0.02
    d_a = NLayerDiscriminator(3, 64, 3, inn).to(device)
    d_b = NLayerDiscriminator(3, 64, 3, inn).to(device)
    for n in gen_nets + [d_a, d_b]:
        init_weights(n, init_gain)

    opt_g = torch.optim.Adam(itertools.chain(*[n.parameters() for n in gen_nets]), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(itertools.chain(d_a.parameters(), d_b.parameters()), lr=lr, betas=(0.5, 0.999))
    l1, mse = nn.L1Loss(), nn.MSELoss()
    pool_a, pool_b = ImagePool(50), ImagePool(50)

    def lr_at(step):
        if method == "paramnet":      # upstream: linear warmup, then linear decay to 0
            if step < warmup:
                return step / warmup
            return max(0.0, 1.0 - (step - warmup) / max(1, total - warmup))
        half = total // 2             # CycleGAN: constant, then linear decay over the 2nd half
        return 1.0 if step < half else max(0.0, 1.0 - (step - half) / max(1, total - half))

    step0 = 0
    last = out_dir / "last.pt"
    nets = {"gen_ab": gen_ab, "gen_ba": gen_ba, "d_a": d_a, "d_b": d_b}
    if method == "paramnet":
        nets.update({"aux_ab": aux_ab, "aux_ba": aux_ba})
    if args.resume and last.exists():
        ck = torch.load(last, map_location=device)
        for k, n in nets.items():
            n.load_state_dict(ck[k])
        opt_g.load_state_dict(ck["opt_g"]); opt_d.load_state_dict(ck["opt_d"])
        step0 = ck["step"]
        print(f"Resumed at step {step0}", flush=True)

    t0 = time.time()
    for step in range(step0 + 1, total + 1):
        f = lr_at(step)
        for o in (opt_g, opt_d):
            for g in o.param_groups:
                g["lr"] = lr * f
        real_a, real_b = sample_unpaired(a_dom, b_dom, device)

        set_requires_grad([d_a, d_b], False)
        if method == "paramnet":
            fake_b = gen_ab(real_a); fake_bb = aux_ab(fake_b)
            fake_a = gen_ba(real_b); fake_aa = aux_ba(fake_a)
            rec_b = gen_ab(fake_aa); rec_a = gen_ba(fake_bb)
            loss_g = (gan_loss(d_a(fake_aa)) + gan_loss(d_b(fake_bb))
                      + 10.0 * l1(rec_a, real_a) + 10.0 * l1(rec_b, real_b)
                      + 10.0 * mse(fake_a, fake_aa.detach()) + 10.0 * mse(fake_b, fake_bb.detach())
                      + 2.0 * (l1(aux_ba(real_a), real_a) + l1(aux_ab(real_b), real_b)
                               + l1(gen_ba(real_a), real_a) + l1(gen_ab(real_b), real_b)))
            gen_for_d_a, gen_for_d_b = fake_aa, fake_bb      # what the discriminators judge
        else:
            fake_b, fake_a = gen_ab(real_a), gen_ba(real_b)
            rec_a, rec_b = gen_ba(fake_b), gen_ab(fake_a)
            loss_g = (gan_loss(d_b(fake_b)) + gan_loss(d_a(fake_a))
                      + 10.0 * l1(rec_a, real_a) + 10.0 * l1(rec_b, real_b)
                      + 5.0 * (l1(gen_ab(real_b), real_b) + l1(gen_ba(real_a), real_a)))
            gen_for_d_a, gen_for_d_b = fake_a, fake_b
        opt_g.zero_grad(); loss_g.backward(); opt_g.step()

        set_requires_grad([d_a, d_b], True)
        loss_d = (gan_loss(d_a(real_a), d_a(pool_a.query(gen_for_d_a.detach())))
                  + gan_loss(d_b(real_b), d_b(pool_b.query(gen_for_d_b.detach()))))
        opt_d.zero_grad(); loss_d.backward(); opt_d.step()

        if not (torch.isfinite(loss_g) and torch.isfinite(loss_d)):
            raise SystemExit(f"Non-finite loss at step {step}")
        if step % 100 == 0 or step == total or step <= 5:
            print(f"step {step}/{total}  loss_G={loss_g.item():.3f}  loss_D={loss_d.item():.3f}  "
                  f"lr={lr * f:.2e}  t={time.time() - t0:.0f}s", flush=True)
        if step % args.ckpt_every == 0:
            ck = {k: n.state_dict() for k, n in nets.items()}
            ck.update({"opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(), "step": step})
            torch.save(ck, last)

    meta = {"resample_size": 128, "channels": 8, "layers": 2} if method == "paramnet" else {"n_blocks": 9}
    return {"model": gen_ab.state_dict()}, total, meta


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("ABORT: no GPU on this node (torch.cuda.is_available() is False).")
    device = torch.device("cuda")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.smoke:
        args.steps, args.ckpt_every = 5, 5

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}  method={args.method}  direction={args.direction}", flush=True)

    aperio, hamamatsu = load_pairs(args.train_dir)
    src, tgt = (aperio, hamamatsu) if args.direction == "A2H" else (hamamatsu, aperio)

    if args.method == "stainnet":
        weights, total, meta = train_stainnet(args, src, tgt, out_dir, device)
    else:
        weights, total, meta = train_cyclegan(args, src, tgt, out_dir, device, args.method)

    ck_path = out_dir / f"{args.method}_{args.direction.lower()}.pt"
    torch.save({**weights, "method": args.method, "direction": args.direction, "arch": meta,
                "steps": total}, ck_path)
    with open(out_dir / "train_metadata.json", "w") as fh:
        json.dump({"script": "train_learned_baseline.py", "method": args.method,
                   "direction": args.direction, "steps": total, "seed": args.seed,
                   "train_dir": args.train_dir, "n_pairs": int(src.shape[0]),
                   "smoke": args.smoke, "selection": "final-step weights, no held-out selection"},
                  fh, indent=2)
    print(f"Saved inference weights -> {ck_path}", flush=True)


if __name__ == "__main__":
    main()
