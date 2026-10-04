#!/usr/bin/env python3
"""
pathology_encoder.py -- P2-14: frozen pathology foundation-model encoder used
by the atypia probe (extract_atypia_features.py now, score_atypia_classifier.py
later). Single place for loading + preprocessing so training-time and
scoring-time features are bit-for-bit the same pipeline.

Primary: UNI2-h (MahmoodLab/UNI2-h), loaded from the local HF cache
(HF_HOME=/datasets/mhoosen/hf_cache, offline). Model-card timm kwargs.
Preprocessing = resize to 224 (LANCZOS, same as score_atypia_classifier.py's
existing convention) + ImageNet mean/std (UNI2-h's own pretrained_cfg).

Determinism: fp32 (no autocast), eval mode, TF32 off, cudnn deterministic.
"""

from __future__ import annotations

import os

ENCODERS = {
    "uni2h": {"hf_id": "MahmoodLab/UNI2-h", "dim": 1536, "input": 224},
}


def set_deterministic():
    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def load_encoder(name: str, device):
    """Returns (model, info) with model in eval mode on `device`."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", "/datasets/mhoosen/hf_cache")
    import timm
    import torch
    if name not in ENCODERS:
        raise SystemExit(f"Unknown encoder {name!r}; known: {sorted(ENCODERS)}")
    spec = ENCODERS[name]
    if name == "uni2h":
        kwargs = dict(
            img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5,
            embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True,
            mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
            reg_tokens=8, dynamic_img_size=True)
        model = timm.create_model(f"hf-hub:{spec['hf_id']}", pretrained=True, **kwargs)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    info = {"encoder": name, "hf_id": spec["hf_id"], "dim": spec["dim"],
            "timm": timm.__version__, "torch": torch.__version__}
    return model, info


def preprocess(rgb_uint8, size: int = 224):
    """HxWx3 uint8 -> normalised 3xSxS float32 tensor (ImageNet mean/std)."""
    import numpy as np
    import torch
    from PIL import Image
    img = Image.fromarray(rgb_uint8).resize((size, size), Image.LANCZOS)
    arr = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (arr - mean) / std


def encode(model, rgb_list, device, batch_size: int = 32, size: int = 224):
    """list of HxWx3 uint8 -> float32 [N, dim] numpy array (fp32, no_grad)."""
    import numpy as np
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(rgb_list), batch_size):
            x = torch.stack([preprocess(r, size) for r in rgb_list[i:i + batch_size]]).to(device)
            out.append(model(x).float().cpu().numpy())
            del x
    return np.concatenate(out, axis=0)
