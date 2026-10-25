#!/usr/bin/env python3
"""
nuclear_boundaries.py -- HoVer-Net nuclear-boundary maps, the "final structural
signal" the proposal names for ControlNet conditioning (proposal sec on
ControlNet: "HoVer-Net instance predictions are converted into black-background,
white-outline nuclear boundary maps"), with Canny as the prototype (canny.py).
P1-19 swaps this in for the Canny half of P1-10's 6-channel source condition.

Reads the per-image instance dicts hovernet_wrapper.py writes (stdlib pickle,
{nuc_id: {'contour': [[x,y],...], ...}}) -- so this module runs in the plain
`stainnorm` env, no tiatoolbox needed.

Pure functions apart from the one pickle read -- mirrors canny.py's contract:
an HxWx3 uint8 map, 0 or 255.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np


def load_instances(instances_dir: str | Path, stem: str) -> dict:
    """Load hovernet_wrapper.py's instances/<stem>.pkl. Hard-fails if missing --
    a missing map must never silently become a blank conditioning image. An
    image with no detected nuclei has an empty dict (-> blank map), which is fine."""
    path = Path(instances_dir) / f"{stem}.pkl"
    if not path.exists():
        raise SystemExit(f"No HoVer-Net instances for {stem} (expected {path}).")
    with open(path, "rb") as fh:
        return pickle.load(fh)


def boundary_control_image(inst_dict: dict, shape: tuple[int, int], thickness: int = 1) -> np.ndarray:
    """White 1 px nucleus outlines on black, expanded to 3 channels (same
    layout as extract_canny_control_image)."""
    edges = np.zeros(shape, dtype=np.uint8)
    for inst in inst_dict.values():
        contour = np.asarray(inst["contour"], dtype=np.int32).reshape(-1, 1, 2)
        if contour.shape[0] >= 2:
            cv2.polylines(edges, [contour], isClosed=True, color=255, thickness=thickness)
    return np.concatenate([edges[:, :, None]] * 3, axis=2)
