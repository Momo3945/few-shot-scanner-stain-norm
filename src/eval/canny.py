#!/usr/bin/env python3
"""
canny.py -- Canny edge-map extraction, the "prototype signal" ControlNet
conditions on for the A1 ablation rung (proposal sec on ControlNet: Geometry
Conditioning and Structural Safety).

Parameters (GaussianBlur (3,3), Otsu-adaptive Canny thresholds) match
archive/empty_stubs/canny_script.py -- the actual exploration script that
produced the proposal's own fig:canny1 figure. Reused here rather than picking
new thresholds, cleaned up into a pure, reusable function (no plotting, no
hardcoded paths).

Pure function, no I/O -- mirrors the style of metrics.py/registration.py/
baseline_methods.py.
"""

from __future__ import annotations

import cv2
import numpy as np


def extract_canny_control_image(rgb: np.ndarray, low_ratio: float = 0.5) -> np.ndarray:
    """Canny edge map from an RGB uint8 crop, expanded to 3 channels.

    ControlNet's Canny conditioning expects a 3-channel image (single-channel
    edge maps are not accepted directly) -- edges[:,:,None] concatenated x3 is
    the standard pattern for lllyasviel/sd-controlnet-canny.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    otsu_val, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(blurred, otsu_val * low_ratio, otsu_val)
    return np.concatenate([edges[:, :, None]] * 3, axis=2)
