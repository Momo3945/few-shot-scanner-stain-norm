#!/usr/bin/env python3
"""
baseline_methods.py -- classical stain-normalisation baselines (P2-11, tab:baselines).

Macenko and Reinhard, verified against a real reference implementation
(github.com/m4ln/stain-normalization-reinhard-macenko-vahadane), not reconstructed
from memory. One deliberate deviation: Macenko's per-pixel concentration solve uses
a closed-form pseudo-inverse + non-negativity clip instead of the reference's
constrained LASSO via the `spams` package -- same core algorithm (OD space, SVD
stain-vector extraction, percentile-robust angle selection, 99th-percentile
concentration rescaling), avoiding an install-finicky extra dependency.

Pure functions/classes, no I/O -- mirrors the style of metrics.py/registration.py.
All three expect and return uint8 RGB arrays (H, W, 3).

Usage: fit each normalizer once on a single target reference crop (few-shot, same
scope as the colour LoRA's own A03/H03 training pair), then transform arbitrary
source crops.

    macenko = MacenkoNormalizer(); macenko.fit(target_rgb)
    out = macenko.transform(source_rgb)

Dependencies: numpy, opencv-python-headless, scikit-image.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.exposure import match_histograms


# ----------------------------------------------------------------------
# Shared helpers (Macenko + Reinhard both brightness-standardize first)
# ----------------------------------------------------------------------
def standardize_brightness(rgb: np.ndarray) -> np.ndarray:
    """Clip to the 90th-percentile pixel value, matching the reference impl."""
    p = np.percentile(rgb, 90)
    return np.clip(rgb * 255.0 / p, 0, 255).astype(np.uint8)


def rgb_to_od(rgb: np.ndarray) -> np.ndarray:
    """RGB (uint8) -> optical density. Zero pixels floored to 1 first (log(0) guard)."""
    I = rgb.copy()
    I[I == 0] = 1
    return -1.0 * np.log(I.astype(np.float64) / 255.0)


def od_to_rgb(od: np.ndarray) -> np.ndarray:
    return (255.0 * np.exp(-1.0 * od)).astype(np.uint8)


# ----------------------------------------------------------------------
# Macenko (macenko2009normalisation)
# ----------------------------------------------------------------------
def _macenko_stain_matrix(rgb: np.ndarray, beta: float = 0.15, alpha: float = 1.0) -> np.ndarray:
    """2x3 stain matrix (rows = hematoxylin, eosin; normalised to unit length)."""
    od = rgb_to_od(rgb).reshape(-1, 3)
    od = od[(od > beta).any(axis=1), :]   # drop near-background (low OD) pixels

    _, V = np.linalg.eigh(np.cov(od, rowvar=False))
    V = V[:, [2, 1]]   # top 2 eigenvectors (largest variance directions)
    if V[0, 0] < 0:
        V[:, 0] *= -1
    if V[0, 1] < 0:
        V[:, 1] *= -1

    proj = od @ V
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    min_phi = np.percentile(phi, alpha)
    max_phi = np.percentile(phi, 100 - alpha)
    v1 = V @ np.array([np.cos(min_phi), np.sin(min_phi)])
    v2 = V @ np.array([np.cos(max_phi), np.sin(max_phi)])
    stain_matrix = np.array([v1, v2]) if v1[0] > v2[0] else np.array([v2, v1])
    return stain_matrix / np.linalg.norm(stain_matrix, axis=1, keepdims=True)


def _macenko_concentrations(rgb: np.ndarray, stain_matrix: np.ndarray) -> np.ndarray:
    """Per-pixel (N, 2) non-negative stain concentrations.

    Closed-form least squares (OD @ pinv(stain_matrix)) clipped to non-negative,
    NOT the reference implementation's constrained LASSO (spams.lasso) -- see
    module docstring. Same 2-stain deconvolution objective, cheaper solve.
    """
    od = rgb_to_od(rgb).reshape(-1, 3)
    conc = od @ np.linalg.pinv(stain_matrix)
    return np.clip(conc, 0, None)


class MacenkoNormalizer:
    def __init__(self, beta: float = 0.15, alpha: float = 1.0):
        self.beta = beta
        self.alpha = alpha
        self.target_stain_matrix = None
        self.target_concentrations = None

    def fit(self, target_rgb: np.ndarray) -> None:
        target_rgb = standardize_brightness(target_rgb)
        self.target_stain_matrix = _macenko_stain_matrix(target_rgb, self.beta, self.alpha)
        self.target_concentrations = _macenko_concentrations(target_rgb, self.target_stain_matrix)

    def transform(self, source_rgb: np.ndarray) -> np.ndarray:
        if self.target_stain_matrix is None:
            raise RuntimeError("MacenkoNormalizer.fit() must be called before transform().")
        shape = source_rgb.shape
        source_rgb = standardize_brightness(source_rgb)
        source_stain_matrix = _macenko_stain_matrix(source_rgb, self.beta, self.alpha)
        source_conc = _macenko_concentrations(source_rgb, source_stain_matrix)

        max_c_source = np.percentile(source_conc, 99, axis=0, keepdims=True)
        max_c_target = np.percentile(self.target_concentrations, 99, axis=0, keepdims=True)
        source_conc = source_conc * (max_c_target / max_c_source)

        od_out = source_conc @ self.target_stain_matrix
        return od_to_rgb(od_out).reshape(shape)


# ----------------------------------------------------------------------
# Reinhard (reinhard2001colortransfer)
# ----------------------------------------------------------------------
def _lab_split(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    return l / 2.55, a - 128.0, b - 128.0


def _lab_merge(l: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    l = l * 2.55
    a = a + 128.0
    b = b + 128.0
    lab = np.clip(cv2.merge((l, a, b)), 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


class ReinhardNormalizer:
    def __init__(self):
        self.target_means = None
        self.target_stds = None

    def fit(self, target_rgb: np.ndarray) -> None:
        target_rgb = standardize_brightness(target_rgb)
        l, a, b = _lab_split(target_rgb)
        self.target_means = (l.mean(), a.mean(), b.mean())
        self.target_stds = (l.std(), a.std(), b.std())

    def transform(self, source_rgb: np.ndarray) -> np.ndarray:
        if self.target_means is None:
            raise RuntimeError("ReinhardNormalizer.fit() must be called before transform().")
        source_rgb = standardize_brightness(source_rgb)
        l, a, b = _lab_split(source_rgb)
        means = (l.mean(), a.mean(), b.mean())
        stds = (l.std(), a.std(), b.std())

        out = []
        for ch, m, sd, tm, tsd in zip((l, a, b), means, stds, self.target_means, self.target_stds):
            sd = sd if sd != 0 else 1.0
            out.append((ch - m) * (tsd / sd) + tm)
        return _lab_merge(*out)


# ----------------------------------------------------------------------
# Histogram matching -- thin wrapper, standard technique
# ----------------------------------------------------------------------
def histogram_match_normalize(source_rgb: np.ndarray, target_rgb: np.ndarray) -> np.ndarray:
    matched = match_histograms(source_rgb, target_rgb, channel_axis=-1)
    return np.clip(matched, 0, 255).astype(np.uint8)
