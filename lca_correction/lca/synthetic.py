"""Synthetic ground-truth test set for LCA calibration.

A single sharp RGB image is generated, then *position-dependent* radial
displacement fields (the same model family the calibrator fits, plus a
higher-order radial term the 4-parameter model must approximate) are applied
per channel with dense sub-pixel remapping. Ground-truth displacement is known
exactly, so recovery can be checked quantitatively.

Two controls are included on purpose:

* ``colorful=True`` draws genuine colored outlines (red/green/blue shapes with
  no counterpart in other channels). These are real object colors, not
  mis-registration; the detector must reject them rather than chase them.
* Brightness-only variants scale/bias individual channels. Edge zero-crossing
  positions are invariant to affine intensity changes, so recovered geometry
  must stay unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np

from .model import CHANNELS, LcaModel, ChannelDisplacement


@dataclass
class SyntheticSpec:
    width: int = 640
    height: int = 480
    seed: int = 7
    n_rects: int = 26
    n_circles: int = 14
    colorful: bool = False


def _draw_base_scene(spec: SyntheticSpec) -> np.ndarray:
    rng = np.random.default_rng(spec.seed)
    h, w = spec.height, spec.width
    img = np.full((h, w, 3), 0.85, dtype=np.float64)

    # Neutral geometric content: grayscale rectangles and circles carrying
    # strong edges at many orientations and field positions.
    for _ in range(spec.n_rects):
        x0 = rng.integers(0, w - 80)
        y0 = rng.integers(0, h - 80)
        rw = rng.integers(25, 110)
        rh = rng.integers(25, 110)
        val = float(rng.uniform(0.05, 0.7))
        cv2.rectangle(img, (x0, y0), (x0 + rw, y0 + rh), (val, val, val), -1)
    for _ in range(spec.n_circles):
        cx = int(rng.integers(30, w - 30))
        cy = int(rng.integers(30, h - 30))
        rad = int(rng.integers(10, 34))
        val = float(rng.uniform(0.05, 0.65))
        cv2.circle(img, (cx, cy), rad, (val, val, val), -1)

    # Fine grid gives edges spread across the whole field of view, but it is
    # neutral (shared) content: disable it for the colorful-only negative
    # control, whose whole point is that no cross-channel edge exists.
    if not spec.colorful or spec.n_rects or spec.n_circles:
        for gx in range(20, w, 48):
            cv2.line(img, (gx, 0), (gx, h), (0.2, 0.2, 0.2), 1)
        for gy in range(20, h, 48):
            cv2.line(img, (0, gy), (w, gy), (0.2, 0.2, 0.2), 1)

    if spec.colorful:
        # Genuine colored outlines: edges exist in only one channel, so they
        # are object color, never channel mis-registration.
        cv2.rectangle(img, (40, 40), (180, 150), (0.95, 0.1, 0.1), 3)
        cv2.circle(img, (w - 110, h - 110), 55, (0.1, 0.9, 0.2), 4)
        cv2.ellipse(img, (w - 140, 120), (80, 40), 30, 0, 360,
                    (0.15, 0.2, 0.95), 4)
        cv2.line(img, (60, h - 60), (w - 80, 90), (0.95, 0.85, 0.1), 3)
    img = cv2.GaussianBlur(img, (0, 0), 0.8)  # mild anti-alias, still sharp
    return np.clip(img, 0.0, 1.0)


def make_ground_truth_model(width: int, height: int,
                            reference: str = "G") -> LcaModel:
    model = LcaModel(width=width, height=height,
                     reference_channel=reference)
    # Realistic values: a few pixels at the corner, red outward, blue inward
    # relative to green (signs chosen arbitrarily but consistently).
    model.channels["R"] = ChannelDisplacement("R", k1=0.012, k2=-0.004,
                                              p1=0.0006, p2=-0.0004)
    model.channels["B"] = ChannelDisplacement("B", k1=-0.010, k2=0.003,
                                              p1=-0.0005, p2=0.0007)
    return model


def _apply_forward(rgb: np.ndarray, model: LcaModel,
                   extra_radial: Dict[str, float] | None = None
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Warp channels by the *forward* LCA field (simulating lens capture)."""
    h, w = rgb.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float64),
                         np.arange(h, dtype=np.float64))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    out = np.zeros_like(rgb)
    truth_disp = {}
    for ci, c in enumerate(CHANNELS):
        if c == model.reference_channel:
            out[:, :, ci] = rgb[:, :, ci]
            continue
        d = model.displacement(grid)[c]
        if extra_radial and c in extra_radial:
            xy = model.normalized_coords(grid)
            r2 = (xy ** 2).sum(axis=1, keepdims=True)
            d = d + model.radius_scale * extra_radial[c] * r2 * xy
        src = (grid - d).astype(np.float32)
        warped = cv2.remap(rgb[:, :, ci], src[:, 0].reshape(h, w),
                           src[:, 1].reshape(h, w),
                           interpolation=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_REFLECT)
        out[:, :, ci] = warped
        truth_disp[c] = d
    return out, truth_disp


def generate_dataset(spec: SyntheticSpec,
                     reference: str = "G"
                     ) -> Dict[str, np.ndarray | LcaModel | dict]:
    base = _draw_base_scene(spec)
    gt = make_ground_truth_model(spec.width, spec.height, reference)
    # Add a fifth-order radial component that lies outside the fitted model
    # family, so the test also probes approximation / residual behavior.
    warped, truth = _apply_forward(
        base, gt, extra_radial={"R": 0.0015, "B": -0.0012})

    # Brightness variant: per-channel affine intensity change only.
    rng = np.random.default_rng(spec.seed + 100)
    bright = warped.copy()
    gains = {c: float(rng.uniform(0.7, 1.3)) for c in CHANNELS}
    biases = {c: float(rng.uniform(-0.08, 0.08)) for c in CHANNELS}
    for ci, c in enumerate(CHANNELS):
        bright[:, :, ci] = np.clip(bright[:, :, ci] * gains[c] + biases[c],
                                   0, 1)
    return {
        "base": base,
        "warped": warped,
        "brightness_variant": bright,
        "ground_truth": gt,
        "truth_displacement": truth,
        "gains": gains,
        "biases": biases,
    }
