"""Resample all channels onto one common frame using the fitted LCA model.

The model states where a feature *observed* in channel c sits relative to the
reference frame:  p_c(observed) = p_ref + d_c(p_ref).  Correction therefore
remaps each channel with the inverse field, evaluated on a dense grid by
fixed-point inversion (the displacement is smooth and small).

Pixels whose source coordinate falls outside the frame, and pixels beyond the
calibrated radial coverage, are flagged in the output validity mask instead of
being silently extrapolated.
"""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

from .imageio import split_channels, join_channels
from .model import CHANNELS, LcaModel


def _inverse_maps(model: LcaModel, channel: str
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (map_x, map_y) giving source coords for every output pixel.

    Forward lens warp (see synthetic.py)::

        observed(x) = true(x - d(x))

    so a reference-frame feature at p_ref is seen at p_ref + d in the
    channel. To rebuild the reference frame we therefore sample the observed
    channel at source = output + d evaluated at the corresponding location.
    Fixed-point iteration inverts the position-dependent field.
    """
    h, w = model.height, model.width
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float64),
                         np.arange(h, dtype=np.float64))
    out = np.column_stack([gx.ravel(), gy.ravel()])
    src = out + model.displacement(out)[channel]
    for _ in range(6):
        d_src = model.displacement(src)[channel]
        src_new = out + d_src
        if np.max(np.abs(src_new - src)) < 1e-4:
            src = src_new
            break
        src = src_new
    map_x = src[:, 0].reshape(h, w).astype(np.float32)
    map_y = src[:, 1].reshape(h, w).astype(np.float32)
    return map_x, map_y


def build_validity_mask(model: LcaModel, maps: Dict[str, Tuple[np.ndarray, np.ndarray]]
                        ) -> np.ndarray:
    h, w = model.height, model.width
    valid = np.ones((h, w), dtype=bool)
    for (map_x, map_y) in maps.values():
        in_frame = (map_x >= 0) & (map_x <= w - 1) & \
                   (map_y >= 0) & (map_y <= h - 1)
        valid &= in_frame
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    r = model.normalized_radius(pts).reshape(h, w)
    valid &= r <= model.r_norm_max + 1e-9
    return valid


def correct_image(rgb: np.ndarray, model: LcaModel,
                  mark_invalid: bool = False
                  ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Correct an RGB image.

    Returns (corrected_rgb, validity_mask, displacement_maps).
    The reference channel is resampled through an identity map so every
    channel shares an identical pixel grid.
    """
    if rgb.shape[:2] != (model.height, model.width):
        raise ValueError(
            f"image {rgb.shape[:2]} does not match model "
            f"{(model.height, model.width)}")
    chans = dict(zip(CHANNELS, split_channels(rgb)))
    h, w = model.height, model.width
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    maps: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
        model.reference_channel: (gx, gy)}
    for c in CHANNELS:
        if c == model.reference_channel:
            continue
        maps[c] = _inverse_maps(model, c)

    out_chans = {}
    for c in CHANNELS:
        mx, my = maps[c]
        out_chans[c] = cv2.remap(
            chans[c], mx, my, interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    corrected = join_channels(*(out_chans[c] for c in CHANNELS))
    valid = build_validity_mask(model, maps)

    if mark_invalid:
        # Magenta indicator only for visualization, never a saturation cut.
        corrected[~valid] = np.array([1.0, 0.0, 1.0])
    return corrected, valid, maps
