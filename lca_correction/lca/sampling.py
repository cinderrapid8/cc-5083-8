"""Sampling (color) layer: apply a ChannelMapping to actual pixel data.

This is the only layer that touches channel values. It samples each
*source* channel at the source coordinates prepared by the geometry layer
and writes the result onto the shared *target* grid, then joins the
channels back into an RGB image. Given an independent synthetic
displacement field, correctness here can be verified without any
calibration machinery: warp forward, resample back, compare.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .geometry import ChannelMapping
from .imageio import split_channels, join_channels
from .model import CHANNELS


def resample_image(rgb: np.ndarray, mapping: ChannelMapping,
                   mark_invalid: bool = False
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Resample an RGB image onto the mapping's target grid.

    Returns (corrected_rgb, validity_mask). Pixels outside the validity
    mask are still resampled; with ``mark_invalid`` they are painted
    magenta for visualization only (never a saturation cut).
    """
    if rgb.shape[:2] != (mapping.height, mapping.width):
        raise ValueError(
            f"image {rgb.shape[:2]} does not match mapping "
            f"{(mapping.height, mapping.width)}")
    chans = dict(zip(CHANNELS, split_channels(rgb)))
    out_chans = {}
    for c in CHANNELS:
        mx, my = mapping.source_maps[c]
        out_chans[c] = cv2.remap(
            chans[c], mx, my, interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    corrected = join_channels(*(out_chans[c] for c in CHANNELS))
    valid = mapping.validity_mask
    if mark_invalid:
        corrected[~valid] = np.array([1.0, 0.0, 1.0])
    return corrected, valid
