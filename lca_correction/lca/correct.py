"""Correction pipeline: geometry layer + sampling layer composition.

``correct_image`` keeps its pre-refactor signature and semantics, but the
work is now split into two inspectable stages:

1. :func:`lca.geometry.build_channel_mapping` -- target->source coordinate
   maps and the validity mask, derived from the model only (no colors).
2. :func:`lca.sampling.resample_image` -- per-channel resampling of the
   actual pixel data through those maps.

Model meaning, inversion tolerance and masking rules are unchanged.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .geometry import ChannelMapping, build_channel_mapping
from .model import LcaModel
from .sampling import resample_image


def correct_image(rgb: np.ndarray, model: LcaModel,
                  mark_invalid: bool = False
                  ) -> Tuple[np.ndarray, np.ndarray,
                             Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Correct an RGB image.

    Returns (corrected_rgb, validity_mask, source_maps). The reference
    channel is resampled through an identity map so every channel shares an
    identical pixel grid.
    """
    if rgb.shape[:2] != (model.height, model.width):
        raise ValueError(
            f"image {rgb.shape[:2]} does not match model "
            f"{(model.height, model.width)}")
    mapping: ChannelMapping = build_channel_mapping(model)
    corrected, valid = resample_image(rgb, mapping, mark_invalid=mark_invalid)
    return corrected, valid, mapping.source_maps
