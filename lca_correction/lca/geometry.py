"""Geometry layer: turn an LcaModel into dense per-channel sampling maps.

Coordinate conventions
----------------------
*target* coordinates
    The shared reference-frame pixel grid of the output image. Both the GUI
    preview and the batch writer produce images on this grid.
*source* coordinates
    Positions inside each observed input channel that a target pixel must
    sample from.

The model displacement ``d_c(p)`` is defined on *target* positions ``p`` and
points from the reference frame into channel c's observed frame (see
lca.model). Correction therefore needs the inverse field: for every target
pixel, the source coordinate ``src = p + d(p)`` recovered by fixed-point
iteration (the field is smooth and small, tolerance 1e-4 px, at most 6
iterations -- unchanged from the pre-refactor pipeline).

This layer reads the model only -- never image colors -- so the mapping it
produces can be inspected and diffed on its own. Building one
:class:`ChannelMapping` per model and sharing it between the preview and the
batch entry points is what removes the duplicated center/radius conversion
both sides used to perform inside ``correct_image``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .model import CHANNELS, LcaModel

# Fixed-point inversion tolerance / iteration cap (model semantics, kept).
_INVERSE_TOL_PX = 1e-4
_INVERSE_MAX_ITER = 6


def target_grid(width: int, height: int) -> np.ndarray:
    """(N, 2) array of target-frame pixel coordinates (x, y)."""
    gx, gy = np.meshgrid(np.arange(width, dtype=np.float64),
                         np.arange(height, dtype=np.float64))
    return np.column_stack([gx.ravel(), gy.ravel()])


def invert_displacement(model: LcaModel, channel: str
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Source maps (map_x, map_y) for one channel, on the target grid.

    Forward lens warp (see synthetic.py)::

        observed(x) = true(x - d(x))

    so a reference-frame feature at p_ref is seen at p_ref + d in the
    channel. To rebuild the reference frame we sample the observed channel
    at source = target + d evaluated at the corresponding location.
    Fixed-point iteration inverts the position-dependent field.
    """
    h, w = model.height, model.width
    out = target_grid(w, h)
    src = out + model.displacement(out)[channel]
    for _ in range(_INVERSE_MAX_ITER):
        src_new = out + model.displacement(src)[channel]
        if np.max(np.abs(src_new - src)) < _INVERSE_TOL_PX:
            src = src_new
            break
        src = src_new
    map_x = src[:, 0].reshape(h, w).astype(np.float32)
    map_y = src[:, 1].reshape(h, w).astype(np.float32)
    return map_x, map_y


def _validity_mask(model: LcaModel,
                   maps: Dict[str, Tuple[np.ndarray, np.ndarray]]
                   ) -> np.ndarray:
    """Target-grid mask: source in frame for every channel AND inside the
    calibrated radial coverage (no silent extrapolation)."""
    h, w = model.height, model.width
    valid = np.ones((h, w), dtype=bool)
    for (map_x, map_y) in maps.values():
        in_frame = (map_x >= 0) & (map_x <= w - 1) & \
                   (map_y >= 0) & (map_y <= h - 1)
        valid &= in_frame
    r = model.normalized_radius(target_grid(w, h)).reshape(h, w)
    valid &= r <= model.r_norm_max + 1e-9
    return valid


@dataclass(frozen=True)
class ChannelMapping:
    """Dense target->source sampling plan derived from a model alone.

    Attributes
    ----------
    center, radius_scale
        The principal point and normalization actually adopted -- resolved
        once here so every consumer uses the same coordinate convention.
    source_maps
        channel -> (map_x, map_y) float32 source coordinates on the target
        grid. The reference channel carries the identity map so all channels
        share one pixel grid.
    validity_mask
        Target-frame bool grid; False where a source coordinate leaves the
        frame or the pixel lies beyond the calibrated radial coverage.
    """

    width: int
    height: int
    reference_channel: str
    center: Tuple[float, float]
    radius_scale: float
    r_norm_max: float
    source_maps: Dict[str, Tuple[np.ndarray, np.ndarray]]
    validity_mask: np.ndarray

    def channel_map(self, channel: str) -> Tuple[np.ndarray, np.ndarray]:
        return self.source_maps[channel]

    def describe(self) -> dict:
        """Color-free summary for inspection / regression diffing."""
        per_channel = {}
        for c, (mx, my) in self.source_maps.items():
            per_channel[c] = {
                "x_range": [float(mx.min()), float(mx.max())],
                "y_range": [float(my.min()), float(my.max())],
            }
        return {
            "frame": [self.width, self.height],
            "reference_channel": self.reference_channel,
            "center": [float(self.center[0]), float(self.center[1])],
            "radius_scale": float(self.radius_scale),
            "r_norm_max": float(self.r_norm_max),
            "valid_fraction": float(self.validity_mask.mean()),
            "source_ranges": per_channel,
        }


def build_channel_mapping(model: LcaModel) -> ChannelMapping:
    """The one mapping object shared by the preview and batch entries."""
    h, w = model.height, model.width
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    maps: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
        model.reference_channel: (gx, gy)}
    for c in CHANNELS:
        if c == model.reference_channel:
            continue
        maps[c] = invert_displacement(model, c)
    valid = _validity_mask(model, maps)
    return ChannelMapping(
        width=w, height=h,
        reference_channel=model.reference_channel,
        center=model.center,
        radius_scale=model.radius_scale,
        r_norm_max=model.r_norm_max,
        source_maps=maps,
        validity_mask=valid)
