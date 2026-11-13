"""Constrained lateral chromatic aberration (LCA) displacement model.

The displacement of every non-reference channel is modelled relative to the
reference channel as a smooth, center-anchored field made of two physically
motivated parts:

1. Radial LCA: displacement parallel to the radius vector, zero at the
   principal point, growing with r and r**3. This is the classical
   position-dependent lateral chromatic aberration of a lens.
2. Tangential (thin-prism) term: a small linear tilt term (p1, p2) * r**2,
   kept for mild decentring / prism effects.

A pure per-channel constant translation is intentionally *not* part of the
model: LCA vanishes at the optical axis and scales with field position.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Tuple

import numpy as np

CHANNELS = ("R", "G", "B")


def _as_xy(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (N, 2) with (x, y) rows")
    return pts


@dataclass
class ChannelDisplacement:
    """4-parameter constrained displacement for one non-reference channel."""

    channel: str
    k1: float = 0.0  # radial, linear in r
    k2: float = 0.0  # radial, cubic in r
    p1: float = 0.0  # tangential / thin prism, x component
    p2: float = 0.0  # tangential / thin prism, y component

    def to_array(self) -> np.ndarray:
        return np.array([self.k1, self.k2, self.p1, self.p2], dtype=np.float64)

    @classmethod
    def from_array(cls, channel: str, a: np.ndarray) -> "ChannelDisplacement":
        a = np.asarray(a, dtype=np.float64).ravel()
        if a.size != 4:
            raise ValueError("expected 4 coefficients")
        return cls(channel, float(a[0]), float(a[1]), float(a[2]), float(a[3]))


@dataclass
class LcaModel:
    """Full LCA model for one image format / lens setting."""

    width: int
    height: int
    reference_channel: str = "G"
    cx: Optional[float] = None  # principal point, pixels; None => image center
    cy: Optional[float] = None
    channels: Dict[str, ChannelDisplacement] = field(default_factory=dict)
    # Fitting diagnostics (filled in by calibration).
    r_norm_min: float = 0.0
    r_norm_max: float = 0.0  # largest normalized radius covered by control points
    n_control: int = 0
    residuals_px: Dict[str, float] = field(default_factory=dict)
    rmse_px: Dict[str, float] = field(default_factory=dict)
    max_abs_residual_px: Dict[str, float] = field(default_factory=dict)
    extrapolation_ratio: float = 0.0  # fraction of frame beyond r_norm_max
    fitted_center: bool = False
    notes: str = ""

    @property
    def center(self) -> Tuple[float, float]:
        cx = self.width / 2.0 if self.cx is None else self.cx
        cy = self.height / 2.0 if self.cy is None else self.cy
        return float(cx), float(cy)

    @property
    def radius_scale(self) -> float:
        # Normalization: half the image diagonal so corners have r ~= 1.
        return 0.5 * float(np.hypot(self.width, self.height))

    def normalized_coords(self, points: np.ndarray) -> np.ndarray:
        pts = _as_xy(points)
        cx, cy = self.center
        return (pts - np.array([cx, cy])) / self.radius_scale

    def displacement(self, points: np.ndarray) -> Dict[str, np.ndarray]:
        """Predicted displacement d_c (pixels) for every modelled channel."""
        pts = _as_xy(points)
        out: Dict[str, np.ndarray] = {}
        xy = self.normalized_coords(pts)
        x, y = xy[:, 0], xy[:, 1]
        r2 = x * x + y * y
        s = self.radius_scale
        for name, ch in self.channels.items():
            dx = s * (ch.k1 * x + ch.k2 * x * r2 + ch.p1 * r2)
            dy = s * (ch.k1 * y + ch.k2 * y * r2 + ch.p2 * r2)
            out[name] = np.column_stack([dx, dy])
        return out

    def normalized_radius(self, points: np.ndarray) -> np.ndarray:
        xy = self.normalized_coords(points)
        return np.hypot(xy[:, 0], xy[:, 1])

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LcaModel":
        chs = {
            name: ChannelDisplacement(**vals)
            for name, vals in d.get("channels", {}).items()
        }
        d = dict(d)
        d["channels"] = chs
        return cls(**d)


def make_identity_model(width: int, height: int,
                        reference_channel: str = "G") -> LcaModel:
    model = LcaModel(width=width, height=height,
                     reference_channel=reference_channel)
    for c in CHANNELS:
        if c != reference_channel:
            model.channels[c] = ChannelDisplacement(c)
    return model
