"""Sub-pixel edge control-point detection for LCA calibration.

Strategy
--------
1. Edge *anchors* come from the reference channel only (Canny), so geometry is
   defined by one channel and the others are measured against it.
2. At every anchor the image gradient direction (common, smoothed structure
   tensor) gives the edge normal. The sub-pixel edge position of *each*
   channel along that normal is found from the zero crossing of the smoothed
   first derivative (derivative-of-Gaussian parabola fit), which depends on
   edge *shape* and not on the channel brightness/contrast.
3. Cross-channel consistency filters reject anchors where the channels do not
   share a physical edge (genuine color boundaries, texture, noise). Surviving
   anchors become control points whose displacement is the difference between
   the sub-pixel edge positions of two channels.

Brightness changes alone cannot shift a zero-crossing edge estimate, so
illumination variation is not misread as localization error (verified by the
synthetic test suite).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .model import CHANNELS


@dataclass
class ControlPoints:
    """Control points for one non-reference channel vs the reference."""

    channel: str
    reference: str
    # Reference anchor positions (x, y), sub-pixel, shape (N, 2).
    ref_points: np.ndarray
    # Same physical edge measured in the target channel, shape (N, 2).
    ch_points: np.ndarray
    # Observed displacement ch - ref along (x, y), shape (N, 2).
    displacements: np.ndarray
    # Edge normal (unit) at each anchor, shape (N, 2).
    normals: np.ndarray
    # Normal-direction signed residual after a model fit, shape (M,) subset.
    fit_inliers: Optional[np.ndarray] = None
    normal_residuals: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return int(self.ref_points.shape[0])


@dataclass
class DetectionConfig:
    canny_low: float = 0.05
    canny_high: float = 0.15
    sigma: float = 1.6           # Gaussian derivative scale, pixels
    min_gradient: float = 0.04   # normalized gradient magnitude threshold
    search_radius: int = 6       # max expected LCA + localization window, px
    max_normal_mismatch: float = 0.35  # radians of normal disagreement
    # Reject color boundaries: ratio of channel gradient *directions* /
    # relative magnitude spread beyond these means the edge is not shared.
    max_grad_ratio_spread: float = 1.8
    bucket_size: int = 12        # spatial bucketing for even field coverage
    max_per_bucket: int = 4
    min_points: int = 40


def _gaussian_derivatives(gray: np.ndarray, sigma: float):
    ksize = int(2 * round(3 * sigma) + 1)
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    gxb = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=ksize)
    gyb = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=ksize)
    return gxb, gyb, blur


def _subpixel_edge_on_normal(gray_blur: np.ndarray, x: float, y: float,
                             nx: float, ny: float, radius: int,
                             ) -> Optional[float]:
    """Signed distance along (nx, ny) of the sub-pixel edge nearest (x, y).

    Samples the blurred channel on the normal line and locates the maximum of
    the gradient magnitude via parabola fit of the discrete first difference.
    Returns None if no well-formed edge maximum exists inside the window.
    """
    ts = np.arange(-radius, radius + 1, dtype=np.float64)
    xs = x + nx * ts
    ys = y + ny * ts
    if xs.min() < 1 or ys.min() < 1 or xs.max() > gray_blur.shape[1] - 2 or \
       ys.max() > gray_blur.shape[0] - 2:
        return None
    vals = cv2.remap(gray_blur, xs.astype(np.float32).reshape(1, -1),
                     ys.astype(np.float32).reshape(1, -1),
                     interpolation=cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_REPLICATE).ravel()
    d = np.diff(vals)  # derivative samples at half-integer offsets
    if d.size < 3:
        return None
    idx = int(np.argmax(np.abs(d)))
    if idx == 0 or idx == len(d) - 1:
        return None  # edge at window border -> unreliable
    d0, d1, d2 = d[idx - 1], d[idx], d[idx + 1]
    denom = (d0 - 2.0 * d1 + d2)
    if abs(denom) < 1e-12:
        delta = 0.0
    else:
        delta = 0.5 * (d0 - d2) / denom
    if abs(delta) > 1.0:
        return None
    t_edge = ts[idx] + 0.5 + delta
    if abs(t_edge) > radius:
        return None
    return float(t_edge)


def _spatial_bucket_mask(pts: np.ndarray, width: int, height: int,
                         bucket: int, max_per_bucket: int) -> np.ndarray:
    keep = np.zeros(pts.shape[0], dtype=bool)
    counts: Dict[Tuple[int, int], int] = {}
    order = np.arange(pts.shape[0])
    rng = np.random.default_rng(0)
    rng.shuffle(order)
    for i in order:
        key = (int(pts[i, 0] // bucket), int(pts[i, 1] // bucket))
        if counts.get(key, 0) < max_per_bucket:
            keep[i] = True
            counts[key] = counts.get(key, 0) + 1
    return keep


def detect_control_points(rgb: np.ndarray,
                          reference: str = "G",
                          config: Optional[DetectionConfig] = None
                          ) -> Tuple[Dict[str, ControlPoints], dict]:
    """Detect shared-edge control points for each channel vs the reference."""
    cfg = config or DetectionConfig()
    h, w = rgb.shape[:2]
    idx = {c: i for i, c in enumerate(CHANNELS)}
    ref_i = idx[reference]
    ref_gray = rgb[:, :, ref_i]

    gxb, gyb, blur = _gaussian_derivatives(ref_gray, cfg.sigma)
    edges = cv2.Canny((np.clip(ref_gray, 0, 1) * 255).astype(np.uint8),
                      cfg.canny_low * 255, cfg.canny_high * 255,
                      L2gradient=True)
    # Thin anchors: keep pixels that are local maxima along the gradient.
    ys, xs = np.nonzero(edges)
    if xs.size < cfg.min_points:
        raise RuntimeError(f"too few reference edges ({xs.size})")

    # Precompute blurred channels for sub-pixel localization.
    blurs = {c: cv2.GaussianBlur(rgb[:, :, idx[c]], (0, 0), cfg.sigma)
             for c in CHANNELS}
    grads = {}
    for c in CHANNELS:
        gxc = cv2.Sobel(blurs[c], cv2.CV_64F, 1, 0, ksize=3)
        gyc = cv2.Sobel(blurs[c], cv2.CV_64F, 0, 1, ksize=3)
        grads[c] = (gxc, gyc, cv2.magnitude(gxc, gyc))

    records: List[dict] = []
    for x, y in zip(xs, ys):
        gxv, gyv = float(gxb[y, x]), float(gyb[y, x])
        gm = float(np.hypot(gxv, gyv))
        if gm < cfg.min_gradient:
            continue
        nx, ny = gxv / gm, gyv / gm  # reference edge normal

        # Shared-edge test: every channel must have a strong, co-directed edge.
        ok = True
        ch_t: Dict[str, float] = {}
        ch_gm: Dict[str, float] = {}
        ref_sign = None
        for c in CHANNELS:
            gxc, gyc, mc = grads[c]
            m = float(mc[y, x])
            if m < cfg.min_gradient * 0.5:
                ok = False
                break
            dot = (gxc[y, x] * gxv + gyc[y, x] * gyv) / (m * gm + 1e-12)
            if dot < np.cos(cfg.max_normal_mismatch):
                ok = False  # color boundary: gradients point different ways
                break
            # Edge polarity must agree: a dark-to-light transition in the
            # reference has to be dark-to-light in every channel. Colored
            # outlines flip or vanish this sign in some channels.
            if ref_sign is None:
                ref_sign = np.sign(dot)
            elif np.sign(dot) != ref_sign:
                ok = False
                break
            t = _subpixel_edge_on_normal(blurs[c], float(x), float(y),
                                         nx, ny, cfg.search_radius)
            if t is None:
                ok = False
                break
            ch_t[c] = t
            ch_gm[c] = m
        if not ok:
            continue
        # Reject edges whose channel gradient strengths differ implausibly
        # (strong color transition rather than a shared intensity edge).
        gm_vals = np.array([ch_gm[c] for c in CHANNELS])
        if gm_vals.max() / max(gm_vals.min(), 1e-9) > cfg.max_grad_ratio_spread:
            continue
        records.append({
            "x": float(x), "y": float(y), "nx": nx, "ny": ny,
            "t": ch_t, "ref_mag": gm,
        })

    if len(records) < cfg.min_points:
        raise RuntimeError(
            f"only {len(records)} shared-edge control points survived "
            "color-consistency filtering; use a target with neutral edges")

    pts0 = np.array([[r["x"], r["y"]] for r in records])
    keep = _spatial_bucket_mask(pts0, w, h, cfg.bucket_size, cfg.max_per_bucket)
    records = [r for k, r in zip(keep, records) if k]

    out: Dict[str, ControlPoints] = {}
    ref_pts = np.array([[r["x"] + r["nx"] * r["t"][reference],
                         r["y"] + r["ny"] * r["t"][reference]]
                        for r in records])
    normals = np.array([[r["nx"], r["ny"]] for r in records])
    for c in CHANNELS:
        if c == reference:
            continue
        ch_pts = np.array([[r["x"] + r["nx"] * r["t"][c],
                            r["y"] + r["ny"] * r["t"][c]]
                           for r in records])
        disp = ch_pts - ref_pts
        out[c] = ControlPoints(
            channel=c, reference=reference,
            ref_points=ref_pts.copy(), ch_points=ch_pts,
            displacements=disp, normals=normals.copy())

    stats = {
        "canny_anchors": int(xs.size),
        "shared_edge_points": int(len(pts0)),
        "used_points": int(len(records)),
        "rejected_color_or_weak": int(xs.size - len(records)),
    }
    return out, stats
