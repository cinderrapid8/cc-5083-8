"""Robust fitting of the constrained LCA model from control points."""

from __future__ import annotations

import json
from typing import Dict, Tuple

import numpy as np

from .detection import ControlPoints
from .model import ChannelDisplacement, LcaModel, make_identity_model


class CalibrationError(RuntimeError):
    pass


def _irls_fit(A: np.ndarray, b: np.ndarray, n_iter: int = 10,
              huber: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """Huber-weighted least squares. Returns (coeffs, inlier_mask).

    Each row of A/b is one scalar observation (an edge-normal projection),
    so the tangentially-unobservable component of displacement never enters
    the normal equations.
    """
    w = np.ones(b.shape[0])
    coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
    for _ in range(n_iter):
        sw = np.sqrt(w)
        Aw = A * sw[:, None]
        bw = b * sw
        coeffs = np.linalg.lstsq(Aw, bw, rcond=None)[0]
        resid = b - A @ coeffs
        s = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-6
        u = np.abs(resid) / s
        w = np.where(u <= huber, 1.0, huber / np.maximum(u, 1e-9))
    resid = b - A @ coeffs
    s = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-6
    inliers = np.abs(resid) <= 2.5 * s
    return coeffs, inliers


def _normal_design(model: LcaModel, cp: ControlPoints) -> np.ndarray:
    """Project the displacement design onto each edge's unit normal."""
    pts = cp.ref_points
    xy = model.normalized_coords(pts)
    x, y = xy[:, 0], xy[:, 1]
    r2 = x * x + y * y
    nx, ny = cp.normals[:, 0], cp.normals[:, 1]
    # d = s * (k1*(x,y) + k2*r2*(x,y) + r2*(p1,p2))
    # n . d / s gives one scalar equation per edge.
    return np.column_stack([
        nx * x + ny * y,
        r2 * (nx * x + ny * y),
        nx * r2,
        ny * r2,
    ])


def fit_channel(model: LcaModel, cp: ControlPoints
                ) -> Tuple[ChannelDisplacement, dict]:
    pts = cp.ref_points
    s = model.radius_scale
    A = _normal_design(model, cp)
    # Observed normal displacement (pixels) -> normalized units.
    b = (cp.displacements * cp.normals).sum(axis=1) / s
    coeffs, _ = _irls_fit(A, b)

    normal_res = (b - A @ coeffs) * s  # back to pixels
    # Residuals are bimodal on real targets: a tight inlier population plus a
    # small population of color-edge localization failures. Gate on BOTH a
    # robust scale and an absolute sub-pixel-accurate physical ceiling, then
    # refit once on the surviving inliers to remove their leverage.
    mad = 1.4826 * np.median(np.abs(normal_res
                                    - np.median(normal_res))) + 1e-6
    inliers = (np.abs(normal_res) <= 6.0 * mad) & \
              (np.abs(normal_res) <= 1.5)
    if inliers.sum() >= 8:
        coeffs = np.linalg.lstsq(A[inliers], b[inliers], rcond=None)[0]
        normal_res = (b - A @ coeffs) * s
        inliers = np.abs(normal_res) <= 1.5

    ch = ChannelDisplacement.from_array(cp.channel, coeffs)
    r = model.normalized_radius(pts)
    diag = {
        "n": int(pts.shape[0]),
        "n_inliers": int(inliers.sum()),
        "rmse_px": float(np.sqrt(np.mean(normal_res[inliers] ** 2))),
        "max_abs_px": float(np.max(np.abs(normal_res[inliers]))),
        "r_min": float(r.min()),
        "r_max": float(r.max()),
        "coeffs": coeffs,
        "inliers": inliers,
        "normal_residuals": normal_res,
    }
    return ch, diag


def _fit_center(model: LcaModel, all_cp: Dict[str, ControlPoints]) -> Tuple[float, float]:
    """Optional small principal-point refinement by grid search + LS.

    Only a tight box around the image center is searched, keeping the result
    physically constrained; radial coefficients are refit at each candidate.
    """
    w, h = model.width, model.height
    base_cx, base_cy = w / 2.0, h / 2.0
    span = 0.04 * min(w, h)
    best = (base_cx, base_cy, np.inf)
    for dcx in np.linspace(-span, span, 7):
        for dcy in np.linspace(-span, span, 7):
            model.cx, model.cy = base_cx + dcx, base_cy + dcy
            err = 0.0
            for cp in all_cp.values():
                A = _normal_design(model, cp)
                b = (cp.displacements * cp.normals).sum(axis=1) \
                    / model.radius_scale
                c, *_ = np.linalg.lstsq(A, b, rcond=None)
                err += float(np.sum((b - A @ c) ** 2))
            if err < best[2]:
                best = (model.cx, model.cy, err)
    return best[0], best[1]


def calibrate(width: int, height: int,
              control: Dict[str, ControlPoints],
              reference: str = "G",
              fit_center: bool = False) -> LcaModel:
    if not control:
        raise CalibrationError("no control points supplied")
    model = make_identity_model(width, height, reference)

    if fit_center:
        cx, cy = _fit_center(model, control)
        model.cx, model.cy = float(cx), float(cy)
        model.fitted_center = True

    rmax_pts = []
    n_total = 0
    for c, cp in control.items():
        ch, diag = fit_channel(model, cp)
        model.channels[c] = ch
        model.rmse_px[c] = diag["rmse_px"]
        model.max_abs_residual_px[c] = diag["max_abs_px"]
        model.residuals_px[c] = diag["rmse_px"]
        rmax_pts.append(diag["r_max"])
        n_total += diag["n"]
        cp.fit_inliers = diag["inliers"]
        cp.normal_residuals = diag["normal_residuals"]

    # Coverage from all control points (they share anchors across channels).
    any_cp = next(iter(control.values()))
    r = model.normalized_radius(any_cp.ref_points)
    model.r_norm_min = float(r.min())
    model.r_norm_max = float(np.percentile(r, 98))
    model.n_control = int(any_cp.ref_points.shape[0])

    # Extrapolation ratio: fraction of the full pixel grid outside r_norm_max.
    H, W = np.mgrid[0:height, 0:width]
    grid_pts = np.column_stack([W.ravel(), H.ravel()])
    rg = model.normalized_radius(grid_pts)
    model.extrapolation_ratio = float(np.mean(rg > model.r_norm_max))
    return model


def save_model(path: str, model: LcaModel) -> None:
    payload = {"format": "lca-model-v1", "model": model.to_dict()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_model(path: str) -> LcaModel:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("format") != "lca-model-v1":
        raise CalibrationError("unrecognized model file")
    return LcaModel.from_dict(payload["model"])
