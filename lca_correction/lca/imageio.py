"""Image loading / saving and small array helpers."""

from __future__ import annotations

import os
from typing import Tuple

import cv2
import numpy as np


def load_image_rgb(path: str) -> np.ndarray:
    """Load an image as float64 RGB in [0, 1], preserving bit depth."""
    raw = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"cannot read image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    maxv = float(np.iinfo(img.dtype).max) if np.issubdtype(img.dtype, np.integer) else 1.0
    return img.astype(np.float64) / maxv


def save_image_rgb(path: str, rgb_float: np.ndarray) -> None:
    rgb = np.clip(rgb_float, 0.0, 1.0)
    ext = os.path.splitext(path)[1] or ".png"
    if ext.lower() in (".jpg", ".jpeg"):
        out = (rgb * 255.0 + 0.5).astype(np.uint8)
        bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(ext, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    else:
        out = (rgb * 65535.0 + 0.5).astype(np.uint16)
        bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(ext, bgr)
    if not ok:
        raise IOError(f"cannot encode image: {path}")
    buf.tofile(path)


def save_mask(path: str, mask: np.ndarray) -> None:
    m = (mask.astype(np.uint8) * 255)
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, m)
    if not ok:
        raise IOError(f"cannot encode mask: {path}")
    buf.tofile(path)


def split_channels(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return rgb[:, :, 0].copy(), rgb[:, :, 1].copy(), rgb[:, :, 2].copy()


def join_channels(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.dstack([r, g, b])


def to_uint8_display(rgb_float: np.ndarray) -> np.ndarray:
    return (np.clip(rgb_float, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
