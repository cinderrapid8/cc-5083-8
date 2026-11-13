"""Image rendering helpers for the LCA GUI (numpy RGB -> QPixmap)."""

from __future__ import annotations

import numpy as np
from PySide6.QtGui import QImage, QPixmap

from .imageio import to_uint8_display


def rgb_to_pixmap(rgb_float: np.ndarray) -> QPixmap:
    arr = np.ascontiguousarray(to_uint8_display(rgb_float))
    h, w = arr.shape[:2]
    img = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img.copy())


def mask_to_pixmap(mask: np.ndarray) -> QPixmap:
    m = (mask.astype(np.uint8) * 255)
    img = QImage(m.data, m.shape[1], m.shape[0], m.shape[1],
                 QImage.Format.Format_Grayscale8)
    return QPixmap.fromImage(img.copy())
