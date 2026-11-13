"""LCA calibration & correction workbench (PySide6).

Layout
------
* Toolbar: open image, detect control points, fit model, correct, save model,
  batch apply.
* Left: image canvas with editable per-channel control points and channel
  edge overlays; before/after edge-position comparison.
* Right tabs: control-point table (add/remove/enable), normal-residual
  scatter, fitted coefficients / coverage / extrapolation report, and the
  running-status log that flags abnormal channel edges and large residuals.
"""

from __future__ import annotations

import os
import traceback
from typing import Dict, Optional

import numpy as np
from PySide6.QtCore import Qt, QObject, Signal, QThread
from PySide6.QtGui import QAction, QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QTableWidget, QTableWidgetItem, QTabWidget,
    QPlainTextEdit, QComboBox, QCheckBox, QMessageBox,
    QSplitter, QHeaderView, QGroupBox, QFormLayout, QSpinBox, QStatusBar,
)

from . import imageio
from .calibrate import calibrate, save_model, load_model
from .correct import correct_image
from .detection import (
    detect_control_points, DetectionConfig, ControlPoints,
)
from .gui_render import rgb_to_pixmap
from .model import CHANNELS, LcaModel

CHANNEL_COLOR = {"R": QColor(255, 60, 60),
                 "G": QColor(60, 220, 90),
                 "B": QColor(70, 110, 255)}


class Worker(QObject):
    """Run a blocking pipeline step off the UI thread."""

    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            self.done.emit(self._fn(*self._args, **self._kwargs))
        except Exception:
            self.failed.emit(traceback.format_exc())


class ImageCanvas(QLabel):
    """Zoomable canvas that paints control points and channel edges."""

    pointsEdited = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background:#20242b;")
        self._pix: Optional[QPixmap] = None
        self._scale = 1.0
        self.control: Dict[str, ControlPoints] = {}
        self.enabled: Dict[str, np.ndarray] = {}
        self.show_points = True
        self.show_edges = True
        self.active_channel = "R"
        self.setMouseTracking(True)

    def set_image(self, rgb: np.ndarray):
        self._rgb = rgb
        self._pix = rgb_to_pixmap(rgb)
        self._rescale()

    def _rescale(self):
        if self._pix is None:
            return
        avail = self.size()
        self._scale = min(avail.width() / self._pix.width(),
                          avail.height() / self._pix.height(), 1.0)
        self.setPixmap(self._pix.scaled(
            self._pix.size() * self._scale,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))
        self.update()

    def resizeEvent(self, event):
        self._rescale()
        super().resizeEvent(event)

    def set_control(self, control: Dict[str, ControlPoints]):
        self.control = control
        self.enabled = {c: np.ones(len(cp), dtype=bool)
                        for c, cp in control.items()}
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._pix is None or not self.control:
            return
        painter = QPainter(self)
        pm = self.pixmap()
        ox = (self.width() - pm.width()) / 2
        oy = (self.height() - pm.height()) / 2

        if self.show_edges:
            for c, cp in self.control.items():
                pen = QPen(CHANNEL_COLOR[c], 1.4)
                painter.setPen(pen)
                m = self.enabled[c]
                p = cp.ch_points[m]
                for x, y in p[:: max(1, len(p) // 900)]:
                    painter.drawPoint(int(ox + x * self._scale),
                                      int(oy + y * self._scale))

        if self.show_points:
            for c, cp in self.control.items():
                color = CHANNEL_COLOR[c]
                pen = QPen(color, 1.6)
                painter.setPen(pen)
                m = self.enabled[c]
                p = cp.ref_points[m]
                r = 3.2 if c == self.active_channel else 2.2
                for x, y in p[:: max(1, len(p) // 500)]:
                    painter.drawEllipse(
                        int(ox + x * self._scale - r),
                        int(oy + y * self._scale - r),
                        int(2 * r), int(2 * r))

            # Outliers (large post-fit normal residual) get an orange cross.
            painter.setPen(QPen(QColor(255, 165, 0), 2.0))
            for c, cp in self.control.items():
                if cp.fit_inliers is None:
                    continue
                outlier = ~cp.fit_inliers
                p = cp.ref_points[outlier]
                for x, y in p[:: max(1, len(p) // 500)]:
                    x0, y0 = int(ox + x * self._scale), int(oy + y * self._scale)
                    painter.drawLine(x0 - 4, y0 - 4, x0 + 4, y0 + 4)
                    painter.drawLine(x0 - 4, y0 + 4, x0 + 4, y0 - 4)
        painter.end()

    def _image_coords(self, event) -> Optional[tuple]:
        if self._pix is None:
            return None
        pm = self.pixmap()
        ox = (self.width() - pm.width()) / 2
        oy = (self.height() - pm.height()) / 2
        x = (event.position().x() - ox) / self._scale
        y = (event.position().y() - oy) / self._scale
        if not (0 <= x < self._pix.width() and 0 <= y < self._pix.height()):
            return None
        return x, y

    def mousePressEvent(self, event):
        xy = self._image_coords(event)
        if xy is None or self.active_channel not in self.control:
            return
        cp = self.control[self.active_channel]
        if event.button() == Qt.MouseButton.LeftButton:
            # Toggle the nearest control point within 8 image pixels.
            d = np.linalg.norm(cp.ref_points - np.array(xy), axis=1)
            i = int(np.argmin(d))
            if d[i] <= 8:
                self.enabled[self.active_channel][i] = \
                    not self.enabled[self.active_channel][i]
                self.update()
                self.pointsEdited.emit()

    def add_manual_point(self, x: float, y: float):
        """Append a user control point with zero measured displacement."""
        if self.active_channel not in self.control:
            return
        cp = self.control[self.active_channel]
        cp.ref_points = np.vstack([cp.ref_points, [x, y]])
        cp.ch_points = np.vstack([cp.ch_points, [x, y]])
        cp.displacements = np.vstack([cp.displacements, [0.0, 0.0]])
        cp.normals = np.vstack([cp.normals, [1.0, 0.0]])
        self.enabled[self.active_channel] = np.append(
            self.enabled[self.active_channel], True)
        self.update()
        self.pointsEdited.emit()

    def remove_disabled(self):
        for c in list(self.control):
            m = self.enabled[c]
            if m.all():
                continue
            cp = self.control[c]
            self.control[c] = ControlPoints(
                cp.channel, cp.reference,
                cp.ref_points[m], cp.ch_points[m],
                cp.displacements[m], cp.normals[m])
            self.enabled[c] = np.ones(len(self.control[c]), dtype=bool)
        self.update()
        self.pointsEdited.emit()


class ResidualCanvas(QLabel):
    """Scatter of normal residuals vs normalized radius, with threshold."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(260)
        self.setStyleSheet("background:#1b1f26;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def plot(self, model: LcaModel, control: Dict[str, ControlPoints]):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=110)
        for c, cp in control.items():
            if cp.normal_residuals is None:
                continue
            r = model.normalized_radius(cp.ref_points)
            color = {"R": "r", "G": "g", "B": "b"}[c]
            inl = cp.fit_inliers if cp.fit_inliers is not None \
                else np.ones(len(r), bool)
            ax.scatter(r[inl], cp.normal_residuals[inl], s=6, c=color,
                       alpha=0.5, label=f"{c} inlier {inl.sum()}")
            if (~inl).any():
                ax.scatter(r[~inl], cp.normal_residuals[~inl], s=22,
                           c="orange", marker="x",
                           label=f"{c} outlier {(~inl).sum()}")
        ax.axhline(0, color="white", lw=0.8)
        ax.set_xlabel("normalized radius")
        ax.set_ylabel("normal residual (px)")
        ax.set_title("control-point residuals vs field position")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = "/tmp/_lca_residual.png"
        fig.savefig(path, facecolor="#1b1f26")
        plt.close(fig)
        self.setPixmap(QPixmap(path).scaledToWidth(
            620, Qt.TransformationMode.SmoothTransformation))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lateral Chromatic Aberration Workbench")
        self.resize(1380, 860)
        self.rgb: Optional[np.ndarray] = None
        self.image_path: Optional[str] = None
        self.control: Dict[str, ControlPoints] = {}
        self.model: Optional[LcaModel] = None
        self.corrected: Optional[np.ndarray] = None
        self.valid_mask: Optional[np.ndarray] = None
        self._thread = None
        self._worker = None

        self._build_ui()
        self._log("ready. open a calibration target to begin.")

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        toolbar = self.addToolBar("main")
        toolbar.setMovable(False)
        for text, slot in (
            ("Open image", self.open_image),
            ("Detect points", self.detect_points),
            ("Fit model", self.fit_model),
            ("Correct", self.correct),
            ("Save model", self.save_model_file),
            ("Load model", self.load_model_file),
            ("Batch apply…", self.batch_apply),
        ):
            act = QAction(text, self)
            act.triggered.connect(slot)
            toolbar.addAction(act)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left: canvas + controls.
        left = QWidget()
        lv = QVBoxLayout(left)
        self.canvas = ImageCanvas()
        self.canvas.pointsEdited.connect(self._refresh_tables)
        lv.addWidget(self.canvas, 1)

        ctl = QGroupBox("display / edit")
        form = QFormLayout(ctl)
        self.ch_combo = QComboBox()
        self.ch_combo.addItems([c for c in CHANNELS if c != "G"])
        self.ch_combo.currentTextChanged.connect(self._on_active_channel)
        form.addRow("active channel", self.ch_combo)
        self.chk_points = QCheckBox("control points")
        self.chk_points.setChecked(True)
        self.chk_points.toggled.connect(self._toggle_overlays)
        self.chk_edges = QCheckBox("channel edges")
        self.chk_edges.setChecked(True)
        self.chk_edges.toggled.connect(self._toggle_overlays)
        ov = QHBoxLayout()
        ov.addWidget(self.chk_points)
        ov.addWidget(self.chk_edges)
        form.addRow("overlays", ov)

        xy_row = QHBoxLayout()
        self.sp_x = QSpinBox(); self.sp_x.setRange(0, 100000)
        self.sp_y = QSpinBox(); self.sp_y.setRange(0, 100000)
        btn_add = QPushButton("add point")
        btn_add.clicked.connect(self._add_point)
        btn_drop = QPushButton("remove disabled")
        btn_drop.clicked.connect(self.canvas.remove_disabled)
        xy_row.addWidget(QLabel("x")); xy_row.addWidget(self.sp_x)
        xy_row.addWidget(QLabel("y")); xy_row.addWidget(self.sp_y)
        xy_row.addWidget(btn_add); xy_row.addWidget(btn_drop)
        form.addRow("manual", xy_row)

        self.view_combo = QComboBox()
        self.view_combo.addItems(["original", "corrected", "side-by-side"])
        self.view_combo.currentIndexChanged.connect(self._refresh_view)
        form.addRow("result view", self.view_combo)
        lv.addWidget(ctl)
        splitter.addWidget(left)

        # Right: tabs.
        self.tabs = QTabWidget()
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["channel", "x", "y", "nx", "ny", "enabled"])
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.tabs.addTab(self.table, "control points")

        self.residual_canvas = ResidualCanvas()
        self.tabs.addTab(self.residual_canvas, "residuals")

        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.tabs.addTab(self.report, "model / coverage")

        self.logbox = QPlainTextEdit()
        self.logbox.setReadOnly(True)
        self.tabs.addTab(self.logbox, "running status")
        splitter.addWidget(self.tabs)
        splitter.setSizes([820, 520])

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("idle")

    # ------------------------------------------------------------- helpers
    def _log(self, msg: str, level: str = "info"):
        tag = {"info": "•", "warn": "⚠", "error": "✖"}[level]
        self.logbox.appendPlainText(f"{tag} {msg}")
        self.statusBar().showMessage(msg)
        if level == "warn":
            self.logbox.setStyleSheet("color:#e0b34a;")
        elif level == "error":
            self.logbox.setStyleSheet("color:#e06a5a;")

    def _run_async(self, message: str, fn, on_done, *args, **kwargs):
        self.statusBar().showMessage(message)
        self._thread = QThread(self)
        self._worker = Worker(fn, *args, **kwargs)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(lambda r: self._finish(on_done, r))
        self._worker.failed.connect(self._on_fail)
        self._worker.done.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.start()

    def _finish(self, on_done, result):
        on_done(result)

    def _on_fail(self, tb: str):
        self._log(tb.strip().splitlines()[-1], "error")
        QMessageBox.critical(self, "step failed", tb)

    # ------------------------------------------------------------- actions
    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "open image", "", "Images (*.png *.tif *.tiff *.jpg *.jpeg)")
        if not path:
            return
        try:
            self.rgb = imageio.load_image_rgb(path)
        except Exception as exc:
            QMessageBox.critical(self, "open failed", str(exc))
            return
        self.image_path = path
        self.canvas.set_image(self.rgb)
        self.control = {}
        self.model = None
        self.corrected = None
        self._log(f"loaded {path} {self.rgb.shape[1]}x{self.rgb.shape[0]}")

    def detect_points(self):
        if self.rgb is None:
            QMessageBox.information(self, "no image", "open an image first")
            return
        cfg = DetectionConfig()

        def job():
            return detect_control_points(self.rgb, "G", cfg)

        self._run_async("detecting shared-edge control points…", job,
                        self._after_detect)

    def _after_detect(self, result):
        control, stats = result
        self.control = control
        self.canvas.set_control(control)
        self._refresh_tables()
        self._log(f"detected {stats['used_points']} usable control points "
                  f"({stats['rejected_color_or_weak']} color/weak edges rejected)")
        # Abnormal edge flag: very low yield suggests poor target.
        if stats["used_points"] < cfg_min_points():
            self._log("low control-point count: coverage will be limited",
                      "warn")

    def fit_model(self):
        if not self.control:
            QMessageBox.information(self, "no points", "detect points first")
            return
        h, w = self.rgb.shape[:2]
        active = {c: cp for c, cp in self.control.items()
                  if self.canvas.enabled[c].any()}

        def job():
            filtered = {}
            for c, cp in active.items():
                m = self.canvas.enabled[c]
                filtered[c] = ControlPoints(
                    cp.channel, cp.reference, cp.ref_points[m],
                    cp.ch_points[m], cp.displacements[m], cp.normals[m])
            return calibrate(w, h, filtered, "G"), filtered

        self._run_async("fitting constrained LCA model…", job,
                        self._after_fit)

    def _after_fit(self, result):
        self.model, used = result
        self.control = used
        self.canvas.set_control(used)
        self._refresh_tables()
        self.residual_canvas.plot(self.model, used)
        self._report_model(used)
        for c, cp in used.items():
            rmse = self.model.rmse_px[c]
            mx = self.model.max_abs_residual_px[c]
            n_out = int((~cp.fit_inliers).sum())
            frac_out = n_out / max(len(cp), 1)
            level = "warn" if rmse > 0.5 or mx > 1.5 or frac_out > 0.15 \
                else "info"
            self._log(
                f"channel {c}: rmse={rmse:.3f}px max={mx:.3f}px "
                f"inliers {cp.fit_inliers.sum()}/{len(cp)} "
                f"outliers {n_out}", level)
            if frac_out > 0.15:
                self._log(
                    f"{c}: {frac_out*100:.1f}% outlier control points "
                    "(possible color edges / mis-localization)", "warn")
        if self.model.extrapolation_ratio > 0.15:
            self._log(f"extrapolation risk: "
                      f"{self.model.extrapolation_ratio*100:.1f}% of frame is "
                      "outside calibrated coverage", "warn")

    def correct(self):
        if self.model is None:
            QMessageBox.information(self, "no model", "fit a model first")
            return

        def job():
            return correct_image(self.rgb, self.model)

        self._run_async("resampling channels onto common frame…", job,
                        self._after_correct)

    def _after_correct(self, result):
        self.corrected, self.valid_mask, _ = result
        self._refresh_view()
        invalid = 100 * (1 - self.valid_mask.mean())
        self._log(f"correction complete; {invalid:.1f}% of pixels masked as "
                  "out-of-range / extrapolated")
        if invalid > 20:
            self._log("large masked region: widen calibration coverage",
                      "warn")
        self._log_edge_positions()

    def _log_edge_positions(self):
        """Quantitative before/after channel edge-position comparison."""
        from .detection import DetectionConfig
        cfg = DetectionConfig(bucket_size=24, max_per_bucket=2)
        try:
            before, _ = detect_control_points(self.rgb, "G", cfg)
            after, _ = detect_control_points(self.corrected, "G", cfg)
        except RuntimeError as exc:
            self._log(f"edge-position comparison skipped: {exc}", "warn")
            return
        for c in before:
            if c not in after:
                continue
            sb = np.median(np.linalg.norm(before[c].displacements, axis=1))
            sa = np.median(np.linalg.norm(after[c].displacements, axis=1))
            level = "warn" if sa >= sb else "info"
            self._log(
                f"edge position {c}: median channel separation "
                f"{sb:.3f}px -> {sa:.3f}px", level)

    def save_model_file(self):
        if self.model is None:
            QMessageBox.information(self, "no model", "fit a model first")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "save model", "lens_lca.json", "JSON (*.json)")
        if path:
            save_model(path, self.model)
            self._log(f"model saved: {path}")

    def load_model_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "load model", "", "JSON (*.json)")
        if path:
            self.model = load_model(path)
            self._report_model()
            self._log(f"model loaded: {path}")

    def batch_apply(self):
        if self.model is None:
            QMessageBox.information(self, "no model", "fit or load a model first")
            return
        files, _ = QFileDialog.getOpenFileNames(
            self, "choose images", "", "Images (*.png *.tif *.jpg *.jpeg)")
        if not files:
            return
        outdir = QFileDialog.getExistingDirectory(self, "output directory")
        if not outdir:
            return
        ok, fail = 0, 0
        for path in files:
            try:
                rgb = imageio.load_image_rgb(path)
                if rgb.shape[:2] != (self.model.height, self.model.width):
                    self._log(f"skip {os.path.basename(path)}: size mismatch",
                              "warn")
                    fail += 1
                    continue
                corrected, valid, _ = correct_image(rgb, self.model)
                base = os.path.splitext(os.path.basename(path))[0]
                imageio.save_image_rgb(
                    os.path.join(outdir, base + "_lca.png"), corrected)
                imageio.save_mask(
                    os.path.join(outdir, base + "_validmask.png"), valid)
                ok += 1
            except Exception as exc:
                fail += 1
                self._log(f"failed {path}: {exc}", "error")
        self._log(f"batch done: {ok} corrected, {fail} skipped/failed -> {outdir}")

    # ------------------------------------------------------------- display
    def _report_model(self, used: Optional[Dict[str, ControlPoints]] = None):
        used = used if used is not None else self.control
        m = self.model
        lines = [
            f"frame {m.width}x{m.height}  reference {m.reference_channel}",
            f"principal point ({m.center[0]:.2f}, {m.center[1]:.2f})"
            f"{' fitted' if m.fitted_center else ' (center)'}",
            f"control points {m.n_control}",
            f"radial coverage (normalized) {m.r_norm_min:.3f} .. {m.r_norm_max:.3f}",
            f"extrapolation ratio {m.extrapolation_ratio*100:.2f}% of frame",
            "",
            "coefficients (normalized units):",
        ]
        for c, ch in m.channels.items():
            lines.append(
                f"  {c}: k1={ch.k1:+.5f} k2={ch.k2:+.5f} "
                f"p1={ch.p1:+.5f} p2={ch.p2:+.5f}")
        lines.append("")
        lines.append("residuals:")
        for c in m.channels:
            cp = used.get(c)
            n_out = int((~cp.fit_inliers).sum()) if cp is not None \
                and cp.fit_inliers is not None else 0
            lines.append(
                f"  {c}: rmse={m.rmse_px[c]:.3f}px "
                f"max={m.max_abs_residual_px[c]:.3f}px outliers={n_out}")
        self.report.setPlainText("\n".join(lines))

    def _refresh_tables(self):
        self.table.setRowCount(0)
        row = 0
        for c, cp in self.control.items():
            n = len(cp)
            self.table.setRowCount(row + n)
            en = self.canvas.enabled[c]
            for i in range(n):
                vals = [c, f"{cp.ref_points[i,0]:.2f}",
                        f"{cp.ref_points[i,1]:.2f}",
                        f"{cp.normals[i,0]:.2f}",
                        f"{cp.normals[i,1]:.2f}",
                        "yes" if en[i] else "no"]
                for j, v in enumerate(vals):
                    item = QTableWidgetItem(v)
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.table.setItem(row, j, item)
                row += 1

    def _refresh_view(self):
        mode = self.view_combo.currentText()
        if mode == "original" and self.rgb is not None:
            self.canvas.set_image(self.rgb)
        elif mode == "corrected" and self.corrected is not None:
            self.canvas.set_image(self.corrected)
        elif mode == "side-by-side" and self.corrected is not None:
            h = self.rgb.shape[0]
            gap = np.full((h, 8, 3), 0.3)
            combo = np.concatenate(
                [self.rgb, gap, self.corrected], axis=1)
            self.canvas.set_image(combo)
        self.canvas.set_control(self.control)

    def _on_active_channel(self, c: str):
        self.canvas.active_channel = c
        self.canvas.update()

    def _toggle_overlays(self):
        self.canvas.show_points = self.chk_points.isChecked()
        self.canvas.show_edges = self.chk_edges.isChecked()
        self.canvas.update()

    def _add_point(self):
        if self.rgb is None or not self.control:
            return
        self.canvas.active_channel = self.ch_combo.currentText()
        self.canvas.add_manual_point(float(self.sp_x.value()),
                                     float(self.sp_y.value()))


def cfg_min_points() -> int:
    return DetectionConfig().min_points


def run():
    import sys
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
