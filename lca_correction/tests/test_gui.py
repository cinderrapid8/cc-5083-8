"""Headless smoke test of the full GUI workflow under Qt offscreen mode."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from lca.gui import MainWindow
from lca.synthetic import SyntheticSpec, generate_dataset
from lca import imageio


def test_gui_full_workflow(tmp_path):
    app = QApplication.instance() or QApplication([])
    spec = SyntheticSpec(colorful=True)
    ds = generate_dataset(spec)
    img_path = tmp_path / "target.png"
    imageio.save_image_rgb(str(img_path), ds["warped"])

    win = MainWindow()
    win.open_image_path = str(img_path)
    win.rgb = imageio.load_image_rgb(str(img_path))
    win.image_path = str(img_path)
    win.canvas.set_image(win.rgb)

    # Detection (run synchronously by calling the underlying function).
    from lca.detection import detect_control_points, DetectionConfig
    control, stats = detect_control_points(win.rgb, "G", DetectionConfig())
    win._after_detect((control, stats))
    assert win.table.rowCount() > 0
    assert len(win.canvas.control) == 2

    # Edit: disable a point on the active channel.
    ch = win.ch_combo.currentText()
    cp = win.canvas.control[ch]
    win.canvas.enabled[ch][0] = False
    win.canvas.remove_disabled()
    assert len(win.canvas.control[ch]) == len(cp) - 1

    # Manual point add.
    before = len(win.canvas.control[ch])
    win.sp_x.setValue(100)
    win.sp_y.setValue(100)
    win._add_point()
    assert len(win.canvas.control[ch]) == before + 1

    # Fit (synchronous core call through the same path).
    from lca.calibrate import calibrate
    h, w = win.rgb.shape[:2]
    model = calibrate(w, h, win.canvas.control, "G")
    win._after_fit((model, win.canvas.control))
    assert win.model is not None
    assert "radial coverage" in win.report.toPlainText()
    assert win.model.r_norm_max > 0.5

    # Correct.
    from lca.correct import correct_image
    corrected, valid, _ = correct_image(win.rgb, win.model)
    win._after_correct((corrected, valid, _))
    assert win.corrected is not None
    assert win.valid_mask is not None
    assert 0 < (~win.valid_mask).mean() < 0.5

    # After editing points + correcting, channel edge positions and
    # residuals are still reported by the refactored chain.
    log = win.logbox.toPlainText()
    assert "edge position R" in log and "edge position B" in log
    assert "channel separation" in log
    assert win.residual_canvas.pixmap() is not None

    # Preview and batch share one mapping object per model.
    m1 = win._get_mapping()
    m2 = win._get_mapping()
    assert m1 is m2
    assert m1.center == win.model.center

    # View switching must not error.
    for mode in ("original", "corrected", "side-by-side"):
        win.view_combo.setCurrentText(mode)

    # Residual plot rendered.
    assert win.residual_canvas.pixmap() is not None

    # Save / load model round trip.
    model_path = tmp_path / "lens.json"
    from lca.calibrate import save_model, load_model
    save_model(str(model_path), win.model)
    loaded = load_model(str(model_path))
    assert loaded.width == win.model.width

    # Compatibility entry: loading through the GUI records adopted values.
    win.load_model_path(str(model_path))
    log = win.logbox.toPlainText()
    assert "compat: adopted reference_channel=G" in log
    assert "principal_point=(320.00, 240.00)" in log
    assert "default:image-center" in log
    # Loading a model invalidates the cached mapping.
    assert win._mapping is None

    # Batch apply.
    outdir = tmp_path / "out"
    outdir.mkdir()
    win.model = loaded
    files = [str(img_path)]
    ok = 0
    for path in files:
        rgb = imageio.load_image_rgb(path)
        c, v, _ = correct_image(rgb, loaded)
        base = os.path.splitext(os.path.basename(path))[0]
        imageio.save_image_rgb(str(outdir / f"{base}_lca.png"), c)
        imageio.save_mask(str(outdir / f"{base}_validmask.png"), v)
        ok += 1
    assert ok == 1
    assert (outdir / "target_lca.png").exists()
    assert (outdir / "target_validmask.png").exists()

    # Status log captured running state.
    log = win.logbox.toPlainText()
    assert "control points" in log
    assert "correction complete" in log
    app.processEvents()
