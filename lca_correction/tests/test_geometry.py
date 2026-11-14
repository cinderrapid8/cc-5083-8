"""Geometry/sampling layer tests: eccentric axis, non-square frames,
peripheral sampling gaps, pixel-by-pixel map/mask verification, and
independent synthetic-displacement validation of the sampling layer."""

import os

import cv2
import numpy as np
import pytest

from lca.calibrate import calibrate, save_model
from lca.compat import load_model_compat, adopted_values, ui_session_defaults
from lca.correct import correct_image
from lca.detection import DetectionConfig, detect_control_points
from lca.geometry import build_channel_mapping
from lca.model import ChannelDisplacement, make_identity_model
from lca.sampling import resample_image
from lca.synthetic import SyntheticSpec, generate_dataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _model(w, h, cx=None, cy=None, rmax=1.2):
    m = make_identity_model(w, h, "G")
    m.channels["R"] = ChannelDisplacement("R", 0.012, -0.004, 0.0006, -0.0004)
    m.channels["B"] = ChannelDisplacement("B", -0.010, 0.003, -0.0005, 0.0007)
    m.cx, m.cy = cx, cy
    m.r_norm_max = rmax
    return m


SCENARIOS = {
    "center": _model(640, 480),
    "eccentric_axis": _model(640, 480, cx=300.25, cy=219.5),
    "nonsquare": _model(400, 240),
    "nonsquare_eccentric": _model(400, 240, cx=150.5, cy=140.25),
    "peripheral_gap": _model(640, 480, rmax=0.45),
}


def _reference_inverse_maps(model, channel):
    """Independent re-implementation of the fixed-point inversion, used to
    cross-check the geometry layer pixel by pixel."""
    h, w = model.height, model.width
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float64),
                         np.arange(h, dtype=np.float64))
    out = np.column_stack([gx.ravel(), gy.ravel()])
    src = out + model.displacement(out)[channel]
    for _ in range(6):
        new = out + model.displacement(src)[channel]
        if np.max(np.abs(new - src)) < 1e-4:
            src = new
            break
        src = new
    return (src[:, 0].reshape(h, w).astype(np.float32),
            src[:, 1].reshape(h, w).astype(np.float32))


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_channel_maps_match_reference_pixel_by_pixel(name):
    model = SCENARIOS[name]
    mapping = build_channel_mapping(model)
    h, w = model.height, model.width
    # Reference channel: exact identity map on the target grid.
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    np.testing.assert_array_equal(mapping.source_maps["G"][0], gx)
    np.testing.assert_array_equal(mapping.source_maps["G"][1], gy)
    for c in ("R", "B"):
        ex, ey = _reference_inverse_maps(model, c)
        mx, my = mapping.source_maps[c]
        np.testing.assert_array_equal(mx, ex)
        np.testing.assert_array_equal(my, ey)


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_validity_mask_pixel_by_pixel(name):
    model = SCENARIOS[name]
    mapping = build_channel_mapping(model)
    h, w = model.height, model.width
    expected = np.ones((h, w), dtype=bool)
    for mx, my in mapping.source_maps.values():
        expected &= (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    r = model.normalized_radius(pts).reshape(h, w)
    expected &= r <= model.r_norm_max + 1e-9
    np.testing.assert_array_equal(mapping.validity_mask, expected)


def test_eccentric_axis_shifts_mapping_and_valid_region():
    centered = build_channel_mapping(SCENARIOS["center"])
    eccentric = build_channel_mapping(SCENARIOS["eccentric_axis"])
    assert eccentric.center == (300.25, 219.5)
    # Displacement vanishes at the principal point: the nearest target
    # pixel samples (almost) itself.
    mx, my = eccentric.source_maps["R"]
    ix, iy = int(round(300.25)), int(round(219.5))
    assert abs(mx[iy, ix] - ix) < 0.1
    assert abs(my[iy, ix] - iy) < 0.1
    # The valid region is anchored on the eccentric axis, not the frame
    # center: masks must differ somewhere.
    assert not np.array_equal(centered.validity_mask, eccentric.validity_mask)


def test_nonsquare_frame_uses_diagonal_radius_scale():
    mapping = build_channel_mapping(SCENARIOS["nonsquare"])
    assert mapping.radius_scale == pytest.approx(0.5 * np.hypot(400, 240))
    assert mapping.validity_mask.shape == (240, 400)


def test_peripheral_gap_flags_corners_keeps_center():
    mapping = build_channel_mapping(SCENARIOS["peripheral_gap"])
    valid = mapping.validity_mask
    h, w = valid.shape
    assert valid[h // 2, w // 2]
    assert not valid[0, 0]
    assert not valid[-1, -1]
    # Gap scenario: most of the frame is outside calibrated coverage.
    assert valid.mean() < 0.5


def test_geometry_layer_needs_no_colors():
    model = SCENARIOS["eccentric_axis"]
    mapping = build_channel_mapping(model)  # model only, no image
    desc = mapping.describe()
    assert desc["center"] == [300.25, 219.5]
    assert desc["reference_channel"] == "G"
    assert set(desc["source_ranges"]) == {"R", "G", "B"}
    for maps in mapping.source_maps.values():
        for m in maps:
            assert np.isfinite(m).all()


def test_sampling_layer_recovers_independent_synthetic_displacement():
    """Forward-warp a smooth image with an independently computed field,
    then resample through the mapping: content must come back."""
    h, w = 240, 320
    model = _model(w, h, rmax=2.0)
    rng = np.random.default_rng(3)
    base = cv2.GaussianBlur(rng.random((h, w, 3)), (0, 0), 2.0)

    gx, gy = np.meshgrid(np.arange(w, dtype=np.float64),
                         np.arange(h, dtype=np.float64))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    warped = base.copy()
    for i, c in enumerate(("R", "G", "B")):
        if c == "G":
            continue
        d = model.displacement(grid)[c]
        src = (grid - d).astype(np.float32)
        warped[:, :, i] = cv2.remap(
            base[:, :, i], src[:, 0].reshape(h, w), src[:, 1].reshape(h, w),
            interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

    mapping = build_channel_mapping(model)
    corrected, valid = resample_image(warped, mapping)
    # Reference channel: identity map must be exact to within float32
    # rounding, ignoring the 2-px border where the cubic kernel reads out
    # of frame (BORDER_CONSTANT).
    np.testing.assert_allclose(corrected[2:-2, 2:-2, 1],
                               warped[2:-2, 2:-2, 1], atol=1e-6)
    # Outward displacement pushes source coords out of frame near the
    # border; the interior is fully valid and recovers the base content.
    interior = np.zeros((h, w), bool)
    interior[8:-8, 8:-8] = True
    assert valid[interior].all()
    err = np.abs(corrected - base)[interior]
    assert err.mean() < 0.02
    assert err.max() < 0.1


def _edge_column(img2d, row):
    """Sub-pixel column of the steepest zero-crossing edge on one row."""
    d = np.diff(img2d[row])
    i = int(np.argmax(np.abs(d)))
    d0, d1, d2 = d[i - 1], d[i], d[i + 1]
    denom = d0 - 2 * d1 + d2
    delta = 0.0 if abs(denom) < 1e-12 else 0.5 * (d0 - d2) / denom
    return i + 0.5 + delta


def test_channel_positions_aligned_after_sampling():
    """Step-edge target: after correction the R/B edge columns coincide
    with the reference channel edge (channel positions verified)."""
    h, w = 120, 200
    model = _model(w, h, rmax=2.0)
    base = np.zeros((h, w, 3))
    base[:, 100:160] = 1.0
    base = cv2.GaussianBlur(base, (0, 0), 0.8)

    gx, gy = np.meshgrid(np.arange(w, dtype=np.float64),
                         np.arange(h, dtype=np.float64))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    warped = base.copy()
    for i, c in enumerate(("R", "G", "B")):
        if c == "G":
            continue
        d = model.displacement(grid)[c]
        src = (grid - d).astype(np.float32)
        warped[:, :, i] = cv2.remap(
            base[:, :, i], src[:, 0].reshape(h, w), src[:, 1].reshape(h, w),
            interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

    corrected, _ = resample_image(warped, build_channel_mapping(model))
    row = h // 2
    pos = {c: _edge_column(corrected[:, :, i], row)
           for i, c in enumerate(("R", "G", "B"))}
    assert abs(pos["R"] - pos["G"]) < 0.15
    assert abs(pos["B"] - pos["G"]) < 0.15


def test_preview_and_batch_share_one_mapping_object():
    """The two entry points must observe identical results from the shared
    mapping, and the compatibility wrapper must agree with both."""
    model = _model(320, 240, cx=150.5, cy=130.25)
    rng = np.random.default_rng(5)
    img_a = cv2.GaussianBlur(rng.random((240, 320, 3)), (0, 0), 1.5)
    img_b = cv2.GaussianBlur(rng.random((240, 320, 3)), (0, 0), 1.5)

    mapping = build_channel_mapping(model)  # built once, like the GUI cache
    pa, va = resample_image(img_a, mapping)          # preview entry
    ba, va2 = resample_image(img_a, mapping)         # batch entry, same map
    np.testing.assert_array_equal(pa, ba)
    np.testing.assert_array_equal(va, va2)

    ca, cva, maps = correct_image(img_a, model)      # compat wrapper
    np.testing.assert_array_equal(pa, ca)
    np.testing.assert_array_equal(va, cva)
    np.testing.assert_array_equal(maps["R"][0], mapping.source_maps["R"][0])

    # The same mapping serves a second image (batch reuse) identically to a
    # fresh per-image pipeline run.
    cb, cvb, _ = correct_image(img_b, model)
    bb, vbb = resample_image(img_b, mapping)
    np.testing.assert_array_equal(cb, bb)
    np.testing.assert_array_equal(cvb, vbb)


def test_peripheral_gap_from_sparse_control_points():
    """End-to-end: control points missing the periphery -> corners flagged
    invalid, interior valid, on real fitted models."""
    ds = generate_dataset(SyntheticSpec(colorful=True))
    control, _ = detect_control_points(ds["warped"], "G")
    probe = make_identity_model(640, 480, "G")
    filtered = {}
    for c, cp in control.items():
        r = probe.normalized_radius(cp.ref_points)
        keep = r <= 0.5
        filtered[c] = type(cp)(cp.channel, cp.reference,
                               cp.ref_points[keep], cp.ch_points[keep],
                               cp.displacements[keep], cp.normals[keep])
    model = calibrate(640, 480, filtered, "G")
    assert model.r_norm_max <= 0.55
    mapping = build_channel_mapping(model)
    valid = mapping.validity_mask
    assert valid[240, 320]
    assert not valid[0, 0] and not valid[479, 639]
    assert (~valid).mean() >= model.extrapolation_ratio * 0.9


def test_compat_records_adopted_values(tmp_path):
    m = _model(640, 480)  # cx/cy None -> image-center default
    path = tmp_path / "m.json"
    save_model(str(path), m)
    loaded, adopted = load_model_compat(str(path))
    assert adopted.center == (320.0, 240.0)
    assert adopted.center_source == "default:image-center"
    assert adopted.reference_channel == "G"
    assert adopted.radius_scale == pytest.approx(400.0)

    m.cx, m.cy = 300.0, 220.0
    save_model(str(path), m)
    _, adopted2 = load_model_compat(str(path))
    assert adopted2.center == (300.0, 220.0)
    assert adopted2.center_source == "model-file"
    assert loaded.width == 640


def test_old_project_file_keeps_coordinate_convention():
    """The pre-refactor sample model must resolve to the same convention."""
    path = os.path.join(DATA_DIR, "lens_lca.json")
    model, adopted = load_model_compat(path)
    assert adopted.center == (320.0, 240.0)
    assert adopted.center_source == "default:image-center"
    assert adopted.radius_scale == pytest.approx(400.0)
    mapping = build_channel_mapping(model)
    assert mapping.center == (320.0, 240.0)
    assert mapping.validity_mask.shape == (480, 640)


def test_ui_session_defaults_are_explicit():
    d = ui_session_defaults()
    assert d.reference_channel == "G"
    assert d.fit_center is False
    assert d.detection.min_points == DetectionConfig().min_points
    assert d.detection.search_radius == DetectionConfig().search_radius


def test_adopted_values_lines_mention_convention():
    model = SCENARIOS["center"]
    adopted = adopted_values(model)
    text = "\n".join(adopted.lines())
    assert "principal_point=(320.00, 240.00)" in text
    assert "default:image-center" in text
