"""Unit tests on synthetic ground-truth data with known displacements."""

import numpy as np
import pytest

from lca import (
    SyntheticSpec, generate_dataset, detect_control_points, calibrate,
    correct_image, make_identity_model, load_model, save_model,
)
from lca.detection import DetectionConfig


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(SyntheticSpec(colorful=True))


@pytest.fixture(scope="module")
def fitted(dataset):
    spec_w, spec_h = dataset["warped"].shape[1], dataset["warped"].shape[0]
    control, _ = detect_control_points(dataset["warped"], "G")
    model = calibrate(spec_w, spec_h, control, "G")
    return model, control


def test_model_is_position_dependent_not_translation(fitted):
    model, _ = fitted
    # Displacement must vanish at the principal point and grow with radius.
    cx, cy = model.center
    center = np.array([[cx, cy]])
    corner = np.array([[model.width - 1, model.height - 1]])
    for c, ch in model.channels.items():
        d_center = model.displacement(center)[c]
        d_corner = model.displacement(corner)[c]
        assert np.linalg.norm(d_center) < 1e-6
        assert np.linalg.norm(d_corner) > np.linalg.norm(d_center)
        # A pure translation model would have zero radial coefficients.
        assert not (ch.k1 == 0 and ch.k2 == 0)


def test_recovered_displacement_matches_ground_truth(dataset, fitted):
    model, _ = fitted
    gt = dataset["ground_truth"]
    h, w = dataset["warped"].shape[:2]
    grid = np.column_stack([np.tile(np.arange(w), h),
                            np.repeat(np.arange(h), w)])
    inside = model.normalized_radius(grid) <= model.r_norm_max
    for c in model.channels:
        err = np.linalg.norm(
            model.displacement(grid)[c][inside]
            - gt.displacement(grid)[c][inside], axis=1)
        assert np.percentile(err, 95) < 0.5
        assert err.mean() < 0.3


def test_correction_reduces_edge_separation(dataset, fitted):
    model, _ = fitted
    warped = dataset["warped"]
    corrected, valid, _ = correct_image(warped, model)
    cfg = DetectionConfig(bucket_size=24, max_per_bucket=2)
    before, _ = detect_control_points(warped, "G", cfg)
    after, _ = detect_control_points(corrected, "G", cfg)
    for c in model.channels:
        sb = np.median(np.linalg.norm(before[c].displacements, axis=1))
        sa = np.median(np.linalg.norm(after[c].displacements, axis=1))
        assert sa < sb * 0.5


def test_validity_mask_covers_interior_and_flags_extrapolation(fitted):
    model, _ = fitted
    assert 0.0 < model.extrapolation_ratio < 0.5
    h, w = model.height, model.width
    dummy = np.random.default_rng(0).random((h, w, 3))
    _, valid, _ = correct_image(dummy, model)
    # Center is always valid; at least the extrapolation fraction is masked.
    cx, cy = int(w / 2), int(h / 2)
    assert valid[cy, cx]
    assert (~valid).mean() >= model.extrapolation_ratio * 0.9


def test_brightness_change_does_not_move_geometry(dataset, fitted):
    model, _ = fitted
    control_b, _ = detect_control_points(dataset["brightness_variant"], "G")
    h, w = dataset["warped"].shape[:2]
    model_b = calibrate(w, h, control_b, "G")
    for c in model.channels:
        drift = np.max(np.abs(model.channels[c].to_array()
                              - model_b.channels[c].to_array()))
        assert drift < 5e-3


def test_colorful_negative_control_rejected(dataset):
    neg_spec = SyntheticSpec(seed=99, n_rects=0, n_circles=0, colorful=True)
    neg = generate_dataset(neg_spec)
    with pytest.raises(RuntimeError):
        detect_control_points(neg["warped"], "G")


def test_model_roundtrip(tmp_path, fitted):
    model, _ = fitted
    path = tmp_path / "m.json"
    save_model(str(path), model)
    loaded = load_model(str(path))
    assert loaded.width == model.width
    for c in model.channels:
        np.testing.assert_allclose(loaded.channels[c].to_array(),
                                   model.channels[c].to_array())


def test_identity_model_leaves_image_unchanged(dataset):
    warped = dataset["warped"]
    h, w = warped.shape[:2]
    identity = make_identity_model(w, h, "G")
    identity.r_norm_max = 2.0  # trust whole frame for the identity check
    out, valid, _ = correct_image(warped, identity)
    # Ignore a 2-px remap border; cubic identity sampling must be exact there.
    interior = np.zeros((h, w), dtype=bool)
    interior[2:-2, 2:-2] = True
    sel = valid & interior
    np.testing.assert_allclose(out[sel], warped[sel], atol=1e-5)
