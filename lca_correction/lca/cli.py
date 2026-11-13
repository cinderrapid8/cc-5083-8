"""Command-line interface: calibrate, batch-apply a model, and run the
synthetic self-test.

Examples
--------
python -m lca.cli calibrate --image target.png --model lens_lca.json
python -m lca.cli apply --model lens_lca.json --inputs a.png b.png --outdir out
python -m lca.cli selftest --outdir data/selftest
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np

from . import imageio
from .calibrate import calibrate, save_model, load_model
from .correct import correct_image
from .detection import detect_control_points, DetectionConfig
from .synthetic import SyntheticSpec, generate_dataset


def _print_model_report(model) -> None:
    print(f"frame: {model.width}x{model.height}, reference: "
          f"{model.reference_channel}")
    cx, cy = model.center
    print(f"principal point: ({cx:.2f}, {cy:.2f})"
          f"{' (fitted)' if model.fitted_center else ' (image center)'}")
    print(f"control points: {model.n_control}, normalized radial coverage: "
          f"{model.r_norm_min:.3f} .. {model.r_norm_max:.3f}")
    print(f"extrapolation risk: {model.extrapolation_ratio * 100:.1f}% "
          "of frame lies outside calibrated coverage")
    for c, ch in model.channels.items():
        print(f"  {c}: k1={ch.k1:+.5f} k2={ch.k2:+.5f} "
              f"p1={ch.p1:+.5f} p2={ch.p2:+.5f} | "
              f"rmse={model.rmse_px.get(c, float('nan')):.3f}px "
              f"max={model.max_abs_residual_px.get(c, float('nan')):.3f}px")


def cmd_calibrate(args) -> int:
    rgb = imageio.load_image_rgb(args.image)
    h, w = rgb.shape[:2]
    cfg = DetectionConfig(bucket_size=args.bucket,
                          max_per_bucket=args.max_per_bucket)
    control, stats = detect_control_points(rgb, args.reference, cfg)
    print("detection:", stats)
    model = calibrate(w, h, control, args.reference,
                      fit_center=args.fit_center)
    save_model(args.model, model)
    _print_model_report(model)
    print(f"model saved: {args.model}")
    if args.preview:
        corrected, valid, _ = correct_image(rgb, model)
        imageio.save_image_rgb(args.preview, corrected)
        imageio.save_mask(os.path.splitext(args.preview)[0] + "_mask.png",
                          valid)
        print("preview + validity mask written")
    return 0


def cmd_apply(args) -> int:
    model = load_model(args.model)
    os.makedirs(args.outdir, exist_ok=True)
    files = []
    for pat in args.inputs:
        files.extend(sorted(glob.glob(pat)))
    if not files:
        print("no input images matched", file=sys.stderr)
        return 2
    for path in files:
        rgb = imageio.load_image_rgb(path)
        if rgb.shape[:2] != (model.height, model.width):
            print(f"skip {path}: size mismatch", file=sys.stderr)
            continue
        corrected, valid, _ = correct_image(rgb, model)
        base = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(args.outdir, base + "_lca.png")
        imageio.save_image_rgb(out_path, corrected)
        imageio.save_mask(os.path.join(args.outdir, base + "_validmask.png"),
                          valid)
        print(f"corrected {path} -> {out_path} "
              f"(invalid {100 * (1 - valid.mean()):.1f}%)")
    return 0


def cmd_selftest(args) -> int:
    os.makedirs(args.outdir, exist_ok=True)
    spec = SyntheticSpec(colorful=True)
    ds = generate_dataset(spec)
    warped = ds["warped"]
    gt = ds["ground_truth"]
    imageio.save_image_rgb(os.path.join(args.outdir, "01_base.png"),
                           ds["base"])
    imageio.save_image_rgb(os.path.join(args.outdir, "02_warped.png"), warped)

    control, stats = detect_control_points(warped, gt.reference_channel)
    print("detection stats:", stats)
    model = calibrate(spec.width, spec.height, control,
                      gt.reference_channel)
    save_model(os.path.join(args.outdir, "fitted_model.json"), model)
    _print_model_report(model)

    # 1. Displacement recovery vs ground truth on a dense grid.
    grid = np.column_stack([np.tile(np.arange(spec.width), spec.height),
                            np.repeat(np.arange(spec.height), spec.width)])
    inside = model.normalized_radius(grid) <= model.r_norm_max
    max_err = 0.0
    for c in model.channels:
        pred = model.displacement(grid)[c][inside]
        truth = gt.displacement(grid)[c][inside]
        err = np.linalg.norm(pred - truth, axis=1)
        max_err = max(max_err, float(err.max()))
        print(f"  {c} displacement error: mean={err.mean():.4f}px "
              f"p95={np.percentile(err, 95):.4f}px max={err.max():.4f}px")

    # 2. Round-trip detail recovery: corrected image vs original base.
    #    Measured on the edge bands where channel mis-registration is visible,
    #    since smooth interiors are unaffected by geometry anyway.
    corrected, valid, _ = correct_image(warped, model)
    imageio.save_image_rgb(os.path.join(args.outdir, "03_corrected.png"),
                           corrected)
    edge_band = cv2.dilate(
        cv2.Canny((ds["base"].mean(axis=2) * 255).astype(np.uint8),
                  40, 120), np.ones((5, 5), np.uint8)).astype(bool)
    measure = valid & edge_band
    if measure.sum() < 1000:
        raise RuntimeError("validity mask unexpectedly empty")
    diff_before = np.abs(warped[measure] - ds["base"][measure]).mean()
    diff_after = np.abs(corrected[measure] - ds["base"][measure]).mean()
    print(f"edge-band abs error vs ground truth: before={diff_before:.4f} "
          f"after={diff_after:.4f} on {measure.sum()} px")

    # 2b. Independent geometric check with the ground-truth model itself,
    #     which must drive channel edge separation back toward zero.
    gt_model = ds["ground_truth"]
    gt_model.r_norm_max = model.r_norm_max
    corrected_gt, valid_gt, _ = correct_image(warped, gt_model)
    sep_cfg = DetectionConfig(bucket_size=24, max_per_bucket=2)
    cps_before, _ = detect_control_points(warped, gt.reference_channel, sep_cfg)
    cps_after, _ = detect_control_points(corrected_gt, gt.reference_channel,
                                         sep_cfg)
    sep_before = np.mean([
        np.median(np.linalg.norm(cps_before[c].displacements, axis=1))
        for c in cps_before])
    sep_after = np.mean([
        np.median(np.linalg.norm(cps_after[c].displacements, axis=1))
        for c in cps_after])
    print(f"channel edge separation (GT model): before={sep_before:.3f}px "
          f"after={sep_after:.3f}px")

    # 3. Brightness invariance: fit again on the gain/bias variant.
    control_b, _ = detect_control_points(ds["brightness_variant"],
                                         gt.reference_channel)
    model_b = calibrate(spec.width, spec.height, control_b,
                        gt.reference_channel)
    drift = []
    for c in model.channels:
        a = model.channels[c].to_array()
        b = model_b.channels[c].to_array()
        drift.append(np.max(np.abs(a - b)))
    print("brightness-variant coefficient drift (max):",
          f"{max(drift):.2e}")

    # 4. Negative control: colorful-only scene must yield few/no points.
    neg_spec = SyntheticSpec(seed=99, n_rects=0, n_circles=0, colorful=True)
    neg = generate_dataset(neg_spec)
    try:
        neg_control, neg_stats = detect_control_points(neg["warped"],
                                                       gt.reference_channel)
        n_neg = sum(len(v) for v in neg_control.values())
        print("negative control points (expect ~0):", n_neg)
    except RuntimeError as exc:
        print("negative control correctly rejected:", exc)
        n_neg = 0

    ok = (max_err < 1.0 and diff_after < diff_before
          and sep_after < 0.25 and sep_after < sep_before * 0.5
          and max(drift) < 5e-3)
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Lateral chromatic aberration tool")
    sub = p.add_subparsers(dest="command", required=True)

    pc = sub.add_parser("calibrate", help="fit an LCA model from a target image")
    pc.add_argument("--image", required=True)
    pc.add_argument("--model", required=True)
    pc.add_argument("--reference", default="G", choices=list("RGB"))
    pc.add_argument("--fit-center", action="store_true")
    pc.add_argument("--bucket", type=int, default=12)
    pc.add_argument("--max-per-bucket", type=int, default=4)
    pc.add_argument("--preview", default=None)
    pc.set_defaults(func=cmd_calibrate)

    pa = sub.add_parser("apply", help="batch apply a saved model")
    pa.add_argument("--model", required=True)
    pa.add_argument("--inputs", nargs="+", required=True)
    pa.add_argument("--outdir", required=True)
    pa.set_defaults(func=cmd_apply)

    ps = sub.add_parser("selftest", help="synthetic ground-truth self-test")
    ps.add_argument("--outdir", default="data/selftest")
    ps.set_defaults(func=cmd_selftest)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
