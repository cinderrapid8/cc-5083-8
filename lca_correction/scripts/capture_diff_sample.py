"""Capture before/after diff samples for the geometry/sampling refactor.

The same script runs against the old and the new code because it only uses
the stable public API (``correct_image``). Typical flow::

    # on the pre-refactor commit:
    python scripts/capture_diff_sample.py capture --out /tmp/diff_before
    # on the refactored commit:
    python scripts/capture_diff_sample.py capture --out /tmp/diff_after
    python scripts/capture_diff_sample.py compare \
        --before /tmp/diff_before --after /tmp/diff_after \
        --report data/refactor_diff/report.json --png-dir data/refactor_diff

Scenarios cover the cases called out for review: default image-center
principal point, an eccentric optical axis, a non-square frame, and a
peripheral sampling gap (calibrated radius much smaller than the corner
radius). Channel maps and the validity mask are compared pixel by pixel.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lca import imageio  # noqa: E402
from lca.correct import correct_image  # noqa: E402
from lca.detection import detect_control_points  # noqa: E402
from lca.calibrate import calibrate  # noqa: E402
from lca.synthetic import SyntheticSpec, generate_dataset  # noqa: E402

SCENARIOS = ("standard", "eccentric", "nonsquare", "peripheral_gap")


def _fitted(spec: SyntheticSpec):
    ds = generate_dataset(spec)
    control, _ = detect_control_points(ds["warped"], "G")
    model = calibrate(spec.width, spec.height, control, "G")
    return ds, model


def _scenarios():
    """Yield (name, image, model) pairs; deterministic across runs."""
    ds1, m1 = _fitted(SyntheticSpec(colorful=True))  # 640x480
    yield "standard", ds1["warped"], m1

    m2 = copy.deepcopy(m1)
    m2.cx, m2.cy = 300.25, 219.5  # eccentric optical axis
    yield "eccentric", ds1["warped"], m2

    ds3, m3 = _fitted(SyntheticSpec(width=400, height=240, seed=11,
                                    colorful=True))
    yield "nonsquare", ds3["warped"], m3

    m4 = copy.deepcopy(m1)
    m4.r_norm_max = 0.45  # peripheral sampling gap: corners uncalibrated
    yield "peripheral_gap", ds1["warped"], m4


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(__file__)).decode().strip()
    except Exception:
        return "unknown"


def capture(out: str) -> None:
    os.makedirs(out, exist_ok=True)
    meta = {"git_rev": _git_rev(), "scenarios": {}}
    for name, img, model in _scenarios():
        corrected, valid, maps = correct_image(img, model)
        arrays = {"corrected": corrected.astype(np.float32),
                  "valid": valid}
        for c, (mx, my) in maps.items():
            arrays[f"map_{c}_x"] = mx
            arrays[f"map_{c}_y"] = my
        np.savez_compressed(os.path.join(out, f"{name}.npz"), **arrays)
        meta["scenarios"][name] = {
            "frame": [int(model.width), int(model.height)],
            "center": [float(model.center[0]), float(model.center[1])],
            "radius_scale": float(model.radius_scale),
            "r_norm_max": float(model.r_norm_max),
            "invalid_fraction": float(1.0 - valid.mean()),
        }
        print(f"captured {name}: invalid={100*(1-valid.mean()):.1f}%")
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def compare(before: str, after: str, report: str, png_dir: str) -> int:
    os.makedirs(os.path.dirname(report), exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    with open(os.path.join(before, "meta.json"), encoding="utf-8") as f:
        meta_b = json.load(f)
    with open(os.path.join(after, "meta.json"), encoding="utf-8") as f:
        meta_a = json.load(f)
    result = {"before_rev": meta_b["git_rev"], "after_rev": meta_a["git_rev"],
              "scenarios": {}}
    worst = 0.0
    for name in SCENARIOS:
        b = np.load(os.path.join(before, f"{name}.npz"))
        a = np.load(os.path.join(after, f"{name}.npz"))
        entry = {
            "frame": meta_a["scenarios"][name]["frame"],
            "center": meta_a["scenarios"][name]["center"],
            "r_norm_max": meta_a["scenarios"][name]["r_norm_max"],
            "corrected_max_abs_diff": float(
                np.max(np.abs(b["corrected"] - a["corrected"]))),
            "validity_mask_mismatches": int(
                np.count_nonzero(b["valid"] != a["valid"])),
            "maps_max_abs_diff": {},
        }
        for key in b.files:
            if key.startswith("map_"):
                d = float(np.max(np.abs(b[key] - a[key])))
                entry["maps_max_abs_diff"][key] = d
                worst = max(worst, d)
        worst = max(worst, entry["corrected_max_abs_diff"])
        result["scenarios"][name] = entry
        # Committed sample images come from the refactored (after) side.
        imageio.save_image_rgb(
            os.path.join(png_dir, f"{name}_corrected.png"),
            a["corrected"].astype(np.float64))
        imageio.save_mask(
            os.path.join(png_dir, f"{name}_validmask.png"),
            a["valid"])
        print(f"{name}: corrected max|d|={entry['corrected_max_abs_diff']}"
              f" mask mismatches={entry['validity_mask_mismatches']}")
    result["identical"] = bool(worst == 0.0)
    with open(report, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"report written: {report} (identical={result['identical']})")
    return 0 if result["identical"] else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    pc = sub.add_parser("capture")
    pc.add_argument("--out", required=True)
    pm = sub.add_parser("compare")
    pm.add_argument("--before", required=True)
    pm.add_argument("--after", required=True)
    pm.add_argument("--report", required=True)
    pm.add_argument("--png-dir", required=True)
    args = p.parse_args(argv)
    if args.command == "capture":
        capture(args.out)
        return 0
    return compare(args.before, args.after, args.report, args.png_dir)


if __name__ == "__main__":
    raise SystemExit(main())
