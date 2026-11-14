# Lateral Chromatic Aberration (LCA) Correction

Lens lateral chromatic aberration calibration and correction: sub-pixel
shared-edge control points are fitted to a constrained radial + thin-prism
displacement model, and all channels are resampled onto one common frame.

## Graphical entry (image recovery window)

```
python run_gui.py
# or
python -m lca.gui
```

The workbench window hosts the lens LCA operation end to end: open a
calibration target, detect control points, edit them (toggle / add / remove),
fit the model, correct, and batch apply. After control-point edits and
correction it still displays the real channel edges, per-point outliers and
normal-residual scatter, plus the before/after channel edge-position
separation in the running-status log.

## Pipeline layers and coordinate conventions

| layer | module | input -> output | coordinates |
|---|---|---|---|
| control-point extraction | `lca.detection` | image -> `ControlPoints` | observed (source) positions per channel |
| displacement fitting | `lca.calibrate` | `ControlPoints` -> `LcaModel` | displacement defined on reference-frame (target) positions |
| sampling-coordinate generation | `lca.geometry` | `LcaModel` -> `ChannelMapping` | target grid -> per-channel source maps + validity mask |
| color resampling | `lca.sampling` | image + `ChannelMapping` -> corrected image | samples source channels, writes target grid |

* *target* coordinates: the shared reference-frame pixel grid of the output.
* *source* coordinates: where each observed input channel is sampled from.

The geometry layer reads the model only — never image colors — so a
`ChannelMapping` can be inspected and diffed on its own
(`mapping.describe()`). The GUI preview ("Correct") and the batch writer
("Batch apply…", CLI `apply`) share one mapping object built once per model
via `build_channel_mapping`, so the center/radius conversion happens in
exactly one place. `lca.correct.correct_image` remains as the compatibility
composition of the two layers.

## Compatibility entry (old projects)

Model files keep the `lca-model-v1` format. Parameters the UI previously
filled from implicit defaults (reference channel `G`, `fit_center=False`,
`DetectionConfig()` defaults, null principal point = image center) now go
through explicit entry points and are recorded:

* `lca.compat.load_model_compat(path)` -> `(model, AdoptedValues)` — the GUI
  logs every adopted value (`compat: adopted ...`) when a model is opened;
* `lca.compat.ui_session_defaults()` — the single source of the workbench's
  session defaults, logged at startup;
* CLI `apply` prints the adopted values before processing.

An old project therefore keeps its original coordinate convention; nothing
is silently reinterpreted.

## Command line (auxiliary)

The CLI only assists the lens LCA workflow:

```
python -m lca.cli calibrate --image target.png --model lens_lca.json [--preview out.png]
python -m lca.cli apply --model lens_lca.json --inputs a.png b.png --outdir out
python -m lca.cli selftest --outdir data/selftest
```

## Tests and refactor evidence

```
python -m pytest tests/
```

* `tests/test_lca.py` — model semantics on synthetic ground truth.
* `tests/test_geometry.py` — geometry/sampling layers: eccentric optical
  axis, non-square frames, peripheral sampling gaps, pixel-by-pixel map and
  validity-mask checks, independent synthetic-displacement recovery, and
  preview/batch sharing of one mapping object.
* `tests/test_gui.py` — headless run of the full workbench chain.
* `data/refactor_diff/` — before/after diff samples of the layer split
  (corrected images, validity masks, `report.json`: all four scenarios are
  pixel-identical). Regenerate with
  `python scripts/capture_diff_sample.py capture --out <dir>` on each side
  of the refactor and `... compare --before A --after B ...`.
