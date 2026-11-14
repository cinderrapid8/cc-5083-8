"""Explicit compatibility entry for parameters the UI used to fill silently.

Historically the workbench completed several parameters from implicit
defaults: reference channel "G", ``fit_center=False``, ``DetectionConfig()``
defaults, and -- inside the model file -- a null principal point meaning
"image center". Opening an old project must not silently switch to another
coordinate convention, so loading goes through :func:`load_model_compat`,
which returns the model together with an :class:`AdoptedValues` record of
every value actually adopted. The GUI logs that record; the CLI prints it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .calibrate import load_model
from .detection import DetectionConfig
from .model import LcaModel


@dataclass(frozen=True)
class SessionDefaults:
    """Values the GUI previously supplied through hard-coded defaults."""

    reference_channel: str
    fit_center: bool
    detection: DetectionConfig


def ui_session_defaults() -> SessionDefaults:
    """The single place where the workbench's implicit defaults live."""
    return SessionDefaults(
        reference_channel="G",
        fit_center=False,
        detection=DetectionConfig())


@dataclass
class AdoptedValues:
    """Record of the coordinate convention a model is interpreted with."""

    reference_channel: str
    center: Tuple[float, float]
    center_source: str  # "model-file" | "default:image-center"
    radius_scale: float
    r_norm_max: float
    fitted_center: bool
    notes: str = ""

    def lines(self) -> List[str]:
        return [
            f"reference_channel={self.reference_channel}",
            f"principal_point=({self.center[0]:.2f}, {self.center[1]:.2f})"
            f" [{self.center_source}]",
            f"radius_scale={self.radius_scale:.3f} (half diagonal)",
            f"r_norm_max={self.r_norm_max:.4f}",
            f"fitted_center={self.fitted_center}",
        ]


def adopted_values(model: LcaModel) -> AdoptedValues:
    """Resolve every convention-relevant value of a model, explicitly."""
    cx, cy = model.center
    if model.cx is None and model.cy is None:
        source = "default:image-center"
    else:
        source = "model-file"
    return AdoptedValues(
        reference_channel=model.reference_channel,
        center=(float(cx), float(cy)),
        center_source=source,
        radius_scale=float(model.radius_scale),
        r_norm_max=float(model.r_norm_max),
        fitted_center=bool(model.fitted_center),
        notes=model.notes)


def load_model_compat(path: str) -> Tuple[LcaModel, AdoptedValues]:
    """Load a pre-refactor (lca-model-v1) file and record adopted values."""
    model = load_model(path)
    return model, adopted_values(model)
