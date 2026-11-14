"""Lateral chromatic aberration correction package."""

from .model import LcaModel, ChannelDisplacement, CHANNELS, make_identity_model
from .detection import detect_control_points, ControlPoints, DetectionConfig
from .calibrate import calibrate, save_model, load_model, CalibrationError
from .geometry import ChannelMapping, build_channel_mapping
from .sampling import resample_image
from .correct import correct_image
from .compat import (
    AdoptedValues, SessionDefaults, adopted_values, load_model_compat,
    ui_session_defaults,
)
from .synthetic import generate_dataset, SyntheticSpec

__all__ = [
    "LcaModel", "ChannelDisplacement", "CHANNELS", "make_identity_model",
    "detect_control_points", "ControlPoints", "DetectionConfig",
    "calibrate", "save_model", "load_model", "CalibrationError",
    "ChannelMapping", "build_channel_mapping",
    "resample_image",
    "correct_image",
    "AdoptedValues", "SessionDefaults", "adopted_values",
    "load_model_compat", "ui_session_defaults",
    "generate_dataset", "SyntheticSpec",
]
