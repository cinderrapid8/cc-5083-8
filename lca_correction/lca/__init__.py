"""Lateral chromatic aberration correction package."""

from .model import LcaModel, ChannelDisplacement, CHANNELS, make_identity_model
from .detection import detect_control_points, ControlPoints, DetectionConfig
from .calibrate import calibrate, save_model, load_model, CalibrationError
from .correct import correct_image
from .synthetic import generate_dataset, SyntheticSpec

__all__ = [
    "LcaModel", "ChannelDisplacement", "CHANNELS", "make_identity_model",
    "detect_control_points", "ControlPoints", "DetectionConfig",
    "calibrate", "save_model", "load_model", "CalibrationError",
    "correct_image",
    "generate_dataset", "SyntheticSpec",
]
