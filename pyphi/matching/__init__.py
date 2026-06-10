"""The matching formalism: perception and matching (Mayner, Juel & Tononi)."""

from .differentiation import Differentiation
from .perception import Perception
from .system import PerceptualSystem
from .triggered_tpm import TriggeredTPM
from .triggered_tpm import build_triggered_tpm
from .triggering import TriggeringCoefficient
from .triggering import triggering_coefficient

__all__ = [
    "Differentiation",
    "Perception",
    "PerceptualSystem",
    "TriggeredTPM",
    "TriggeringCoefficient",
    "build_triggered_tpm",
    "triggering_coefficient",
]
