"""The matching formalism: perception and matching (Mayner, Juel & Tononi)."""

from .system import PerceptualSystem
from .triggered_tpm import TriggeredTPM
from .triggered_tpm import build_triggered_tpm
from .triggering import TriggeringCoefficient
from .triggering import triggering_coefficient

__all__ = [
    "PerceptualSystem",
    "TriggeredTPM",
    "TriggeringCoefficient",
    "build_triggered_tpm",
    "triggering_coefficient",
]
