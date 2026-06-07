"""The matching formalism: perception and matching (Mayner, Juel & Tononi)."""

from .system import PerceptualSystem
from .triggered_tpm import TriggeredTPM
from .triggered_tpm import build_triggered_tpm

__all__ = ["PerceptualSystem", "TriggeredTPM", "build_triggered_tpm"]
