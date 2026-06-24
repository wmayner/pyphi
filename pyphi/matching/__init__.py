"""The matching formalism: perception and matching (Mayner, Juel & Tononi)."""

from .differentiation import Differentiation
from .environment import mixture
from .environment import noise
from .environment import point
from .environment import sample
from .environment import segment
from .environment import superpose
from .matching import MatchingAnalysis
from .matching import MatchingResult
from .perception import Perception
from .system import PerceptualSystem
from .triggered_tpm import TriggeredTPM
from .triggered_tpm import build_triggered_tpm
from .triggering import TriggeringCoefficient
from .triggering import triggering_coefficient

__all__ = [
    "Differentiation",
    "MatchingAnalysis",
    "MatchingResult",
    "Perception",
    "PerceptualSystem",
    "TriggeredTPM",
    "TriggeringCoefficient",
    "build_triggered_tpm",
    "mixture",
    "noise",
    "point",
    "sample",
    "segment",
    "superpose",
    "triggering_coefficient",
]
