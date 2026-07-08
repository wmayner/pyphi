"""The matching formalism: perception and matching.

Implements the extension of IIT that relates a complex's intrinsic
cause-effect structure to the extrinsic stimuli that trigger it. A stimulus
acts as a trigger for the complex's state; the *perception* is the portion of
the Φ-structure that stimulus triggers, and *matching* measures how much more
perceptual differentiation a complex's environment evokes than random noise.

Equation numbers throughout this package refer to:

Mayner WGP, Juel BE, Tononi G (2024). Intrinsic meaning, perception, and
matching. arXiv:2412.21111.
"""

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
