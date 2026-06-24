"""Intrinsic-units macro framework (Marshall et al. 2024).

Macro units are defined by sliding-window state mappings over their
micro constituents; macro cause and effect TPMs are built by the
four-step construction (Eqs. 26-40) and analyzed by the IIT 4.0
pipeline exactly as micro systems are. The intrinsic-unit criteria
(Eqs. 15-16) and the bounded grain search (Eq. 19) decide which units
and which grain are intrinsic for a substrate in a state.
"""

from pyphi.macro.criteria import Reason
from pyphi.macro.criteria import UnitVerdict
from pyphi.macro.criteria import constituent_system
from pyphi.macro.criteria import judge_candidate
from pyphi.macro.criteria import unit_integration
from pyphi.macro.search import ComplexesResult
from pyphi.macro.search import DecompositionVerdict
from pyphi.macro.search import EvaluationRecord
from pyphi.macro.search import IntrinsicUnitsResult
from pyphi.macro.search import SearchBounds
from pyphi.macro.search import candidate_mappings
from pyphi.macro.search import competing_systems
from pyphi.macro.search import complexes
from pyphi.macro.search import intrinsic_units
from pyphi.macro.search import is_intrinsic_unit
from pyphi.macro.search import valid_systems
from pyphi.macro.system import MacroSystem
from pyphi.macro.tpm import macro_tpms
from pyphi.macro.units import MacroUnit
from pyphi.macro.units import blackbox
from pyphi.macro.units import coarse_grain
from pyphi.macro.units import micro_unit

__all__ = [
    "ComplexesResult",
    "DecompositionVerdict",
    "EvaluationRecord",
    "IntrinsicUnitsResult",
    "MacroSystem",
    "MacroUnit",
    "Reason",
    "SearchBounds",
    "UnitVerdict",
    "blackbox",
    "candidate_mappings",
    "coarse_grain",
    "competing_systems",
    "complexes",
    "constituent_system",
    "intrinsic_units",
    "is_intrinsic_unit",
    "judge_candidate",
    "macro_tpms",
    "micro_unit",
    "unit_integration",
    "valid_systems",
]
