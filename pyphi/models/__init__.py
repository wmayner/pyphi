# models/__init__.py
"""See :mod:`pyphi.models.sia`, :mod:`pyphi.models.mice`, and
:mod:`pyphi.models.partitions` for documentation.

Attributes
----------
Account
    Alias for :class:`pyphi.models.actual_causation.Account`.
AcRepertoireIrreducibilityAnalysis
    Alias for
    :class:`pyphi.models.actual_causation.AcRepertoireIrreducibilityAnalysis`.
AcSystemIrreducibilityAnalysis
    Alias for
    :class:`pyphi.models.actual_causation.AcSystemIrreducibilityAnalysis`.
DirectedJointPartition
    Alias for :class:`pyphi.models.partitions.DirectedJointPartition`.
JointBipartition
    Alias for :class:`pyphi.models.partitions.JointBipartition`.
CausalLink
    Alias for :class:`pyphi.models.actual_causation.CausalLink`.
CauseEffectStructure
    Alias for :class:`pyphi.models.ces.CauseEffectStructure` — the
    distinctions-plus-relations object specified by any candidate system
    (Albantakis et al. 2023). When the candidate is a complex, this is what
    the IIT 4.0 paper calls a *Φ-structure*.
Distinctions
    Alias for :class:`pyphi.models.distinctions.Distinctions` — the bag of
    distinctions, no relations.
Distinction
    Alias for :class:`pyphi.models.distinction.Distinction`.
DirectedAccount
    Alias for :class:`pyphi.models.actual_causation.DirectedAccount`.
MaximallyIrreducibleCause
    Alias for :class:`pyphi.models.mice.MaximallyIrreducibleCause`.
MaximallyIrreducibleEffect
    Alias for :class:`pyphi.models.mice.MaximallyIrreducibleEffect`.
MaximallyIrreducibleCauseOrEffect
    Alias for :class:`pyphi.models.mice.MaximallyIrreducibleCauseOrEffect`.
Part
    Alias for :class:`pyphi.models.partitions.Part`.
RepertoireIrreducibilityAnalysis
    Alias for :class:`pyphi.models.ria.RepertoireIrreducibilityAnalysis`.
IIT3SystemIrreducibilityAnalysis
    Alias for :class:`pyphi.models.sia.IIT3SystemIrreducibilityAnalysis` — the
    IIT 3.0 result type, used to reproduce published results. The system
    irreducibility analysis of IIT is ``SystemIrreducibilityAnalysis`` in
    :mod:`pyphi.formalism.iit4`.
"""

from .actual_causation import Account
from .actual_causation import AcRepertoireIrreducibilityAnalysis
from .actual_causation import AcSystemIrreducibilityAnalysis
from .actual_causation import CausalLink
from .actual_causation import DirectedAccount
from .actual_causation import Event
from .actual_causation import _null_ac_ria as _null_ac_ria
from .actual_causation import _null_ac_sia as _null_ac_sia
from .ces import CauseEffectStructure
from .complex import Complex
from .complex import ExcludedCandidate
from .diff import Change
from .diff import ResultDiff
from .distinction import Distinction
from .distinctions import Distinctions
from .distinctions import ResolvedDistinctions
from .distinctions import UnresolvedDistinctions
from .distinctions import _null_ces as _null_ces
from .explanation import Explanation
from .explanation import Finding
from .explanation import NullResultReason
from .mice import MaximallyIrreducibleCause
from .mice import MaximallyIrreducibleCauseOrEffect
from .mice import MaximallyIrreducibleEffect
from .partitions import DirectedBipartition
from .partitions import DirectedJointPartition
from .partitions import DirectedSetPartition
from .partitions import EdgeCut
from .partitions import JointBipartition
from .partitions import JointPartition
from .partitions import JointTripartition
from .partitions import NullCut
from .partitions import Part
from .partitions import TotalCut
from .protocols import AcSIAInterface
from .protocols import CauseEffectStructureInterface
from .protocols import SIAInterface
from .ria import RepertoireIrreducibilityAnalysis
from .ria import _null_ria as _null_ria
from .sia import IIT3SystemIrreducibilityAnalysis
from .sia import _null_sia as _null_sia
from .state_specification import (
    DistinctionPhiNormalizationRegistry as DistinctionPhiNormalizationRegistry,
)
from .state_specification import StateSpecification
from .state_specification import SystemStateSpecification
from .state_specification import UnitState
from .state_specification import distinction_phi_normalizations
from .state_specification import normalization_factor

__all__ = [
    "AcRepertoireIrreducibilityAnalysis",
    "AcSIAInterface",
    "AcSystemIrreducibilityAnalysis",
    "Account",
    "CausalLink",
    "CauseEffectStructure",
    "CauseEffectStructureInterface",
    "Change",
    "Complex",
    "DirectedAccount",
    "DirectedBipartition",
    "DirectedJointPartition",
    "DirectedSetPartition",
    "Distinction",
    "Distinctions",
    "EdgeCut",
    "Event",
    "ExcludedCandidate",
    "Explanation",
    "Finding",
    "IIT3SystemIrreducibilityAnalysis",
    "JointBipartition",
    "JointPartition",
    "JointTripartition",
    "MaximallyIrreducibleCause",
    "MaximallyIrreducibleCauseOrEffect",
    "MaximallyIrreducibleEffect",
    "NullCut",
    "NullResultReason",
    "Part",
    "RepertoireIrreducibilityAnalysis",
    "ResolvedDistinctions",
    "ResultDiff",
    "SIAInterface",
    "StateSpecification",
    "SystemStateSpecification",
    "TotalCut",
    "UnitState",
    "UnresolvedDistinctions",
    "distinction_phi_normalizations",
    "normalization_factor",
]


def __getattr__(name: str):
    message = f"module {__name__!r} has no attribute {name!r}"
    if name == "Concept":
        message += ". PyPhi 2.0 renamed it `Distinction`"
    raise AttributeError(message)
