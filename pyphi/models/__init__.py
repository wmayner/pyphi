# models/__init__.py
"""See |models.system|, |models.mechanism|, and |models.partitions| for documentation.

Attributes:
    Account: Alias for :class:`pyphi.models.actual_causation.Account`.
    AcRepertoireIrreducibilityAnalysis: Alias for
     :class:`pyphi.models.actual_causation.AcRepertoireIrreducibilityAnalysis`.
    AcSystemIrreducibilityAnalysis: Alias for
        :class:`pyphi.models.actual_causation.AcSystemIrreducibilityAnalysis`.
    DirectedJointPartition: Alias for
        :class:`pyphi.models.partitions.DirectedJointPartition`.
    JointBipartition: Alias for :class:`pyphi.models.partitions.JointBipartition`.
    CausalLink: Alias for :class:`pyphi.models.actual_causation.CausalLink`.
    CauseEffectStructure: Alias for
        :class:`pyphi.models.ces.CauseEffectStructure` — the
        distinctions-plus-relations object specified by any candidate
        system (Albantakis et al. 2023). When the candidate is a
        complex, this is what the IIT 4.0 paper calls a *Φ-structure*.
    Distinctions: Alias for
        :class:`pyphi.models.distinctions.Distinctions` — the bag of
        distinctions, no relations.
    Concept: Alias for :class:`pyphi.models.distinction.Distinction`
        — IIT 3.0 paper terminology for the same object.
    Distinction: Alias for :class:`pyphi.models.distinction.Distinction`.
    DirectedAccount: Alias for
        :class:`pyphi.models.actual_causation.DirectedAccount`.
    MaximallyIrreducibleCause: Alias for
        :class:`pyphi.models.mechanism.MaximallyIrreducibleCause`.
    MaximallyIrreducibleEffect: Alias for
        :class:`pyphi.models.mechanism.MaximallyIrreducibleEffect`.
    MaximallyIrreducibleCauseOrEffect: Alias for
        :class:`pyphi.models.mechanism.MaximallyIrreducibleCauseOrEffect`.
    Part: Alias for :class:`pyphi.models.partitions.Part`.
    RepertoireIrreducibilityAnalysis: Alias for
        :class:`pyphi.models.mechanism.RepertoireIrreducibilityAnalysis`.
    IIT3SystemIrreducibilityAnalysis: Alias for
        :class:`pyphi.models.sia.IIT3SystemIrreducibilityAnalysis` — the
        IIT 3.0 result type. The IIT 4.0 result type with the same role
        lives in :mod:`pyphi.formalism.iit4` and is called
        ``SystemIrreducibilityAnalysis`` there.
"""

from .actual_causation import Account
from .actual_causation import AcRepertoireIrreducibilityAnalysis
from .actual_causation import AcSystemIrreducibilityAnalysis
from .actual_causation import CausalLink
from .actual_causation import DirectedAccount
from .actual_causation import Event
from .actual_causation import _null_ac_ria
from .actual_causation import _null_ac_sia
from .ces import CauseEffectStructure
from .complex import Complex
from .complex import ExcludedCandidate
from .distinction import Concept
from .distinction import Distinction
from .distinctions import Distinctions
from .distinctions import ResolvedDistinctions
from .distinctions import UnresolvedDistinctions
from .distinctions import _null_ces
from .explanation import Explanation
from .explanation import Finding
from .explanation import NullResultReason
from .mice import MaximallyIrreducibleCause
from .mice import MaximallyIrreducibleCauseOrEffect
from .mice import MaximallyIrreducibleEffect
from .partitions import CompleteEdgeCut
from .partitions import DirectedBipartition
from .partitions import DirectedJointPartition
from .partitions import DirectedSetPartition
from .partitions import EdgeCut
from .partitions import JointBipartition
from .partitions import JointPartition
from .partitions import JointTripartition
from .partitions import NullCut
from .partitions import Part
from .protocols import AcSIAInterface
from .protocols import CauseEffectStructureInterface
from .protocols import SIAInterface
from .ria import RepertoireIrreducibilityAnalysis
from .ria import _null_ria
from .sia import IIT3SystemIrreducibilityAnalysis
from .sia import _null_sia
from .state_specification import DistinctionPhiNormalizationRegistry
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
    "CompleteEdgeCut",
    "Complex",
    "Concept",
    "DirectedAccount",
    "DirectedBipartition",
    "DirectedJointPartition",
    "DirectedSetPartition",
    "Distinction",
    "DistinctionPhiNormalizationRegistry",
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
    "SIAInterface",
    "StateSpecification",
    "SystemStateSpecification",
    "UnitState",
    "UnresolvedDistinctions",
    "_null_ac_ria",
    "_null_ac_sia",
    "_null_ces",
    "_null_ria",
    "_null_sia",
    "distinction_phi_normalizations",
    "normalization_factor",
]
