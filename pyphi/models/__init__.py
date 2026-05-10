# models/__init__.py
"""See |models.system|, |models.mechanism|, and |models.cuts| for documentation.

Attributes:
    Account: Alias for :class:`pyphi.models.actual_causation.Account`.
    AcRepertoireIrreducibilityAnalysis: Alias for
     :class:`pyphi.models.actual_causation.AcRepertoireIrreducibilityAnalysis`.
    AcSystemIrreducibilityAnalysis: Alias for
        :class:`pyphi.models.actual_causation.AcSystemIrreducibilityAnalysis`.
    ActualCut: Alias for :class:`pyphi.models.cuts.ActualCut`.
    Bipartition: Alias for :class:`pyphi.models.cuts.Bipartition`.
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
    Part: Alias for :class:`pyphi.models.cuts.Part`.
    RepertoireIrreducibilityAnalysis: Alias for
        :class:`pyphi.models.mechanism.RepertoireIrreducibilityAnalysis`.
    SystemIrreducibilityAnalysis: Alias for
        :class:`pyphi.models.system.SystemIrreducibilityAnalysis`.
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
from .cuts import ActualCut
from .cuts import Bipartition
from .cuts import KCut
from .cuts import KPartition
from .cuts import NullCut
from .cuts import Part
from .cuts import SystemPartition
from .cuts import Tripartition
from .distinction import Concept
from .distinction import Distinction
from .distinctions import Distinctions
from .distinctions import ResolvedDistinctions
from .distinctions import UnresolvedDistinctions
from .distinctions import _null_ces
from .mice import MaximallyIrreducibleCause
from .mice import MaximallyIrreducibleCauseOrEffect
from .mice import MaximallyIrreducibleEffect
from .ria import RepertoireIrreducibilityAnalysis
from .ria import ShortCircuitConditions
from .ria import _null_ria
from .sia import SystemIrreducibilityAnalysis
from .sia import _null_sia
from .state_specification import DistinctionPhiNormalizationRegistry
from .state_specification import StateSpecification
from .state_specification import SystemStateSpecification
from .state_specification import UnitState
from .state_specification import distinction_phi_normalizations
from .state_specification import normalization_factor

__all__ = [
    "AcRepertoireIrreducibilityAnalysis",
    "AcSystemIrreducibilityAnalysis",
    "Account",
    "ActualCut",
    "Bipartition",
    "CausalLink",
    "CauseEffectStructure",
    "Concept",
    "DirectedAccount",
    "Distinction",
    "DistinctionPhiNormalizationRegistry",
    "Distinctions",
    "Event",
    "KCut",
    "KPartition",
    "MaximallyIrreducibleCause",
    "MaximallyIrreducibleCauseOrEffect",
    "MaximallyIrreducibleEffect",
    "NullCut",
    "Part",
    "RepertoireIrreducibilityAnalysis",
    "ResolvedDistinctions",
    "ShortCircuitConditions",
    "StateSpecification",
    "SystemIrreducibilityAnalysis",
    "SystemPartition",
    "SystemStateSpecification",
    "Tripartition",
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
