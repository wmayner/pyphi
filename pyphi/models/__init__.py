#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# models/__init__.py

"""
See |models.subsystem|, |models.mechanism|, and |models.cuts| for
documentation.

Attributes:
    SystemIrreducibilityAnalysis: Alias for
        :class:`pyphi.models.subsystem.SystemIrreducibilityAnalysis`.
    RepertoireIrreducibilityAnalysis: Alias for
        :class:`pyphi.models.mechanism.RepertoireIrreducibilityAnalysis`.
    MaximallyIrreducibleCause: Alias for
        :class:`pyphi.models.mechanism.MaximallyIrreducibleCause`.
    MaximallyIrreducibleEffect: Alias for
        :class:`pyphi.models.mechanism.MaximallyIrreducibleEffect`.
    MaximallyIrreducibleCauseOrEffect: Alias for
        :class:`pyphi.models.mechanism.MaximallyIrreducibleCauseOrEffect`.
    Concept: Alias for :class:`pyphi.models.mechanism.Concept`.
    CauseEffectStructure: Alias for
        :class:`pyphi.models.subsystem.CauseEffectStructure`.
    Cut: Alias for :class:`pyphi.models.cuts.Cut`.
    Part: Alias for :class:`pyphi.models.cuts.Part`.
    Bipartition: Alias for :class:`pyphi.models.cuts.Bipartition`.
    ActualCut: Alias for :class:`pyphi.models.cuts.ActualCut`.
    AcRepertoireIrreducibilityAnalysis: Alias for.
     :class:`pyphi.models.actual_causation.AcRepertoireIrreducibilityAnalysis`.
    CausalLink: Alias for :class:`pyphi.models.actual_causation.CausalLink`.
    AcSystemIrreducibilityAnalysis: Alias for
        :class:`pyphi.models.actual_causation.AcSystemIrreducibilityAnalysis`.
    Account: Alias for :class:`pyphi.models.actual_causation.Account`.
    DirectedAccount: Alias for
        :class:`pyphi.models.actual_causation.DirectedAccount`.
"""

# pylint: disable=unused-import

from .actual_causation import (AcSystemIrreducibilityAnalysis, CausalLink,
                               AcRepertoireIrreducibilityAnalysis,
                               _null_ac_ria, Event, _null_ac_sia,
                               DirectedAccount, Account)
from .subsystem import (SystemIrreducibilityAnalysis, _null_sia,
                        CauseEffectStructure)
from .mechanism import (RepertoireIrreducibilityAnalysis, _null_ria,
                        MaximallyIrreducibleCauseOrEffect,
                        MaximallyIrreducibleCause, MaximallyIrreducibleEffect,
                        Concept)
from .cuts import (ActualCut, Cut, Part, Bipartition, NullCut, Tripartition,
                   KPartition, KCut)
