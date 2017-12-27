#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# models/__init__.py

"""
See |models.subsystem|, |models.mechanism|, and |models.cuts| for
documentation.

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
        :class:`pyphi.models.subsystem.CauseEffectStructure`.
    Concept: Alias for :class:`pyphi.models.mechanism.Concept`.
    Cut: Alias for :class:`pyphi.models.cuts.Cut`.
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
        :class:`pyphi.models.subsystem.SystemIrreducibilityAnalysis`.
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
