#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# models/__init__.py

"""
See |models.subsystem|, |models.mechanism|, and |models.cuts| for
documentation.

Attributes:
    SystemIrreducibilityAnalysis: Alias for
        :class:`subsystem.SystemIrreducibilityAnalysis`
    RepertoireIrreducibilityAnalysis: Alias for
        :class:`mechanism.RepertoireIrreducibilityAnalysis`
    MaximallyIrreducibleCause: Alias for
        :class:`mechanism.MaximallyIrreducibleCause`
    MaximallyIrreducibleEffect: Alias for
        :class:`mechanism.MaximallyIrreducibleEffect`
    MaximallyIrreducibleCauseOrEffect: Alias for
        :class:`mechanism.MaximallyIrreducibleCauseOrEffect`
    Concept: Alias for :class:`mechanism.Concept`
    CauseEffectStructure: Alias for :class:`subsystem.CauseEffectStructure`
    Cut: Alias for :class:`cuts.Cut`
    Part: Alias for :class:`cuts.Part`
    Bipartition: Alias for :class:`cuts.Bipartition`
    ActualCut: Alias for :class:`cuts.ActualCut`
    AcRepertoireIrreducibilityAnalysis: Alias for
        :class:`actual_causation.AcRepertoireIrreducibilityAnalysis`
    CausalLink: Alias for :class:`actual_causation.CausalLink`
    AcSystemIrreducibilityAnalysis: Alias for
        :class:`actual_causation.AcSystemIrreducibilityAnalysis`
    Account: Alias for :class:`actual_causation.Account`
    DirectedAccount: Alias for :class:`actual_causation.DirectedAccount`
"""

from .actual_causation import (AcSystemIrreducibilityAnalysis, CausalLink,
                               AcRepertoireIrreducibilityAnalysis, _null_ac_ria,
                               Event, _null_ac_sia, DirectedAccount, Account)
from .subsystem import (SystemIrreducibilityAnalysis, _null_sia,
                        CauseEffectStructure)
from .mechanism import (RepertoireIrreducibilityAnalysis, _null_ria,
                        MaximallyIrreducibleCauseOrEffect,
                        MaximallyIrreducibleCause, MaximallyIrreducibleEffect,
                        Concept)
from .cuts import (ActualCut, Cut, Part, Bipartition, NullCut, Tripartition,
                   KPartition, KCut)
