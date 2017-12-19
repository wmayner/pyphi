#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# compute/__init__.py

"""Maintains backwards compatability with the old ``compute`` API.

See :mod:`compute.subsystem` and :mod:`compute.network` for documentation.

Attributes:
    conceptual_info: Alias for :func:`subsystem.conceptual_info`.
    ces: Alias for :func:`subsystem.ces`.
    concept_distance: Alias for :func:`distance.concept_distance`.
    ces_distance: Alias for :func:`distance.ces_distance`.
    all_complexes: Alias for :func:`network.all_complexes`.
    sia: Alias for :func:`subsystem.sia`.
    phi: Alias for :func:`subsystem.phi`.
    complexes: Alias for :func:`network.complexes`.
    condensed: Alias for :func:`network.condensed`.
    evaluate_cut: Alias for :func:`subsystem.evaluate_cut`.
    major_complex: Alias for :func:`network.major_complex`.
    possible_complexes: Alias for :func:`network.possible_complexes`.
    subsystems: Alias for :func:`network.subsystems`.
"""

from .subsystem import (sia, phi, evaluate_cut, ConceptStyleSystem,
                        sia_concept_style, concept_cuts,
                        SystemIrreducibilityAnalysisConceptStyle,
                        conceptual_info, ces)
from .network import (all_complexes, complexes, condensed, major_complex,
                      possible_complexes, subsystems)
from .distance import concept_distance, ces_distance
