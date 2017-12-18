#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# compute/__init__.py

"""Maintains backwards compatability with the old ``compute`` API.

See :mod:`compute.concept` and :mod:`compute.system` for documentation.

Attributes:
    concept: Alias for :func:`concept.concept`.
    conceptual_info: Alias for :func:`concept.conceptual_info`.
    ces: Alias for :func:`concept.ces`.
    concept_distance: Alias for :func:`distance.concept_distance`.
    ces_distance: Alias for :func:`distance.ces_distance`.
    all_complexes: Alias for :func:`system.all_complexes`.
    sia: Alias for :func:`system.sia`.
    system: Alias for :func:`system.phi`.
    complexes: Alias for :func:`system.complexes`.
    condensed: Alias for :func:`system.condensed`.
    evaluate_cut: Alias for :func:`system.evaluate_cut`.
    major_complex: Alias for :func:`system.major_complex`.
    possible_complexes: Alias for :func:`system.possible_complexes`.
    subsystems: Alias for :func:`system.subsystems`.
"""

from .system import (all_complexes, sia, phi, complexes, condensed,
                     evaluate_cut, major_complex, possible_complexes,
                     subsystems, ConceptStyleSystem, sia_concept_style,
                     concept_cuts, SystemIrreducibilityAnalysisConceptStyle)
from .concept import concept, conceptual_info, ces
from .distance import concept_distance, ces_distance
