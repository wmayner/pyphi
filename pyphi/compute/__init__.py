#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# compute/__init__.py

"""Maintains backwards compatability with the old ``compute`` API.

See :mod:`compute.concept` and :mod:`compute.big_phi` for documentation.

Attributes:
    concept: Alias for :func:`concept.concept`.
    conceptual_info: Alias for :func:`concept.conceptual_info`.
    ces: Alias for :func:`concept.ces`.
    concept_distance: Alias for :func:`distance.concept_distance`.
    ces_distance: Alias for :func:`distance.ces_distance`.
    all_complexes: Alias for :func:`big_phi.all_complexes`.
    sia: Alias for :func:`big_phi.sia`.
    big_phi: Alias for :func:`big_phi.phi`.
    complexes: Alias for :func:`big_phi.complexes`.
    condensed: Alias for :func:`big_phi.condensed`.
    evaluate_cut: Alias for :func:`big_phi.evaluate_cut`.
    major_complex: Alias for :func:`big_phi.major_complex`.
    possible_complexes: Alias for :func:`big_phi.possible_complexes`.
    subsystems: Alias for :func:`big_phi.subsystems`.
"""

from .big_phi import (all_complexes, sia, phi, complexes, condensed,
                      evaluate_cut, major_complex, possible_complexes,
                      subsystems, ConceptStyleSystem, sia_concept_style,
                      concept_cuts, SystemIrreducibilityAnalysisConceptStyle)
from .concept import concept, conceptual_info, ces
from .distance import concept_distance, ces_distance
