#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# compute/__init__.py

"""
See |compute.subsystem|, |compute.network|, |compute.distance|, and
|compute.parallel| for documentation.

Attributes:
    conceptual_info: Alias for :func:`pyphi.compute.subsystem.conceptual_info`.
    ces: Alias for :func:`pyphi.compute.subsystem.ces`.
    concept_distance: Alias for
        :func:`pyphi.compute.distance.concept_distance`.
    ces_distance: Alias for :func:`pyphi.compute.distance.ces_distance`.
    all_complexes: Alias for :func:`pyphi.compute.network.all_complexes`.
    sia: Alias for :func:`pyphi.compute.subsystem.sia`.
    phi: Alias for :func:`pyphi.compute.subsystem.phi`.
    complexes: Alias for :func:`pyphi.compute.network.complexes`.
    condensed: Alias for :func:`pyphi.compute.network.condensed`.
    evaluate_cut: Alias for :func:`pyphi.compute.subsystem.evaluate_cut`.
    major_complex: Alias for :func:`pyphi.compute.network.major_complex`.
    possible_complexes: Alias for
        :func:`pyphi.compute.network.possible_complexes`.
    subsystems: Alias for :func:`pyphi.compute.network.subsystems`.
"""

# pylint: disable=unused-import

from .subsystem import (sia, phi, evaluate_cut, ConceptStyleSystem,
                        sia_concept_style, concept_cuts,
                        SystemIrreducibilityAnalysisConceptStyle,
                        conceptual_info, ces)
from .network import (all_complexes, complexes, condensed, major_complex,
                      possible_complexes, subsystems)
from .distance import concept_distance, ces_distance
