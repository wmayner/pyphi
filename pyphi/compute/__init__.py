#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# compute/__init__.py

'''Maintains backwards compatability with the old ``compute`` API.

See :mod:`compute.concept` and :mod:`compute.big_phi` for documentation.

Attributes:
    concept: Alias for :func:`concept.concept`.
    conceptual_information: Alias for :func:`concept.conceptual_information`.
    constellation: Alias for :func:`concept.constellation`.
    concept_distance: Alias for :func:`distance.concept_distance`.
    constellation_distance: Alias for :func:`distance.constellation_distance`.
    all_complexes: Alias for :func:`big_phi.all_complexes`.
    big_mip: Alias for :func:`big_phi.big_mip`.
    big_phi: Alias for :func:`big_phi.big_phi`.
    complexes: Alias for :func:`big_phi.complexes`.
    condensed: Alias for :func:`big_phi.condensed`.
    evaluate_cut: Alias for :func:`big_phi.evaluate_cut`.
    main_complex: Alias for :func:`big_phi.main_complex`.
    possible_complexes: Alias for :func:`big_phi.possible_complexes`.
    subsystems: Alias for :func:`big_phi.subsystems`.
'''

from .big_phi import (all_complexes, big_mip, big_phi, complexes, condensed,
                      evaluate_cut, main_complex, possible_complexes,
                      subsystems, ConceptStyleSystem, big_mip_concept_style,
                      concept_cuts, BigMipConceptStyle)
from .concept import concept, conceptual_information, constellation
from .distance import concept_distance, constellation_distance
