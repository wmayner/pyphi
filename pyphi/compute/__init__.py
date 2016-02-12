#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# compute/__init__.py

"""Maintains backwards compatability with the old ``compute`` API.

See |compute.concept| and |compute.big_phi| for documentation.

Attributes:
    concept: Alias for :func:`concept.concept`.
    concept_distance: Alias for :func:`concept.concept_distance`.
    conceptual_information: Alias for :func:`concept.conceptual_information`.
    constellation: Alias for :func:`concept.constellation`.
    constellation_distance: Alias for :func:`concept.constellation_distance`.
    all_complexes: Alias for :func:`big_phi.all_complexes`.
    big_mip: Alias for :func:`big_phi.big_mip`.
    big_phi: Alias for :func:`big_phi.big_phi`.
    complexes: Alias for :func:`big_phi.complexes`.
    condensed: Alias for :func:`big_phi.condensed`.
    evaluate_cut: Alias for :func:`big_phi.evaluate_cut`.
    main_complex: Alias for :func:`big_phi.main_complex`.
    possible_complexes: Alias for :func:`big_phi.possible_complexes`.
    subsystems: Alias for :func:`big_phi.subsystems`.
"""

from .concept import (concept, concept_distance, conceptual_information,
                      constellation, constellation_distance)
from .big_phi import (all_complexes, big_mip, big_phi, complexes, condensed,
                      evaluate_cut, main_complex, possible_complexes,
                      subsystems)
