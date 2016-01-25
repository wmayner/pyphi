#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# compute.py

"""Maintains backwards compatability with the old compute API."""

from .concept import (concept, constellation, concept_distance,
                      constellation_distance, conceptual_information)
from .big_phi import (big_mip, big_phi, subsystems, all_complexes,
                      possible_complexes, complexes, main_complex, condensed)

__all__ = ['concept', 'constellation', 'concept_distance']
