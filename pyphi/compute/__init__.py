#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# compute/__init__.py

"""Maintains backwards compatability with the old compute API."""

from .concept import (concept, concept_distance, conceptual_information,
                      constellation, constellation_distance)
from .big_phi import (all_complexes, big_mip, big_phi, complexes, condensed,
                      evaluate_cut, main_complex, possible_complexes,
                      subsystems)
