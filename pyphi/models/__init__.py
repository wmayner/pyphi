#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# models/__init__.py

"""See |models.big_phi|, |models.concept|, and |models.cuts| for documentation.

Attributes:
    BigMip: Alias for :class:`big_phi.BigMip`
    Mip: Alias for :class:`concept.Mip`
    Mice: Alias for :class:`concept.Mice`
    Concept: Alias for :class:`concept.Concept`
    Constellation: Alias for :class:`concept.Constellation`
    Cut: Alias for :class:`cuts.Cut`
    Part: Alias for :class:`cuts.Part`
    Bipartition: Alias for :class:`cuts.Bipartition`
"""

from .big_phi import BigMip, _null_bigmip, _single_node_bigmip
from .concept import (Mip, _null_mip, Mice, Concept, Constellation,
                      normalize_constellation)
from .cuts import Cut, Part, Bipartition, Tripartition
