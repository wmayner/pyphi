"""Phi formalisms — strategies for computing integrated information.

A *formalism* is the top-level strategy that bundles a partition scheme, a
compatible distance metric, and the algorithms that combine them into
mechanism-level RIAs, system-level SIAs, and Φ-structures. The
:class:`pyphi.formalism.base.PhiFormalism` Protocol declares the contract;
concrete implementations live in :mod:`pyphi.formalism.iit3` and
:mod:`pyphi.formalism.iit4`. The active formalism is selected by name via
``config.FORMALISM``.
"""

from .base import FORMALISM_REGISTRY
from .base import ApproximateFormalism
from .base import ErrorInfo
from .base import ExactFormalism
from .base import FormalismRegistry
from .base import PhiFormalism

__all__ = [
    "FORMALISM_REGISTRY",
    "ApproximateFormalism",
    "ErrorInfo",
    "ExactFormalism",
    "FormalismRegistry",
    "PhiFormalism",
]
