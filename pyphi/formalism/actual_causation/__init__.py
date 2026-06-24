"""Actual Causation formalism (Albantakis et al. 2019).

Holds the AC compute algorithms (:mod:`.compute`) and the registered
``AC2019Formalism`` object. The data layer (``Transition`` /
``TransitionSystem``) lives in :mod:`pyphi.actual`, which dispatches its
public functions through this package.
"""

from __future__ import annotations

from pyphi.formalism.base import ACTUAL_CAUSATION_FORMALISM_REGISTRY

from .compute import account_distance
from .compute import alpha_aggregations
from .compute import background_strategies
from .compute import partitioned_repertoire_schemes
from .compute import probability_distance
from .formalism import AC2019Formalism

ACTUAL_CAUSATION_FORMALISM_REGISTRY.register("AC_2019", AC2019Formalism())

__all__ = [
    "AC2019Formalism",
    "account_distance",
    "alpha_aggregations",
    "background_strategies",
    "partitioned_repertoire_schemes",
    "probability_distance",
]
