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
from .base import MetricNotCompatibleError
from .base import PhiFormalism
from .base import check_metric_compatible
from .iit3.formalism import IIT3Formalism
from .iit4.formalism import IIT4_2023Formalism
from .iit4.formalism import IIT4_2026Formalism

# Register the concrete formalisms. The string keys match the values
# ``config.FORMALISM`` will hold once the cut-over commit lands.
FORMALISM_REGISTRY.register("IIT_3_0", IIT3Formalism())
FORMALISM_REGISTRY.register("IIT_4_0_2023", IIT4_2023Formalism())
FORMALISM_REGISTRY.register("IIT_4_0_2026", IIT4_2026Formalism())

__all__ = [
    "FORMALISM_REGISTRY",
    "ApproximateFormalism",
    "ErrorInfo",
    "ExactFormalism",
    "FormalismRegistry",
    "IIT3Formalism",
    "IIT4_2023Formalism",
    "IIT4_2026Formalism",
    "MetricNotCompatibleError",
    "PhiFormalism",
    "check_metric_compatible",
]
