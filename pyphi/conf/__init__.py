"""Layered configuration system for PyPhi 2.0.

Three frozen dataclass layers (``FormalismConfig``, ``InfrastructureConfig``,
``NumericsConfig``) wrapped in a ``ConfigSnapshot`` value type, accessed
through the ``config`` singleton. During the P10 cutover, ``config`` is a
:class:`_GlobalConfig` instance that exposes the layered view as live
properties over the wrapped legacy :class:`PyphiConfig` instance, so both
old uppercase access (``config.PRECISION``) and new layered access
(``config.numerics.precision``) reflect the same source of truth.

Phase 6 deletes the legacy backend and replaces this with a self-owning
:class:`_GlobalConfig` that stores its layers directly.
"""

from pyphi import _conf_legacy as _legacy
from pyphi._conf_legacy import Config
from pyphi._conf_legacy import Option
from pyphi._conf_legacy import PyphiConfig
from pyphi._conf_legacy import fallback
from pyphi._conf_legacy import parallel_kwargs
from pyphi.conf._field_routing import FIELD_TO_LAYER
from pyphi.conf._field_routing import ConfigurationError
from pyphi.conf.formalism import FormalismConfig
from pyphi.conf.infrastructure import InfrastructureConfig
from pyphi.conf.legacy_global import _GlobalConfig
from pyphi.conf.numerics import NumericsConfig
from pyphi.conf.snapshot import ConfigSnapshot

config: _GlobalConfig = _GlobalConfig(_legacy.config)

__all__ = [
    "FIELD_TO_LAYER",
    "Config",
    "ConfigSnapshot",
    "ConfigurationError",
    "FormalismConfig",
    "InfrastructureConfig",
    "NumericsConfig",
    "Option",
    "PyphiConfig",
    "config",
    "fallback",
    "parallel_kwargs",
]
