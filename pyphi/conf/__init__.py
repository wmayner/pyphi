"""Layered configuration system for PyPhi 2.0.

Three frozen dataclass layers (:class:`FormalismConfig`,
:class:`InfrastructureConfig`, :class:`NumericsConfig`) wrapped in a
:class:`ConfigSnapshot` value type, accessed through the :data:`config`
singleton (:class:`_GlobalConfig`). Both flat (``config.precision``) and
layered (``config.numerics.precision``) reads work; writes use the flat
form (``config.precision = 6``) or :meth:`config.override` for scoped
changes.

If a ``pyphi_config.yml`` file exists in the working directory at import
time, its layered-format contents are auto-applied. Legacy uppercase YAML
format raises :class:`ConfigurationError` with a pointer to the rename map.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

from pyphi.conf._callbacks import mark_loaded
from pyphi.conf._field_routing import FIELD_TO_LAYER
from pyphi.conf._field_routing import ConfigurationError
from pyphi.conf._global import _GlobalConfig
from pyphi.conf._helpers import fallback
from pyphi.conf._helpers import parallel_kwargs
from pyphi.conf.formalism import ActualCausationConfig
from pyphi.conf.formalism import FormalismConfig
from pyphi.conf.formalism import IITConfig
from pyphi.conf.infrastructure import InfrastructureConfig
from pyphi.conf.numerics import NumericsConfig
from pyphi.conf.presets import iit3_settings
from pyphi.conf.presets import iit4_2023_settings
from pyphi.conf.presets import iit4_2026_settings
from pyphi.conf.snapshot import ConfigSnapshot

PYPHI_USER_CONFIG_PATH = Path("pyphi_config.yml")


config: _GlobalConfig = _GlobalConfig()

with contextlib.suppress(FileNotFoundError):
    config.load_yaml(PYPHI_USER_CONFIG_PATH)

mark_loaded()


__all__ = [
    "FIELD_TO_LAYER",
    "PYPHI_USER_CONFIG_PATH",
    "ActualCausationConfig",
    "ConfigSnapshot",
    "ConfigurationError",
    "FormalismConfig",
    "IITConfig",
    "InfrastructureConfig",
    "NumericsConfig",
    "config",
    "fallback",
    "iit3_settings",
    "iit4_2023_settings",
    "iit4_2026_settings",
    "parallel_kwargs",
]
