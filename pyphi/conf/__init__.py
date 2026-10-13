"""Layered configuration system for PyPhi.

Three frozen dataclass layers (:class:`FormalismConfig`,
:class:`InfrastructureConfig`, :class:`NumericsConfig`) wrapped in a
:class:`ConfigSnapshot` value type, accessed through the :data:`config`
singleton (:class:`_GlobalConfig`). Both flat (``config.precision``) and
layered (``config.numerics.precision``) reads work; writes use the flat
form (``config.precision = 6``) or :meth:`config.override` for scoped
changes.

If a ``pyphi_config.yml`` file exists in the working directory at import
time, its layered-format contents are auto-applied. A YAML file using the
uppercase flat format raises :class:`ConfigurationError` with a pointer to
the rename map.
"""

from __future__ import annotations

from pathlib import Path

from pyphi.conf._field_routing import FIELD_TO_LAYER
from pyphi.conf._field_routing import FILE_VERSION_ADVICE
from pyphi.conf._field_routing import ConfigurationError
from pyphi.conf._field_routing import UnregisteredValueError
from pyphi.conf._global import _GlobalConfig
from pyphi.conf._helpers import fallback
from pyphi.conf._helpers import parallel_kwargs
from pyphi.conf._helpers import reject_unknown_kwargs
from pyphi.conf.formalism import ActualCausationConfig
from pyphi.conf.formalism import FormalismConfig
from pyphi.conf.formalism import IITConfig
from pyphi.conf.infrastructure import InfrastructureConfig
from pyphi.conf.numerics import NumericsConfig
from pyphi.conf.presets import iit3
from pyphi.conf.presets import iit4_2023
from pyphi.conf.presets import iit4_2026
from pyphi.conf.snapshot import ConfigSnapshot

PYPHI_USER_CONFIG_PATH = Path("pyphi_config.yml")


config: _GlobalConfig = _GlobalConfig()


def _config_file_error(
    error: ConfigurationError, *, combination: bool = False
) -> ConfigurationError:
    """Return ``error`` restated as a problem with ``pyphi_config.yml``.

    Used for errors raised while ``import pyphi`` applies the file, so that the
    message names the file. ``combination`` marks an error about options that
    must agree, whose message then says how to select a version of IIT in the
    file.
    """
    message = f"{PYPHI_USER_CONFIG_PATH.resolve()} is not a valid configuration: {error}"
    if combination and not isinstance(error, UnregisteredValueError):
        message += " " + FILE_VERSION_ADVICE
    return type(error)(message)


try:
    config.load_yaml(PYPHI_USER_CONFIG_PATH)
except FileNotFoundError:
    pass
except ConfigurationError as error:
    raise _config_file_error(error) from None


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
    "iit3",
    "iit4_2023",
    "iit4_2026",
    "parallel_kwargs",
    "reject_unknown_kwargs",
]
