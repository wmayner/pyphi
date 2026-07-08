"""YAML I/O for the layered config system.

The nested format groups options under three top-level keys
(``formalism``, ``infrastructure``, ``numerics``). A flat format with
uppercase top-level keys is detected and rejected with a pointer to the
rename map.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from pyphi.conf._field_routing import ConfigurationError

KNOWN_LAYERS = frozenset({"formalism", "infrastructure", "numerics"})


def load_yaml(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load a nested-format YAML config file.

    Returns
    -------
    dict[str, dict[str, Any]]
        The parsed top-level layer sections, keyed by layer name.

    Raises
    ------
    ConfigurationError
        If the file does not parse to a dict, uses the flat format with
        uppercase top-level keys, or contains a top-level key outside
        ``formalism`` / ``infrastructure`` / ``numerics``.
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigurationError(f"Config file {path!r} doesn't parse to a dict.")
    upper_keys = [k for k in data if isinstance(k, str) and k.isupper()]
    if upper_keys:
        raise ConfigurationError(
            f"Config file {path!r} uses the 1.x flat format "
            f"(e.g., {upper_keys[0]!r}). In 2.0, options are grouped by "
            f"layer (formalism / infrastructure / numerics). See "
            f"changelog.d/p10-config-split.refactor.md for the rename map."
        )
    unknown = set(data) - KNOWN_LAYERS
    if unknown:
        raise ConfigurationError(
            f"Unknown top-level YAML key(s) in {path!r}: {sorted(unknown)}. "
            f"Expected one of: {sorted(KNOWN_LAYERS)}."
        )
    return data
