"""Small config helpers used throughout PyPhi.

:func:`fallback` returns the first non-``None`` of its arguments — a tiny
shorthand that shows up wherever a function accepts an optional override
that should otherwise come from the global config.

:func:`parallel_kwargs` builds the kwargs dict for a parallel-call site
by overlaying user-provided kwargs onto a default option dict and
respecting the global parallel + progress switches.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import copy
from dataclasses import asdict
from typing import Any

import yaml


def yaml_repr(self: Any) -> str:
    """Render a frozen-dataclass config layer as nested YAML.

    Used as ``__repr__`` on the five config-layer dataclasses
    (``FormalismConfig``, ``IITConfig``, ``ActualCausationConfig``,
    ``InfrastructureConfig``, ``NumericsConfig``) so a layer or
    sub-namespace inspected in isolation renders the same way as
    the top-level ``pyphi.config``.
    """
    return yaml.safe_dump(
        asdict(self), sort_keys=False, default_flow_style=False
    ).rstrip()


def fallback(*args: Any) -> Any:
    """Return the first argument that is not ``None``."""
    for arg in args:
        if arg is not None:
            return arg
    return None


PARALLEL_KWARGS = [
    "reduce_func",
    "reduce_kwargs",
    "parallel",
    "ordered",
    "total",
    "chunksize",
    "sequential_threshold",
    "max_depth",
    "max_size",
    "max_leaves",
    "branch_factor",
    "shortcircuit_func",
    "shortcircuit_callback",
    "shortcircuit_callback_args",
    "inflight_limit",
    "progress",
    "desc",
    "map_kwargs",
]


def parallel_kwargs(
    option_kwargs: Mapping[str, Any], **user_kwargs: Any
) -> dict[str, Any]:
    """Build kwargs for a parallel function call.

    Overlays user overrides on the option-level defaults, then applies
    the global parallel + progress switches as gates: if either is off
    globally, the per-site setting is forced off.
    """
    from pyphi.conf import config

    kwargs = copy(dict(option_kwargs))
    if not config.infrastructure.progress_bars:
        kwargs["progress"] = False
    if not config.infrastructure.parallel:
        kwargs["parallel"] = False
    kwargs.update({k: v for k, v in user_kwargs.items() if k in PARALLEL_KWARGS})
    return kwargs


__all__ = ["PARALLEL_KWARGS", "fallback", "parallel_kwargs"]
