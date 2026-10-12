"""Small config helpers used throughout PyPhi.

:func:`fallback` returns the first non-``None`` of its arguments — a tiny
shorthand that shows up wherever a function accepts an optional override
that should otherwise come from the global config.

:func:`parallel_kwargs` builds the kwargs dict for a parallel-call site
by overlaying user-provided kwargs onto a default option dict and
respecting the global parallel + progress switches.
"""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping
from copy import copy
from dataclasses import asdict
from typing import Any

import yaml


def plain_builtins(value: Any) -> Any:
    """Recursively convert to plain builtins for ``yaml.safe_dump``.

    Mappings (including ``FrozenMap``) become dicts; tuples and lists
    become lists. Other values pass through unchanged.
    """
    if isinstance(value, Mapping):
        return {key: plain_builtins(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_builtins(v) for v in value]
    return value


def yaml_repr(self: Any) -> str:
    """Render a frozen-dataclass config layer as nested YAML.

    Used as ``__repr__`` on the five config-layer dataclasses
    (``FormalismConfig``, ``IITConfig``, ``ActualCausationConfig``,
    ``InfrastructureConfig``, ``NumericsConfig``) so a layer or
    sub-namespace inspected in isolation renders the same way as
    the top-level ``pyphi.config``.
    """
    return yaml.safe_dump(
        plain_builtins(asdict(self)), sort_keys=False, default_flow_style=False
    ).rstrip()


def read_only[T: type](cls: T) -> T:
    """Make assigning to an attribute of the frozen layer ``cls`` raise a
    :class:`~pyphi.conf.ConfigurationError` that shows how to set the option
    on ``pyphi.config``.

    Applied over ``@dataclass(frozen=True)``, which constructs instances with
    ``object.__setattr__`` and so is unaffected.
    """

    def __setattr__(self: Any, name: str, value: Any) -> None:
        from pyphi.conf._field_routing import layer_write_error

        raise layer_write_error(self, name, value)

    cls.__setattr__ = __setattr__  # type: ignore[method-assign,assignment]
    return cls


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
    "shortcircuit_func",
    "shortcircuit_callback",
    "shortcircuit_callback_args",
    "progress",
    "desc",
    "map_kwargs",
    "size_func",
    "backend",
]


def parallel_kwargs(
    option_kwargs: Mapping[str, Any], **user_kwargs: Any
) -> dict[str, Any]:
    """Build kwargs for a parallel function call.

    Applies the global parallel + progress switches as gates to the
    option-level defaults (if either is off globally, the per-site default
    is forced off), then overlays explicit user overrides — so a per-call
    setting always wins over the global gate.
    """
    from pyphi.conf import config

    kwargs = copy(dict(option_kwargs))
    if not config.infrastructure.progress_bars:
        kwargs["progress"] = False
    if not config.infrastructure.parallel:
        kwargs["parallel"] = False
    kwargs.update({k: v for k, v in user_kwargs.items() if k in PARALLEL_KWARGS})
    return kwargs


# Level names mapped to the per-level parallelization options. A single
# analysis pays off at the analysis levels; relations and mechanism partitions
# are excluded because parallelizing them was measured not to pay (see
# docs/howto/parallel.md).
PARALLEL_LEVELS = {
    "partitions": "parallel_partition_evaluation",
    "purviews": "parallel_purview_evaluation",
    "distinctions": "parallel_distinction_evaluation",
    "complexes": "parallel_complex_evaluation",
    "mechanism_partitions": "parallel_mechanism_partition_evaluation",
    "relations": "parallel_relation_evaluation",
    "macro_systems": "parallel_macro_system_evaluation",
}
ANALYSIS_PARALLEL_LEVELS = ("partitions", "purviews", "distinctions")


def parallel_overrides(parallel: bool | Iterable[str] | None) -> dict[str, Any]:
    """Build ``config.override`` values that run one analysis in parallel.

    Parameters
    ----------
    parallel : bool, iterable of str, or None
        ``None`` changes nothing. ``False`` closes the global gate, so every
        level runs sequentially. ``True`` opens the gate and switches on the
        levels a single analysis uses (:data:`ANALYSIS_PARALLEL_LEVELS`); an
        iterable of names from :data:`PARALLEL_LEVELS` switches on those
        levels instead. Either way every other level is switched off.

    Returns
    -------
    dict
        Option values for :meth:`pyphi.conf.config.override`.

    Raises
    ------
    ValueError
        If a level name is not in :data:`PARALLEL_LEVELS`.
    """
    from pyphi.conf import config

    if parallel is None:
        return {}
    if parallel is False:
        return {"parallel": False}
    enabled = ANALYSIS_PARALLEL_LEVELS if parallel is True else tuple(parallel)
    unknown = [name for name in enabled if name not in PARALLEL_LEVELS]
    if unknown:
        raise ValueError(
            f"Unknown parallel level(s) {', '.join(map(repr, unknown))}. "
            f"Available levels: {', '.join(PARALLEL_LEVELS)}."
        )
    overrides: dict[str, Any] = {"parallel": True}
    for name, option in PARALLEL_LEVELS.items():
        overrides[option] = {
            **dict(getattr(config.infrastructure, option)),
            "parallel": name in enabled,
        }
    return overrides


def reject_unknown_kwargs(caller: str, kwargs: Mapping[str, Any]) -> None:
    """Raise ``TypeError`` if ``kwargs`` holds a name outside :data:`PARALLEL_KWARGS`.

    For functions whose ``**kwargs`` reach only :func:`parallel_kwargs`,
    which drops unknown names: a misspelt or misplaced option then fails
    instead of being ignored.
    """
    unknown = sorted(set(kwargs) - set(PARALLEL_KWARGS))
    if unknown:
        names = ", ".join(repr(name) for name in unknown)
        raise TypeError(f"{caller}() got unexpected keyword argument(s): {names}")


__all__ = [
    "ANALYSIS_PARALLEL_LEVELS",
    "PARALLEL_KWARGS",
    "PARALLEL_LEVELS",
    "fallback",
    "parallel_kwargs",
    "parallel_overrides",
    "reject_unknown_kwargs",
]
