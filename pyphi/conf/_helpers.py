"""Small config helpers used throughout PyPhi.

:func:`fallback` returns the first non-``None`` of its arguments — a tiny
shorthand that shows up wherever a function accepts an optional override
that should otherwise come from the global config.

:func:`parallel_kwargs` builds the kwargs dict for a parallel-call site
by overlaying user-provided kwargs onto a default option dict and
respecting the global parallel + progress switches.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable
from collections.abc import Mapping
from copy import copy
from dataclasses import asdict
from typing import Any

import numpy as np
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

    if user_kwargs.get("parallel") is not None:
        user_kwargs["parallel"] = check_parallel_flag(user_kwargs["parallel"])
    if user_kwargs.get("progress") is not None:
        user_kwargs["progress"] = check_progress_flag(user_kwargs["progress"])
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


def _check_flag(name: str, value: Any) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"{name} must be True, False or None, not {value!r}.")


def check_parallel_flag(parallel: Any) -> bool | None:
    """Return a ``parallel=`` argument as a bool, or None if it is None.

    Raises
    ------
    TypeError
        If ``parallel`` is not a bool (Python or NumPy) or None.
    """
    return _check_flag("parallel", parallel)


def check_progress_flag(progress: Any) -> bool | None:
    """Return a ``progress=`` argument as a bool, or None if it is None.

    Raises
    ------
    TypeError
        If ``progress`` is not a bool (Python or NumPy) or None.
    """
    return _check_flag("progress", progress)


def parallel_overrides(
    parallel: bool | list[str] | tuple[str, ...] | None,
) -> dict[str, Any]:
    """Build ``config.override`` values that run one analysis in parallel.

    Parameters
    ----------
    parallel : bool, list or tuple of str, or None
        ``None`` changes nothing. ``False`` closes the global gate, so every
        level runs sequentially. ``True`` opens the gate and switches on the
        levels a single analysis uses (:data:`ANALYSIS_PARALLEL_LEVELS`); a
        list or tuple of names from :data:`PARALLEL_LEVELS` switches on those
        levels instead. Either way every other level is switched off.

    Returns
    -------
    dict
        Option values for :meth:`pyphi.conf.config.override`.

    Raises
    ------
    TypeError
        If ``parallel`` is none of these, for example a string.
    ValueError
        If a level name is not in :data:`PARALLEL_LEVELS`.
    """
    from pyphi.conf import config

    if isinstance(parallel, (list, tuple)):
        enabled = tuple(parallel)
    else:
        parallel = check_parallel_flag(parallel)
        if parallel is None:
            return {}
        if parallel is False:
            return {"parallel": False}
        enabled = ANALYSIS_PARALLEL_LEVELS
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


# Names that PyPhi 1.x and the 2.0 release candidates used for the units of the
# candidate system, which every entry point now calls ``subset``.
_OLD_SUBSET_NAMES = frozenset({"indices", "nodes", "node_indices"})


def reject_unknown_kwargs(
    caller: str,
    kwargs: Mapping[str, Any],
    allowed: Iterable[str] = PARALLEL_KWARGS,
) -> None:
    """Raise ``TypeError`` if ``kwargs`` holds a name outside ``allowed``.

    For functions whose ``**kwargs`` reach code that would drop or ignore
    unknown names: a misspelt or misplaced option then fails instead of being
    ignored. The message names ``caller`` and suggests a close match, or,
    when ``subset`` is allowed, ``subset=`` for an old name of that argument.

    Parameters
    ----------
    caller : str
        The function name the message reports, such as ``"Substrate.sia"``.
    kwargs : Mapping[str, Any]
        The keyword arguments given.
    allowed : Iterable[str], optional
        The accepted names. Defaults to :data:`PARALLEL_KWARGS`.

    Raises
    ------
    TypeError
        If a name in ``kwargs`` is not in ``allowed``.
    """
    allowed = list(allowed)
    for name in kwargs:
        if name in allowed:
            continue
        message = f"{caller}() got an unexpected keyword argument {name!r}"
        close = difflib.get_close_matches(name, allowed, n=1)
        if name in _OLD_SUBSET_NAMES and "subset" in allowed:
            message += "; pass the units of the candidate system as subset=..."
        elif close:
            message += f". Did you mean {close[0]!r}?"
        raise TypeError(message)


__all__ = [
    "ANALYSIS_PARALLEL_LEVELS",
    "PARALLEL_KWARGS",
    "PARALLEL_LEVELS",
    "check_parallel_flag",
    "check_progress_flag",
    "fallback",
    "parallel_kwargs",
    "parallel_overrides",
    "reject_unknown_kwargs",
]
