# resolve_ties.py
"""Resolve ties between IIT objects."""

from itertools import tee

from .conf import config, fallback
from .registry import Registry
from .utils import NO_DEFAULT, iter_with_default


class PhiObjectTieResolutionRegistry(Registry):
    """Storage for functions for resolving ties among phi-objects."""

    desc = "functions for resolving ties among phi-objects"


phi_object_tie_resolution_strategies = PhiObjectTieResolutionRegistry()


@phi_object_tie_resolution_strategies.register("PURVIEW_SIZE")
def _(m):
    return len(m.purview)


@phi_object_tie_resolution_strategies.register("NEGATIVE_PURVIEW_SIZE")
def _(m):
    return -len(m.purview)


@phi_object_tie_resolution_strategies.register("PHI")
def _(m):
    return m.phi


@phi_object_tie_resolution_strategies.register("NEGATIVE_PHI")
def _(m):
    return -m.phi


@phi_object_tie_resolution_strategies.register("NORMALIZED_PHI")
def _(m):
    return m.normalized_phi


@phi_object_tie_resolution_strategies.register("NEGATIVE_NORMALIZED_PHI")
def _(m):
    return -m.normalized_phi


@phi_object_tie_resolution_strategies.register("NONE")
def _(m):
    raise NotImplementedError(
        'tie resolution strategy "NONE" should never be called; '
        "it must be special-cased in the resolve() function"
    )


def _strategies_to_key_function(strategies):
    """Convert a tie resolution strategy to a key function."""
    if isinstance(strategies, str):
        # Allow a single strategy to be specified as a bare string
        strategies = [strategies]
    return lambda obj: tuple(
        phi_object_tie_resolution_strategies[s](obj) for s in strategies
    )


# TODO(4.0) docstring
# TODO(4.0) fix this implementation so we only need one pass; currently,
# all_maxima only works if equality semantics are correct for this purpose, and
# RIA equality checks purview equality, so they are not.
# def resolve(objects, strategy, operation=all_maxima, default=NO_DEFAULT):
#     """Filter phi-objects according to a strategy."""
#     if strategy == "NONE":
#         yield from iter_with_default(objects, default=default)
#         return
#     sort_key = _strategies_to_key_function(strategy)
#     key_args, objects = tee(objects)
#     keys = map(sort_key, key_args)
#     if default is not NO_DEFAULT:
#         default = (sort_key(default), default)
#     ties = operation(zip(keys, objects), default=default)
#     for _, obj in ties:
#         yield obj


def resolve(objects, strategy, operation, default=NO_DEFAULT):
    """Filter phi-objects according to a strategy."""
    if strategy == "NONE":
        yield from iter_with_default(objects, default=default)
        return
    sort_key = _strategies_to_key_function(strategy)
    objects, to_transform = tee(objects)
    values = list(map(sort_key, to_transform))
    extremum = operation(values, default=default)
    ties = (obj for obj, value in zip(objects, values) if value == extremum)
    yield from iter_with_default(ties, default=default)


def states(rias, strategy=None, **kwargs):
    """Resolve ties among states (RIAs).

    Controlled by the STATE_TIE_RESOLUTION configuration option.
    """
    strategy = fallback(strategy, config.STATE_TIE_RESOLUTION)
    return resolve(rias, strategy, operation=max, **kwargs)


def partitions(mips, strategy=None, **kwargs):
    """Resolve ties among mechanism partitions (MIPs).

    Controlled by the MIP_TIE_RESOLUTION configuration option.
    """
    strategy = fallback(strategy, config.MIP_TIE_RESOLUTION)
    return resolve(mips, strategy, operation=min, **kwargs)


def purviews(mice, strategy=None, **kwargs):
    """Resolve ties among purviews (MICEs).

    Controlled by the PURVIEW_TIE_RESOLUTION configuration option.
    """
    strategy = fallback(strategy, config.PURVIEW_TIE_RESOLUTION)
    yield from resolve(mice, strategy, operation=max, **kwargs)
