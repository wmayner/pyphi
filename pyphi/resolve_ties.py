# resolve_ties.py

"""Functions for resolving ties."""

import warnings
from itertools import tee

from . import config, metrics
from .conf import ConfigurationWarning, fallback
from .registry import Registry
from .utils import all_maxima, all_minima, NO_DEFAULT, iter_with_default


class PhiObjectTieResolutionRegistry(Registry):
    """Storage for functions for resolving ties among phi-objects."""

    desc = "functions for resolving ties among phi-objects"


phi_object_tie_resolution_strategies = PhiObjectTieResolutionRegistry()


def max_informativeness(m):
    if m.partitioned_repertoire is not None:
        return max(
            metrics.distribution.pointwise_mutual_information_vector(
                m.repertoire, m.partitioned_repertoire
            )[m.specified_index]
        )
    return 0.0


@phi_object_tie_resolution_strategies.register(
    "MAX_INFORMATIVENESS_THEN_SMALLEST_PURVIEW"
)
def _(m):
    return (
        m.phi,
        max_informativeness(m),
        -len(m.purview),
    )


@phi_object_tie_resolution_strategies.register(
    "MAX_INFORMATIVENESS_THEN_LARGEST_PURVIEW"
)
def _(m):
    return (
        m.phi,
        max_informativeness(m),
        len(m.purview),
    )


@phi_object_tie_resolution_strategies.register("LARGEST_PURVIEW")
def _(m):
    return (m.phi, len(m.purview))


@phi_object_tie_resolution_strategies.register("SMALLEST_PURVIEW")
def _(m):
    return (m.phi, -len(m.purview))


@phi_object_tie_resolution_strategies.register("PHI")
def _(m):
    return m.phi


@phi_object_tie_resolution_strategies.register("NONE")
def _(m):
    raise NotImplementedError(
        'tie resolution strategy "NONE" should never be called; '
        "it must be special-cased in the resolve() function"
    )


# TODO(4.0) docstring
def resolve(objects, strategy, operation=all_maxima, default=NO_DEFAULT):
    """Filter phi-objects to only those that maximize keys according to a strategy."""
    if strategy == "NONE":
        yield from iter_with_default(objects, default=default)
        return
    sort_key = phi_object_tie_resolution_strategies[strategy]
    key_args, objects = tee(objects)
    keys = map(sort_key, key_args)
    if default is not NO_DEFAULT:
        default = (sort_key(default), default)
    ties = operation(zip(keys, objects), default=default)
    for _, m in ties:
        yield m


def states(rias, strategy=None, **kwargs):
    """Resolve ties among states (RIAs).

    Controlled by the STATE_TIE_RESOLUTION configuration option.
    """
    strategy = fallback(strategy, config.STATE_TIE_RESOLUTION)
    return resolve(rias, strategy, **kwargs)


def partitions(mips, strategy=None, **kwargs):
    """Resolve ties among mechanism partitions (MIPs).

    Controlled by the MIP_TIE_RESOLUTION configuration option.
    """
    strategy = fallback(strategy, config.MIP_TIE_RESOLUTION)
    return resolve(mips, strategy, operation=all_minima, **kwargs)


def purviews(mice, strategy=None, **kwargs):
    """Resolve ties among purviews (MICEs).

    Controlled by the PURVIEW_TIE_RESOLUTION configuration option.
    """
    if not config.REPERTOIRE_DISTANCE.startswith("IIT_4.0_SMALL_PHI"):
        msg = f"""
        'PURVIEW_TIE_RESOLUTION = "{config.PURVIEW_TIE_RESOLUTION}"'
        assumes REPERTOIRE_DISTANCE is one of the "IIT_4.0_SMALL_PHI" measures,
        since informativeness is defined as the pointwise mutual information, but
        got REPERTOIRE_DISTANCE = {config.REPERTOIRE_DISTANCE}
        """
        # TODO(4.0) tie resolution docs
        warnings.warn(msg, category=ConfigurationWarning)

    strategy = fallback(strategy, config.PURVIEW_TIE_RESOLUTION)
    yield from resolve(mice, strategy, **kwargs)


class CESTieResolutionRegistry(Registry):
    """Storage for functions for resolving ties in cause-effect structures."""

    desc = "functions for resolving ties among purviews"


def ces(ces, system_state, strategy=None):
    """Resolve ties among CESs.

    Controlled by the CES_TIE_RESOLUTION configuration option.
    """
    strategy = fallback(strategy, config.CES_TIE_RESOLUTION)
    # TODO HERE
    # - resolve based on congruence
    # - resolve based on
    yield from all_maxima
