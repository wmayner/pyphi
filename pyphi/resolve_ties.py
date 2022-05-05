# resolve_ties.py

"""Functions for resolving ties."""

import warnings

from . import config, metrics
from .conf import ConfigurationWarning
from .registry import Registry


class MICETieResolutionRegistry(Registry):
    """Storage for functions for resolving ties among purviews (MICE).

    Users can define custom schemes:

    Examples:
        >>> @mice_resolution_registry.register('NONE')  # doctest: +SKIP
        ... def all_ties(mips):
        ...    return mips

    And use them by setting ``config.MICE_TIE_RESOLUTION = 'NONE'``
    """

    desc = "functions for resolving ties among purviews"


mice_resolution = MICETieResolutionRegistry()


@mice_resolution.register("MAX_INFORMITAVENESS_THEN_SMALLEST_PURVIEW")
def _(m):
    if not config.REPERTOIRE_DISTANCE == "IIT_4.0_SMALL_PHI":
        msg = """
    'MICE_TIE_RESOLUTION = "MAX_INFORMITAVENESS_THEN_SMALLEST_PURVIEW"
    assumes
      REPERTOIRE_DISTANCE = "IIT_4.0_SMALL_PHI"
    since informativeness is defined as the pointwise mutual information, but got
      REPERTOIRE_DISTANCE = {config.REPERTOIRE_DISTANCE}
"""
        # TODO(4.0) tie resolution docs
        warnings.warn(msg, category=ConfigurationWarning)

    informativeness = max(
        metrics.distribution.pointwise_mutual_information_vector(
            m.repertoire, m.partitioned_repertoire
        )[m.specified_index]
    )
    return (
        informativeness,
        -len(m.purview),
    )


@mice_resolution.register("LARGEST_PURVIEW")
def _(m):
    return len(m.purview)


@mice_resolution.register("SMALLEST_PURVIEW")
def _(m):
    return -len(m.purview)


def resolve(mice, sort_key):
    """Return MICE that are tied after sorting by the given key."""
    if not mice:
        return mice
    keys = list(map(sort_key, mice))
    mice = sorted(zip(keys, mice), reverse=True)
    max_key = mice[0][0]
    ties = [m for (k, m) in mice if k == max_key]
    return ties


def mice(tied_mice):
    """Resolve ties among MICE.

    Controlled by the MICE_TIE_RESOLUTION configuration option.
    """
    return resolve(tied_mice, sort_key=mice_resolution[config.MICE_TIE_RESOLUTION])
