# models/pandas.py
"""Utilities for working with Pandas data structures."""

from collections.abc import Iterable
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

# TODO Just use `to_json` instead of `to_dict`?


def try_to_dict(obj: object) -> dict[str, Any] | object:
    try:
        return obj.to_dict()  # type: ignore[attr-defined]
    except AttributeError:
        return obj


class ToDictFromExplicitAttrsMixin:
    """Mixin class for converting a class to a dict from the `_dict_attrs` list."""

    _dict_attrs: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert the object to a dict."""
        if hasattr(self, "_dict_attrs"):
            return {attr: try_to_dict(getattr(self, attr)) for attr in self._dict_attrs}
        raise NotImplementedError("no `_dict_attrs` attribute")


class ToDictMixin:
    """Mixin class for converting a class to a dict from the object's ___dict___."""

    def to_dict(self) -> dict[str, Any]:
        """Convert the object to a dict."""
        return {
            attr: try_to_dict(value)
            for attr, value in self.__dict__.items()
            if not attr.startswith("_")
        }


_DISTRIBUTION_COLUMNS = ["direction", "kind", "purview", "state", "probability"]


def record_to_series(record: Mapping[str, Any], name: str | None = None) -> pd.Series:
    """Build a Series from an ordered field-to-value mapping."""
    return pd.Series(dict(record), name=name)


def records_to_frame(
    rows: Iterable[Mapping[str, Any]],
    index: str | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Stack record mappings into a DataFrame, optionally moving one column to
    the index. ``columns`` fixes the column set so an empty ``rows`` still
    produces the right schema."""
    frame = pd.DataFrame(
        list(rows), columns=None if columns is None else pd.Index(columns)
    )
    if index is not None:
        frame = frame.set_index(index)
    return frame


def state_multiindex(node_labels, indices, alphabet=None) -> pd.MultiIndex:
    """A MultiIndex over all states of ``indices``, level-named by label.

    ``alphabet`` is the per-unit cardinality sequence (k-ary); if ``None`` the
    units are binary.
    """
    from pyphi.utils import all_states

    spec = alphabet if alphabet is not None else len(indices)
    states = list(all_states(spec))
    names = list(node_labels.coerce_to_labels(indices))
    return pd.MultiIndex.from_tuples(states, names=names)


def distribution_rows(
    direction, kind, purview, repertoire, node_labels=None
) -> list[dict[str, Any]]:
    """Tidy ``{direction, kind, purview, state, probability}`` rows for one
    repertoire.

    States are enumerated from the repertoire's per-purview-unit cardinality
    (k-ary aware). ``purview`` renders as labels when ``node_labels`` is given,
    else as integer indices. Returns ``[]`` for a ``None`` repertoire.
    """
    from pyphi import distribution
    from pyphi.utils import all_states

    if repertoire is None:
        return []
    repertoire = np.asarray(repertoire)
    alphabet = [repertoire.shape[i] for i in purview]
    flat = distribution.flatten(repertoire)
    assert flat is not None
    states = list(all_states(alphabet)) if alphabet else [()]
    if node_labels is None:
        purview_labels: tuple[Any, ...] = tuple(purview)
    else:
        purview_labels = tuple(node_labels.coerce_to_labels(purview))
    direction_label = str(direction)
    return [
        {
            "direction": direction_label,
            "kind": kind,
            "purview": purview_labels,
            "state": tuple(state),
            "probability": float(prob),
        }
        for state, prob in zip(states, flat, strict=True)
    ]


class ToPandasMixin:
    """Export a result object to a labeled Pandas structure.

    ``to_pandas()`` returns a ``Series`` for scalar-record types and a
    ``DataFrame`` with a labeled index for collections and distributions.
    Units render as labels. Subclasses implement ``_pandas_record()`` (record
    types, which inherit the Series-building ``_to_pandas``) or override
    ``_to_pandas()`` (collections and distributions).
    """

    def to_pandas(self) -> pd.Series | pd.DataFrame:
        """Return a labeled Pandas view of this object."""
        return self._to_pandas()

    def _to_pandas(self) -> pd.Series | pd.DataFrame:
        return record_to_series(self._pandas_record(), name=type(self).__name__)

    def _pandas_record(self) -> Mapping[str, Any]:
        raise NotImplementedError(type(self).__name__)
