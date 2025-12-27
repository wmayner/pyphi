# models/pandas.py
"""Utilities for working with Pandas data structures."""

from collections.abc import Sequence
from typing import Any

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


class ToPandasMixin:
    """Mixin class for converting a class to a Pandas data structure."""

    def to_pandas(self) -> pd.Series | pd.DataFrame:
        """Convert the object to a Pandas data structure."""
        pandas_type: type[pd.Series] | type[pd.DataFrame] = pd.Series
        if hasattr(self, "to_json"):
            data: Any = self.to_json()
            if isinstance(data, Sequence):
                data = [try_to_dict(d) for d in data]
                pandas_type = pd.DataFrame
        elif hasattr(self, "to_dict"):
            data = self.to_dict()

        df: pd.DataFrame = pd.json_normalize(data)

        if pandas_type is pd.Series:
            if len(df) == 1:
                series = df.iloc[0]
                series.name = self.__class__.__name__
                return series
            raise ValueError(f"expected single row, got {len(df)}")

        return df
