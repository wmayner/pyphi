# models/pandas.py
"""Utilities for working with Pandas data structures."""

from typing import Sequence

import pandas as pd

# TODO Just use `to_json` instead of `to_dict`?


def try_to_dict(obj):
    try:
        return obj.to_dict()
    except AttributeError:
        return obj


class ToDictFromExplicitAttrsMixin:
    """Mixin class for converting a class to a dict from the `_dict_attrs` list."""

    def to_dict(self):
        """Convert the object to a dict."""
        if hasattr(self, "_dict_attrs"):
            return {attr: try_to_dict(getattr(self, attr)) for attr in self._dict_attrs}
        raise NotImplementedError("no `_dict_attrs` attribute")


class ToDictMixin:
    """Mixin class for converting a class to a dict from the object's ___dict___."""

    def to_dict(self):
        """Convert the object to a dict."""
        return {
            attr: try_to_dict(value)
            for attr, value in self.__dict__.items()
            if not attr.startswith("_")
        }


class ToPandasMixin:
    """Mixin class for converting a class to a Pandas data structure."""

    def to_pandas(self):
        """Convert the object to a Pandas data structure."""
        pandas_type = pd.Series
        if hasattr(self, "to_json"):
            data = self.to_json()
            if isinstance(data, Sequence):
                data = [try_to_dict(d) for d in data]
                pandas_type = pd.DataFrame
        elif hasattr(self, "to_dict"):
            data = self.to_dict()

        df = pd.json_normalize(data)

        if pandas_type is pd.Series:
            if len(df) == 1:
                series = df.iloc[0]
                series.name = self.__class__.__name__
                return series
            else:
                raise ValueError(f"expected single row, got {len(df)}")

        return df
