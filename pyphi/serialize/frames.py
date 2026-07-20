"""Exact serialization of pandas DataFrames as embedded parquet.

The frame is written with its named index levels reset to columns, so any
index shape that occurs on PyPhi result tables round-trips exactly —
including levels whose entries are tuples. Object columns holding tuples
are recorded by name at encode time; parquet represents them as lists, and
the recorded names make the restoration exact rather than heuristic. NaN,
None cells, and column dtypes survive bit-exactly via pyarrow.
"""

import io

import numpy as np
import pandas as pd

from . import schema


def dataframe_to_schema(df: pd.DataFrame) -> schema.DataFrameSchema:
    names = list(df.index.names)
    if names == [None]:
        if not isinstance(df.index, pd.RangeIndex):
            raise ValueError(
                "cannot serialize a DataFrame with an unnamed, non-default index"
            )
        reset = df
        index_columns: tuple[str, ...] = ()
    else:
        if any(name is None for name in names):
            raise ValueError("cannot serialize a DataFrame with unnamed index levels")
        reset = df.reset_index()
        index_columns = tuple(str(name) for name in names)
    tuple_columns = tuple(
        str(column)
        for column in reset.columns
        if reset[column].dtype == object
        and any(isinstance(value, tuple) for value in reset[column])
    )
    buffer = io.BytesIO()
    reset.to_parquet(buffer, engine="pyarrow", index=False)
    return schema.DataFrameSchema(
        parquet=buffer.getvalue(),
        index_columns=index_columns,
        tuple_columns=tuple_columns,
    )


def _as_tuple(value):
    if value is None:
        return None
    return tuple(x.item() if isinstance(x, np.generic) else x for x in value)


def schema_to_dataframe(struct: schema.DataFrameSchema) -> pd.DataFrame:
    df = pd.read_parquet(io.BytesIO(struct.parquet), engine="pyarrow")
    for column in struct.tuple_columns:
        df[column] = [_as_tuple(value) for value in df[column]]
    if struct.index_columns:
        df = df.set_index(list(struct.index_columns))
        if len(struct.index_columns) == 1:
            df.index = pd.Index(
                df.index, name=struct.index_columns[0], tupleize_cols=False
            )
    return df
