"""Round-trip serialization of SweepResult and OptimizationResult."""

import math

import pandas as pd
import pytest

from pyphi.serialize import frames


class TestDataFrameCodec:
    def _roundtrip(self, df):
        return frames.schema_to_dataframe(frames.dataframe_to_schema(df))

    def test_multiindex_tuple_levels(self):
        idx = pd.MultiIndex.from_arrays(
            [
                ["IIT_4_0_2026", "IIT_4_0_2026", "IIT_3_0"],
                [(0, 1), (0, 1, 2), (0, 1)],
                [(1, 0), (1, 1, 0), (0, 0)],
            ],
            names=["formalism", "subset", "state"],
        )
        df = pd.DataFrame(
            {
                "phi": [0.5, math.nan, 1.25],
                "is_irreducible": [True, False, True],
                "effectively_tied": [None, True, None],
                "n": [3, 4, 5],
            },
            index=idx,
        )
        pd.testing.assert_frame_equal(self._roundtrip(df), df)

    def test_single_named_tuple_index(self):
        df = pd.DataFrame(
            {"phi": [0.5, 0.7]},
            index=pd.Index([(0, 1), (1, 0)], name="state", tupleize_cols=False),
        )
        back = self._roundtrip(df)
        pd.testing.assert_frame_equal(back, df)
        # Tuple entries stay scalar index entries (not a MultiIndex).
        assert back.index[0] == (0, 1)

    def test_range_index_with_nan_and_tuples(self):
        df = pd.DataFrame(
            {
                "eval": [0, 1, 2],
                "objective": [0.1, math.nan, 0.3],
                "partition": ["a", None, "b"],
                "cause_state": [(1, 0), None, (0, 1)],
            }
        )
        back = self._roundtrip(df)
        pd.testing.assert_frame_equal(back, df)
        assert back["cause_state"][0] == (1, 0)
        assert back["cause_state"][1] is None

    def test_unnamed_nondefault_index_raises(self):
        df = pd.DataFrame({"x": [1, 2]}, index=pd.Index([10, 20]))
        with pytest.raises(ValueError, match="unnamed"):
            frames.dataframe_to_schema(df)
