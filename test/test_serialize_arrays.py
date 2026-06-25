import numpy as np
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from pyphi.serialize.arrays import array_to_bytes
from pyphi.serialize.arrays import bytes_to_array


@settings(max_examples=100)
@given(
    hnp.arrays(
        dtype=st.sampled_from([np.float64, np.float32, np.int64, np.bool_]),
        shape=hnp.array_shapes(min_dims=0, max_dims=4, max_side=6),
    )
)
def test_array_round_trip_is_bit_identical(arr):
    restored = bytes_to_array(array_to_bytes(arr))
    assert restored.dtype == arr.dtype
    assert restored.shape == arr.shape
    # Compare raw bytes for true bit-identity (captures NaN bit patterns,
    # which value equality treats as unequal).
    assert restored.tobytes() == arr.tobytes()


def test_non_contiguous_array_round_trips():
    arr = np.arange(24, dtype=np.float64).reshape(4, 6)[:, ::2]
    assert not arr.flags["C_CONTIGUOUS"]
    restored = bytes_to_array(array_to_bytes(arr))
    assert np.array_equal(restored, arr)
