import json
import tempfile

import numpy as np
import pytest

from pyphi import Direction
from pyphi import config
from pyphi import exceptions
from pyphi import jsonify
from pyphi import labels
from pyphi import models
from pyphi import substrate
from pyphi.formalism import iit3
from pyphi.formalism import iit4 as new_big_phi
from pyphi.measures.distribution import resolve_mechanism_measure
from pyphi.measures.distribution import resolve_system_measure


def test_jsonify_native():
    x = {"list": [1, 2.0, 3], "tuple": (1, 2, 3), "bool": [True, False], "null": None}
    answer = {
        "list": [1, 2.0, 3],
        "tuple": [1, 2, 3],
        "bool": [True, False],
        "null": None,
    }
    assert answer == json.loads(jsonify.dumps(x))


def test_jsonify_numpy():
    x = {
        "ndarray": np.array([1, 2]),
        "np.int32": np.int32(1),
        "np.int64": np.int64(2),
        "np.float64": np.float64(3),
    }
    answer = {
        "ndarray": [1, 2],
        "np.int32": 1,
        "np.int64": 2,
        "np.float64": 3.0,
    }
    assert answer == json.loads(jsonify.dumps(x))


def test_json_deserialization(s, transition):
    objects = [
        Direction.CAUSE,
        s.substrate,  # Substrate
        s,  # System
        models.JointBipartition(models.Part((0,), ()), models.Part((1,), (2, 3))),
        models.JointPartition(models.Part((0,), ()), models.Part((1,), (2, 3))),
        models.JointTripartition(
            models.Part((0,), ()), models.Part((1,), (2, 3)), models.Part((3,), (4,))
        ),
        models.DirectedBipartition(Direction.EFFECT, (0,), (2,)),
        models.NullCut((0, 1)),
        models.DirectedJointPartition(
            Direction.CAUSE,
            models.JointPartition(models.Part((0,), ()), models.Part((1,), (2, 3))),
        ),
        # s.concept((1, 2)),
        # s.concept((1,)),
        iit3.ces(s),
        # iit3.sia(s),
        s.sia(),
        new_big_phi.ces(
            s,
            system_measure=resolve_system_measure(
                config.formalism.iit.system_phi_measure
            ),
            specification_measure=resolve_mechanism_measure(
                config.formalism.iit.specification_measure
            ),
        ),
        transition,
        # transition.find_actual_cause((0,), (0,)),
        # actual.account(transition),
        # actual.sia(transition),
        labels.NodeLabels("AB", (0, 1)),
    ]
    for o in objects:
        print(type(o))
        loaded = jsonify.loads(jsonify.dumps(o))
        assert loaded == o


def test_json_deserialization_non_pyphi_clasess():
    class OtherObject:
        def __init__(self, x):
            self.x = x

    loaded = jsonify.loads(jsonify.dumps(OtherObject(1)))
    assert loaded == {"x": 1}


# NOTE: test_deserialization_memoizes_duplicate_objects was removed because
# it relied on ces.system which was intentionally removed from Distinctions
# during the IIT 3.0 -> 4.0 migration.


@pytest.fixture
def substrate_file(standard):
    # delete_on_close=False allows reopening the file by name on Windows
    # (Windows doesn't allow opening a file that's already open)
    with tempfile.NamedTemporaryFile(mode="w+", delete_on_close=False) as f:
        jsonify.dump(standard, f)
        f.seek(0)
        yield f


def test_load(substrate_file, standard):
    assert jsonify.load(substrate_file) == standard


def test_substrate_from_json(substrate_file, standard):
    loaded_substrate = substrate.from_json(substrate_file.name)
    assert loaded_substrate == standard
    assert np.array_equal(loaded_substrate.node_labels, standard.node_labels)


def test_version_check_during_deserialization(s):
    import pyphi

    string = jsonify.dumps(s)

    # Change the version
    _obj = json.loads(string)
    _obj[jsonify.VERSION_KEY] = "0.1.bogus"
    string = json.dumps(_obj)

    # Re-enable version validation (disabled globally in conftest.py)
    with (
        pyphi.config.override(validate_json_version=True),
        pytest.raises(exceptions.JSONVersionError),
    ):
        jsonify.loads(string)


# Tests for DistanceResult JSON serialization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_distance_result_json_serialization_basic():
    """Test basic DistanceResult serialization without auxiliary data."""
    from pyphi.measures.distribution import DistanceResult

    result = DistanceResult(0.5)
    loaded = jsonify.loads(jsonify.dumps(result))

    assert loaded == result
    assert float(loaded) == 0.5


def test_distance_result_json_serialization_with_metadata():
    """Test DistanceResult serialization with auxiliary metadata."""
    from pyphi.measures.distribution import DistanceResult

    result = DistanceResult(
        0.5, method="INTRINSIC_DIFFERENTIATION", asymmetric=True, state=(1, 0, 0)
    )
    loaded = jsonify.loads(jsonify.dumps(result))

    assert loaded == result
    assert float(loaded) == 0.5
    assert loaded.method == "INTRINSIC_DIFFERENTIATION"
    assert loaded.asymmetric is True
    assert loaded.state == (1, 0, 0)


def test_distance_result_json_preserves_all_attributes():
    """Test that all auxiliary attributes are preserved during serialization."""
    from pyphi.measures.distribution import DistanceResult

    result = DistanceResult(
        0.75, method="EMD", direction="CAUSE", custom_attr="custom_value", num=42
    )
    loaded = jsonify.loads(jsonify.dumps(result))

    assert loaded == result
    assert loaded.method == "EMD"
    assert loaded.direction == "CAUSE"
    assert loaded.custom_attr == "custom_value"
    assert loaded.num == 42


# Tests for enum-keyed dict JSON serialization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_enum_keyed_dict_serialization():
    """Test that dicts with enum keys are properly serialized and deserialized."""
    from pyphi.measures.distribution import DistanceResult

    original = {
        Direction.CAUSE: DistanceResult(0.5, method="GID"),
        Direction.EFFECT: DistanceResult(0.3, method="GID"),
    }

    loaded = jsonify.loads(jsonify.dumps(original))

    assert loaded == original
    assert Direction.CAUSE in loaded
    assert Direction.EFFECT in loaded
    assert loaded[Direction.CAUSE] == DistanceResult(0.5, method="GID")
    assert loaded[Direction.EFFECT] == DistanceResult(0.3, method="GID")


def test_enum_keyed_dict_preserves_metadata():
    """Test that enum-keyed dicts preserve DistanceResult metadata."""
    from pyphi.measures.distribution import DistanceResult

    original = {
        Direction.CAUSE: DistanceResult(
            0.0, method="INTRINSIC_DIFFERENTIATION", asymmetric=True, state=(1, 0, 0)
        ),
        Direction.EFFECT: DistanceResult(
            0.0, method="INTRINSIC_DIFFERENTIATION", asymmetric=True, state=(1, 0, 0)
        ),
    }

    loaded = jsonify.loads(jsonify.dumps(original))

    assert loaded[Direction.CAUSE].method == "INTRINSIC_DIFFERENTIATION"
    assert loaded[Direction.CAUSE].asymmetric is True
    assert loaded[Direction.CAUSE].state == (1, 0, 0)
    assert loaded[Direction.EFFECT].method == "INTRINSIC_DIFFERENTIATION"


def test_enum_keyed_dict_with_mixed_types():
    """Test enum-keyed dicts with various value types."""
    original = {
        Direction.CAUSE: 0.5,
        Direction.EFFECT: [1, 2, 3],
        Direction.BIDIRECTIONAL: {"nested": "dict"},
    }

    loaded = jsonify.loads(jsonify.dumps(original))

    # Lists become tuples in PyPhi, so compare values individually
    assert loaded[Direction.CAUSE] == 0.5
    assert loaded[Direction.EFFECT] == (1, 2, 3)  # Lists become tuples in PyPhi
    assert loaded[Direction.BIDIRECTIONAL] == {"nested": "dict"}


def test_regular_dict_not_affected():
    """Test that regular dicts (without enum keys) are not affected."""
    original = {"key1": "value1", "key2": 42, "key3": [1, 2, 3]}

    loaded = jsonify.loads(jsonify.dumps(original))

    assert loaded == {"key1": "value1", "key2": 42, "key3": (1, 2, 3)}


def test_nested_enum_keyed_dicts():
    """Test that nested structures with enum-keyed dicts work correctly."""
    from pyphi.measures.distribution import DistanceResult

    original = {
        "outer": {
            Direction.CAUSE: DistanceResult(0.5),
            Direction.EFFECT: DistanceResult(0.3),
        },
        "other": "value",
    }

    loaded = jsonify.loads(jsonify.dumps(original))

    assert loaded["outer"][Direction.CAUSE] == DistanceResult(0.5)
    assert loaded["outer"][Direction.EFFECT] == DistanceResult(0.3)
    assert loaded["other"] == "value"


def test_empty_enum_keyed_dict():
    """Test that empty dicts are handled correctly."""
    original = {}
    loaded = jsonify.loads(jsonify.dumps(original))
    assert loaded == original
