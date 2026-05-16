# jsonify.py
"""
PyPhi- and NumPy-aware JSON serialization.

To be properly serialized and deserialized, PyPhi objects must implement a
``to_json`` method which returns a dictionary of attribute names and attribute
values. These attributes should be the names of arguments passed to the object
constructor. If the constructor takes additional, fewer, or different
arguments, the object needs to implement a custom ``classmethod`` called
``from_json`` that takes a Python dictionary as an argument and returns a PyPhi
object. For example::

    class Phi:
        def __init__(self, phi):
            self.phi = phi

        def to_json(self):
            return {'phi': self.phi, 'twice_phi': 2 * self.phi}

        @classmethod
        def from_json(cls, json):
            return Phi(json['phi'])

The object must also be added to ``jsonify._loadable_models``.

The JSON encoder adds the name of the object and the current PyPhi version to
the JSON stream. The JSON decoder uses this metadata to recursively deserialize
the stream to a nested PyPhi object structure. The decoder will raise an
exception if current PyPhi version doesn't match the version in the JSON data.

Canonical JSON shapes for the primary result types
==================================================

Result objects are discriminated by the ``__class__`` metadata key added by
``_push_metadata``; no separate formalism discriminator field is used. Each
class's ``to_json`` emits its field set, and ``from_json`` reconstructs from
the same set.

**IIT 3.0 SIA** — ``__class__: "IIT3SystemIrreducibilityAnalysis"``::

    {"__class__": "IIT3SystemIrreducibilityAnalysis",
     "__version__": "...", "__id__": ...,
     "phi": ..., "partition": ...,
     "partitioned_distinctions": ...,
     "current_state": ..., "node_indices": ..., "node_labels": ...}

**IIT 4.0 SIA** — ``__class__: "SystemIrreducibilityAnalysis"``::

    {"__class__": "SystemIrreducibilityAnalysis",
     "__version__": "...", "__id__": ...,
     "phi": ..., "partition": ...,
     "normalized_phi": ..., "signed_phi": ...,
     "signed_normalized_phi": ..., "cause": ..., "effect": ...,
     "system_state": ..., "current_state": ..., "node_indices": ...,
     "intrinsic_differentiation": ..., "config": ...}

**CauseEffectStructure** — ``__class__: "CauseEffectStructure"``; the shape
is shared by both IIT 3.0 and IIT 4.0 (the inner ``sia`` carries its
formalism via its own ``__class__``)::

    {"__class__": "CauseEffectStructure",
     "__version__": "...", "__id__": ...,
     "sia": {...}, "distinctions": ..., "relations": ...}

**AcSystemIrreducibilityAnalysis** — ``__class__:
"AcSystemIrreducibilityAnalysis"``::

    {"__class__": "AcSystemIrreducibilityAnalysis",
     "__version__": "...", "__id__": ...,
     "alpha": ..., "direction": ...,
     "account": ..., "partitioned_account": ...,
     "partition": ..., "before_state": ..., "after_state": ...,
     "size": ..., "node_indices": ...,
     "cause_indices": ..., "effect_indices": ..., "node_labels": ...}
"""

import json
from importlib.metadata import version as get_version
from typing import TYPE_CHECKING

import numpy as np

import pyphi
from pyphi import cache

if TYPE_CHECKING:
    # These imports are needed for type checking but cause circular imports at runtime
    # They are dynamically available through pyphi's lazy import system
    pass

CLASS_KEY = "__class__"
VERSION_KEY = "__version__"
ID_KEY = "__id__"
ENUM_DICT_MARKER = "__enum_dict__"
ENUM_CLASS_KEY = "__enum_class__"

PYPHI_VERSION = get_version("pyphi")


def _parse_version(version_str: str) -> tuple[str, str | None]:
    """Parse version into (base_version, dev_suffix).

    Examples:
        "2.0.0a1" -> ("2.0.0a1", None)
        "1.2.1.dev1534+g12345" -> ("1.2.1", "dev1534+g12345")
    """
    # Check for .dev suffix pattern (common with hatch-vcs)
    if ".dev" in version_str:
        parts = version_str.split(".dev", 1)
        return parts[0], "dev" + parts[1]
    return version_str, None


# TODO: extend to `macro` objects
# TODO: resolve schema issues with `vphi` and other external consumers
# TODO: somehow check schema instead of version?
# TODO(4.0): ensure that sets/lists/tuples are cast to the correct type in
#            __init__ methods so loading works properly
# TODO(4.0): to_dict() instead?


def _loadable_models():
    """A dictionary of loadable PyPhi models.

    These are stored in this function (instead of module scope) to resolve
    circular import issues.

    Note: pyright cannot statically verify these module attributes because they
    are loaded dynamically through pyphi's lazy import system. These all work
    correctly at runtime.
    """
    classes = [
        pyphi.data_structures.PyPhiFloat,  # pyright: ignore[reportAttributeAccessIssue]
        pyphi.Direction,
        pyphi.labels.NodeLabels,  # pyright: ignore[reportAttributeAccessIssue]
        pyphi.measures.distribution.DistanceResult,  # pyright: ignore[reportAttributeAccessIssue]
        pyphi.models.Account,  # pyright: ignore[reportAttributeAccessIssue]
        pyphi.models.AcRepertoireIrreducibilityAnalysis,  # pyright: ignore[reportAttributeAccessIssue]
        pyphi.models.AcSystemIrreducibilityAnalysis,  # pyright: ignore[reportAttributeAccessIssue]
        pyphi.models.CausalLink,  # pyright: ignore[reportAttributeAccessIssue]
        pyphi.models.CompleteEdgeCut,  # pyright: ignore[reportAttributeAccessIssue]
        pyphi.models.Concept,  # pyright: ignore[reportAttributeAccessIssue]
        pyphi.models.DirectedBipartition,  # pyright: ignore[reportAttributeAccessIssue]
        pyphi.models.DirectedJointPartition,  # pyright: ignore[reportAttributeAccessIssue]
        pyphi.models.DirectedSetPartition,  # pyright: ignore[reportAttributeAccessIssue]
        pyphi.models.Distinctions,  # pyright: ignore[reportAttributeAccessIssue]
        pyphi.models.EdgeCut,  # pyright: ignore[reportAttributeAccessIssue]
        pyphi.models.JointBipartition,  # pyright: ignore[reportAttributeAccessIssue]
        pyphi.models.JointPartition,  # pyright: ignore[reportAttributeAccessIssue]
        pyphi.models.ResolvedDistinctions,  # pyright: ignore[reportAttributeAccessIssue]
        pyphi.models.UnresolvedDistinctions,  # pyright: ignore[reportAttributeAccessIssue]
        pyphi.models.MaximallyIrreducibleCause,  # pyright: ignore[reportAttributeAccessIssue]
        pyphi.models.MaximallyIrreducibleCauseOrEffect,  # pyright: ignore[reportAttributeAccessIssue]
        pyphi.models.MaximallyIrreducibleEffect,  # pyright: ignore[reportAttributeAccessIssue]
        pyphi.models.StateSpecification,  # pyright: ignore[reportAttributeAccessIssue]
        pyphi.models.NullCut,  # pyright: ignore[reportAttributeAccessIssue]
        pyphi.models.Part,  # pyright: ignore[reportAttributeAccessIssue]
        pyphi.models.RepertoireIrreducibilityAnalysis,  # pyright: ignore[reportAttributeAccessIssue]
        pyphi.models.SystemStateSpecification,  # pyright: ignore[reportAttributeAccessIssue]
        pyphi.models.IIT3SystemIrreducibilityAnalysis,  # pyright: ignore[reportAttributeAccessIssue]
        pyphi.models.JointTripartition,  # pyright: ignore[reportAttributeAccessIssue]
        pyphi.Substrate,
        pyphi.formalism.iit4.CauseEffectStructure,  # pyright: ignore[reportAttributeAccessIssue]
        pyphi.formalism.iit4.SystemIrreducibilityAnalysis,  # pyright: ignore[reportAttributeAccessIssue]
        pyphi.formalism.iit4.NullSystemIrreducibilityAnalysis,  # pyright: ignore[reportAttributeAccessIssue]
        pyphi.formalism.iit4.NullCauseEffectStructure,  # pyright: ignore[reportAttributeAccessIssue]
        pyphi.relations.AnalyticalRelations,  # pyright: ignore[reportAttributeAccessIssue]
        pyphi.relations.ConcreteRelations,  # pyright: ignore[reportAttributeAccessIssue]
        pyphi.relations.NullRelations,  # pyright: ignore[reportAttributeAccessIssue]
        pyphi.relations.Relation,  # pyright: ignore[reportAttributeAccessIssue]
        pyphi.relations.RelationFace,  # pyright: ignore[reportAttributeAccessIssue]
        pyphi.System,
        pyphi.Transition,
    ]
    return {cls.__name__: cls for cls in classes}


def _jsonify_dict(dct):
    """Convert a dictionary to a JSON-serializable format.

    Dicts with enum keys are converted to a special format with a marker and
    the enum class name, allowing proper reconstruction during deserialization.
    JSON only supports string keys, so we serialize enum-keyed dicts as lists
    of [key, value] pairs with metadata.
    """
    from enum import Enum

    # Check if any keys are enums
    has_enum_keys = any(isinstance(key, Enum) for key in dct)

    if has_enum_keys:
        # Get the enum class from the first enum key
        enum_class = None
        for key in dct:
            if isinstance(key, Enum):
                enum_class = key.__class__.__name__
                break

        # Convert to list of [key, value] pairs
        # Enum keys are serialized using their to_json() method or as their value
        pairs = []
        for key, value in dct.items():
            json_key = jsonify(key)
            json_value = jsonify(value)
            pairs.append([json_key, json_value])

        return {
            ENUM_DICT_MARKER: pairs,
            ENUM_CLASS_KEY: enum_class,
        }
    # Normal dict without enum keys
    return {key: jsonify(value) for key, value in dct.items()}


def _push_metadata(dct, obj):
    dct.update(
        {
            CLASS_KEY: obj.__class__.__name__,
            VERSION_KEY: PYPHI_VERSION,
            ID_KEY: hash(obj),
        }
    )
    return dct


def _get_metadata(dct):
    return dct[CLASS_KEY], dct[VERSION_KEY], dct[ID_KEY]


def _pop_metadata(dct):
    return dct.pop(CLASS_KEY), dct.pop(VERSION_KEY), dct.pop(ID_KEY)


def jsonify(obj):  # noqa: PLR0911
    """Return a JSON-encodable representation of an object, recursively using
    any available ``to_json`` methods, converting NumPy arrays and datatypes to
    native lists and types along the way.
    """
    # Call the `to_json` method if available and add metadata.
    if hasattr(obj, "to_json"):
        d = obj.to_json()
        if isinstance(d, dict):
            _push_metadata(d, obj)
        return jsonify(d)

    # If we have a numpy array, convert it to a list.
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # If we have NumPy datatypes, convert them to native types.
    # Use numpy's abstract base types to avoid generic type issues
    if isinstance(obj, np.integer):  #  pyright: ignore[reportArgumentType]
        return int(obj)
    if isinstance(obj, np.floating):  # pyright: ignore[reportArgumentType]
        return float(obj)

    # Handle Python Enums by converting to their name (string)
    # This prevents circular reference issues with Enum.__dict__
    # Using .name instead of .value ensures stability across refactoring
    from enum import Enum

    if isinstance(obj, Enum):
        return obj.name

    # Recurse over dictionaries.
    if isinstance(obj, dict):
        return _jsonify_dict(obj)

    # Recurse over object dictionaries.
    if hasattr(obj, "__dict__"):
        dct = _jsonify_dict(obj.__dict__)
        # Push metadata if the model is registered as loadable
        if _is_loadable_model_object(obj):
            _push_metadata(dct, obj)
        return dct

    # Recurse over lists, tuples, sets, and frozensets.
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [jsonify(item) for item in obj]

    # Otherwise, give up and hope it's serializable.
    return obj


class PyPhiJSONEncoder(json.JSONEncoder):
    """JSONEncoder that allows serializing PyPhi objects with ``jsonify``."""

    def encode(self, o):  # pylint: disable=arguments-differ
        """Encode the output of ``jsonify`` with the default encoder."""
        return super().encode(jsonify(o))

    def iterencode(self, o, _one_shot=False):  # pylint: disable=arguments-differ
        """Analog to `encode` used by json.dump."""
        return super().iterencode(jsonify(o), _one_shot=_one_shot)


def _encoder_kwargs(user_kwargs):
    """Update kwargs for `dump` and `dumps` to use the PyPhi encoder."""
    kwargs = {"separators": (",", ":"), "cls": PyPhiJSONEncoder}
    kwargs.update(user_kwargs)

    return kwargs


def dumps(obj, **user_kwargs):
    """Serialize ``obj`` as JSON-formatted stream."""
    return json.dumps(obj, **_encoder_kwargs(user_kwargs))


def dump(obj, fp, **user_kwargs):
    """Serialize ``obj`` as a JSON-formatted stream and write to ``fp`` (a
    ``.write()``-supporting file-like object.
    """
    return json.dump(obj, fp, **_encoder_kwargs(user_kwargs))


def _check_version(version: str) -> None:
    """Check whether the JSON version is compatible with current PyPhi version.

    The check can be disabled via ``config.infrastructure.validate_json_version``.
    When enabled, versions are considered compatible if:
    1. They match exactly, OR
    2. Their base versions match (ignoring .dev suffixes from hatch-vcs)
    """
    if not pyphi.config.infrastructure.validate_json_version:
        return

    if version == PYPHI_VERSION:
        return

    json_base, _ = _parse_version(version)
    current_base, _ = _parse_version(PYPHI_VERSION)

    if json_base == current_base:
        # Same base version, just different dev builds
        return

    raise pyphi.exceptions.JSONVersionError(  # pyright: ignore[reportAttributeAccessIssue]
        f"Cannot load JSON from incompatible PyPhi version. "
        f"JSON version = {version}, current version = {PYPHI_VERSION}."
    )


def _is_loadable_model_object(obj):
    return obj.__class__.__name__ in _loadable_models()


class _ObjectCache(cache.DictCache):
    """Cache mapping ids to loaded objects, keyed by the id of the object."""

    def key(self, dct, **kwargs):  # pylint: disable=arguments-differ
        return _get_metadata(dct)


class PyPhiJSONDecoder(json.JSONDecoder):
    """Extension of the default encoder which automatically deserializes
    PyPhi JSON to the appropriate model classes.
    """

    def __init__(self, *args, **kwargs):
        kwargs["object_hook"] = self._load_object
        super().__init__(*args, **kwargs)

        # Cache for loaded objects
        self._object_cache = _ObjectCache()

    def _load_object(self, obj):
        """Recursively load a PyPhi object.

        PyPhi models are recursively loaded, using the model metadata to
        recreate the original object relations. Lists are cast to tuples
        because most objects in PyPhi which are serialized to lists (eg.
        mechanisms and purviews) are ultimately tuples. Other lists (TPMs,
        repertoires) should be cast to the correct type in init methods.
        """
        if isinstance(obj, dict):
            # Check if this is a serialized enum-keyed dict
            if ENUM_DICT_MARKER in obj and ENUM_CLASS_KEY in obj:
                return self._load_enum_dict(obj)

            obj = {k: self._load_object(v) for k, v in obj.items()}
            # Load a serialized PyPhi model
            if _is_loadable_model_dict(obj):
                return self._load_model(obj)

        # TODO(4.0) remove?
        elif isinstance(obj, list):
            return tuple(self._load_object(item) for item in obj)

        return obj

    def _load_enum_dict(self, obj):
        """Reconstruct a dictionary with enum keys from its serialized form.

        The serialized form is a dict with ENUM_DICT_MARKER containing a list
        of [key, value] pairs and ENUM_CLASS_KEY containing the enum class name.
        """
        enum_class_name = obj[ENUM_CLASS_KEY]
        pairs = obj[ENUM_DICT_MARKER]

        # Get the enum class from loadable models
        # Direction enums are loadable, so we can get them from there
        enum_class = _loadable_models().get(enum_class_name)

        if enum_class is None:
            # If not in loadable models, try to import from pyphi
            import pyphi

            enum_class = getattr(pyphi, enum_class_name, None)

        if enum_class is None:
            raise ValueError(f"Unknown enum class: {enum_class_name}")

        # Reconstruct the dict with enum keys
        result = {}
        for key_data, value_data in pairs:
            # Recursively load the key and value
            key = self._load_object(key_data)
            value = self._load_object(value_data)

            # If the key is a dict with CLASS_KEY == enum_class_name,
            # it's a serialized enum. Need to convert it back to the actual enum value
            if isinstance(key, dict) and key.get(CLASS_KEY) == enum_class_name:
                # This is a serialized Direction enum with {"direction": "CAUSE"}
                if hasattr(enum_class, "from_json"):
                    key = enum_class.from_json(key)
                else:
                    # Fallback: try to get by name
                    direction_name = key.get("direction")
                    if direction_name:
                        key = enum_class[direction_name]

            result[key] = value

        return result

    @cache.method("_object_cache")
    def _load_model(self, dct):
        """Load a serialized PyPhi model.

        The object is memoized for reuse elsewhere in the object graph.
        """
        classname, version, _ = _pop_metadata(dct)

        _check_version(version)
        cls = _loadable_models()[classname]

        # Use `from_json` if available
        if hasattr(cls, "from_json"):
            return cls.from_json(dct)

        # Default to object constructor
        return cls(**dct)


def _is_loadable_model_dict(dct):
    """Check if ``dct`` is a PyPhi model serialization."""
    return CLASS_KEY in dct


def loads(string):
    """Deserialize a JSON string to a Python object."""
    return json.loads(string, cls=PyPhiJSONDecoder)


def load(fp):
    """Deserialize a JSON stream to a Python object."""
    return json.load(fp, cls=PyPhiJSONDecoder)
