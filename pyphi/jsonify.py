#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# jsonify.py

# TODO: extend to `macro` objects
# TODO: resolve schema issues with `vphi` and other external consumers
# TODO: somehow check schema instead of version?

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
"""

import json

import numpy as np

import pyphi
from pyphi import cache

CLASS_KEY = '__class__'
VERSION_KEY = '__version__'
ID_KEY = '__id__'


def _loadable_models():
    """A dictionary of loadable PyPhi models.

    These are stored in this function (instead of module scope) to resolve
    circular import issues.
    """
    classes = [
        pyphi.Direction,
        pyphi.Network,
        pyphi.Subsystem,
        pyphi.Transition,
        pyphi.labels.NodeLabels,
        pyphi.models.Cut,
        pyphi.models.KCut,
        pyphi.models.NullCut,
        pyphi.models.Part,
        pyphi.models.Bipartition,
        pyphi.models.KPartition,
        pyphi.models.Tripartition,
        pyphi.models.RepertoireIrreducibilityAnalysis,
        pyphi.models.MaximallyIrreducibleCauseOrEffect,
        pyphi.models.MaximallyIrreducibleCause,
        pyphi.models.MaximallyIrreducibleEffect,
        pyphi.models.Concept,
        pyphi.models.CauseEffectStructure,
        pyphi.models.SystemIrreducibilityAnalysis,
        pyphi.models.ActualCut,
        pyphi.models.AcRepertoireIrreducibilityAnalysis,
        pyphi.models.CausalLink,
        pyphi.models.Account,
        pyphi.models.AcSystemIrreducibilityAnalysis
    ]
    return {cls.__name__: cls for cls in classes}


def _jsonify_dict(dct):
    return {key: jsonify(value) for key, value in dct.items()}


def _push_metadata(dct, obj):
    dct.update({
        CLASS_KEY: obj.__class__.__name__,
        VERSION_KEY: pyphi.__version__,
        ID_KEY: hash(obj)
    })
    return dct


def _get_metadata(dct):
    return dct[CLASS_KEY], dct[VERSION_KEY], dct[ID_KEY]


def _pop_metadata(dct):
    return dct.pop(CLASS_KEY), dct.pop(VERSION_KEY), dct.pop(ID_KEY)


def jsonify(obj):  # pylint: disable=too-many-return-statements
    """Return a JSON-encodable representation of an object, recursively using
    any available ``to_json`` methods, converting NumPy arrays and datatypes to
    native lists and types along the way.
    """
    # Call the `to_json` method if available and add metadata.
    if hasattr(obj, 'to_json'):
        d = obj.to_json()
        _push_metadata(d, obj)
        return jsonify(d)

    # If we have a numpy array, convert it to a list.
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # If we have NumPy datatypes, convert them to native types.
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.float64):
        return float(obj)

    # Recurse over dictionaries.
    if isinstance(obj, dict):
        return _jsonify_dict(obj)

    # Recurse over object dictionaries.
    if hasattr(obj, '__dict__'):
        return _jsonify_dict(obj.__dict__)

    # Recurse over lists and tuples.
    if isinstance(obj, (list, tuple)):
        return [jsonify(item) for item in obj]

    # Otherwise, give up and hope it's serializable.
    return obj


class PyPhiJSONEncoder(json.JSONEncoder):
    """JSONEncoder that allows serializing PyPhi objects with ``jsonify``."""

    def encode(self, obj):  # pylint: disable=arguments-differ
        """Encode the output of ``jsonify`` with the default encoder."""
        return super().encode(jsonify(obj))

    def iterencode(self, obj, **kwargs):  # pylint: disable=arguments-differ
        """Analog to `encode` used by json.dump."""
        return super().iterencode(jsonify(obj), **kwargs)


def _encoder_kwargs(user_kwargs):
    """Update kwargs for `dump` and `dumps` to use the PyPhi encoder."""
    kwargs = {'separators': (',', ':'), 'cls': PyPhiJSONEncoder}
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


def _check_version(version):
    """Check whether the JSON version matches the PyPhi version."""
    if version != pyphi.__version__:
        raise pyphi.exceptions.JSONVersionError(
            'Cannot load JSON from a different version of PyPhi. '
            'JSON version = {0}, current version = {1}.'.format(
                version, pyphi.__version__))


def _is_model(dct):
    """Check if ``dct`` is a PyPhi model serialization."""
    return CLASS_KEY in dct


class _ObjectCache(cache.DictCache):
    """Cache mapping ids to loaded objects, keyed by the id of the object."""

    def key(self, dct, **kwargs):  # pylint: disable=arguments-differ
        return _get_metadata(dct)


class PyPhiJSONDecoder(json.JSONDecoder):
    """Extension of the default encoder which automatically deserializes
    PyPhi JSON to the appropriate model classes.
    """

    def __init__(self, *args, **kwargs):
        kwargs['object_hook'] = self._load_object
        super().__init__(*args, **kwargs)

        # Memoize available models
        self._models = _loadable_models()

        # Cache for loaded objects
        self._object_cache = _ObjectCache()

    def _load_object(self, obj):
        """Recursively load a PyPhi object.

        PyPhi models are recursively loaded, using the model metadata to
        recreate the original object relations. Lists are cast to tuples
        because most objects in PyPhi which are serialized to lists (eg.
        mechanisms and purviews) are ultimately tuples. Other lists (tpms,
        repertoires) should be cast to the correct type in init methods.
        """
        if isinstance(obj, dict):
            obj = {k: self._load_object(v) for k, v in obj.items()}
            # Load a serialized PyPhi model
            if _is_model(obj):
                return self._load_model(obj)

        elif isinstance(obj, list):
            return tuple(self._load_object(item) for item in obj)

        return obj

    @cache.method('_object_cache')
    def _load_model(self, dct):
        """Load a serialized PyPhi model.

        The object is memoized for reuse elsewhere in the object graph.
        """
        classname, version, _ = _pop_metadata(dct)

        _check_version(version)
        cls = self._models[classname]

        # Use `from_json` if available
        if hasattr(cls, 'from_json'):
            return cls.from_json(dct)

        # Default to object constructor
        return cls(**dct)


def loads(string):
    """Deserialize a JSON string to a Python object."""
    return json.loads(string, cls=PyPhiJSONDecoder)


def load(fp):
    """Deserialize a JSON stream to a Python object."""
    return json.load(fp, cls=PyPhiJSONDecoder)
