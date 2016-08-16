#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# jsonify.py

"""
PyPhi- and NumPy-aware JSON serialization.
"""

import json

import numpy as np

import pyphi


def _jsonify_dict(dct):
    return {key: jsonify(value) for key, value in dct.items()}


def jsonify(obj):
    """Return a JSON-encodable representation of an object, recursively using
    any available ``to_json`` methods, converting NumPy arrays and datatypes to
    native lists and types along the way."""
    try:
        # Call the `to_json` method if available.
        d = obj.to_json()
        d['__class__'] = obj.__class__.__name__
        return jsonify(d)
    except AttributeError:
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

    """Extension of the default JSONEncoder that allows for serializing PyPhi
    objects with ``jsonify``."""

    def encode(self, obj):
        """Encode the output of ``jsonify`` with the default encoder."""
        return super().encode(jsonify(obj))


def dumps(obj, **user_kwargs):
    """Serialize ``obj`` as JSON-formatted stream."""
    kwargs = {'separators': (',', ':'),
              'cls': PyPhiJSONEncoder}
    kwargs.update(user_kwargs)
    return json.dumps(obj, **kwargs)


def dump(obj, fp, **user_kwargs):
    """Serialize ``obj`` as a JSON-formatted stream and write to ``fp`` (a
    ``.write()``-supporting file-like object."""
    kwargs = {'separators': (',', ':'),
              'cls': PyPhiJSONEncoder}
    kwargs.update(user_kwargs)
    return json.dump(obj, fp, **kwargs)


def pyphi_classes():
    return {
        'Network': pyphi.Network,
        'Subsystem': pyphi.Subsystem,
        'Cut': pyphi.models.Cut,
        'Part': pyphi.models.Part,
        'Bipartition': pyphi.models.Bipartition,
    }


def _load_object(d):
    if isinstance(d, dict):
        d = {k: _load_object(v) for k, v in d.items()}

        if '__class__' in d:
            cls = pyphi_classes()[d['__class__']]
            return cls.from_json(d)

    if isinstance(d, list):
        return tuple(d)

    return d


def loads(string):
    d = json.loads(string)
    return _load_object(d)


load = json.load
