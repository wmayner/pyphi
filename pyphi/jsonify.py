#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# jsonify.py

"""
PyPhi- and NumPy-aware JSON serialization.
"""

import json
from collections import Iterable

import numpy as np


def jsonify(obj):
    """Return a JSON-encodable representation of an object, recursively using
    any available ``to_json`` methods, converting NumPy arrays and datatypes to
    native lists and types along the way."""
    try:
        # Call the `to_json` method if available.
        return obj.to_json()
    except:
        # If we have a numpy array, convert it to a list.
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # If we have NumPy datatypes, convert them to native types.
        if isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.float64):
            return float(obj)
        # Recursively jsonify the object dictionary, if available.
        if hasattr(obj, '__dict__'):
            return {key: jsonify(value) for key, value in obj.__dict__.items()}
        # If we have an iterable, recurse on the items (but not for strings,
        # which will recurse forver).
        if isinstance(obj, Iterable) and not isinstance(obj, str):
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
              'sort_keys': True,
              'cls': PyPhiJSONEncoder}
    kwargs.update(user_kwargs)
    return json.dumps(obj, **kwargs)


def dump(obj, fp, **user_kwargs):
    """Serialize ``obj`` as a JSON-formatted stream and write to ``fp`` (a
    ``.write()``-supporting file-like object."""
    kwargs = {'separators': (',', ':'),
              'sort_keys': True,
              'cls': PyPhiJSONEncoder}
    kwargs.update(user_kwargs)
    return json.dump(obj, fp, **kwargs)


# TODO implement proper loading into PyPhi objects.
loads = json.loads
load = json.load
