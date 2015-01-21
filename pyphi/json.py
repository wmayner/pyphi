#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# json.py
"""
PyPhi- and NumPy-aware JSON codec.
"""

from collections import Iterable
import numpy as np
import json as _json

from . import __version__


def get_stamp(obj):
    """Returns a dictionary with the key 'pyphi', containing the object's class
    name and current PyPhi version."""
    return {
        'pyphi': {
            'class': type(obj).__name__,
            'version': __version__
        }
    }


def make_encodable(obj):
    """Return a JSON-encodable representation of an object, recursively using
    any available ``json_dict`` methods, and NumPy's ``tolist`` function for
    arrays."""
    # Use the ``json_dict`` method if available, stamping it with the class
    # name and the current PyPhi version.
    if hasattr(obj, 'json_dict'):
        d = obj.json_dict()
        # Stamp it!
        # TODO stamp?
        # d.update(get_stamp(obj))
        return d
    # If we have an array, convert it to a list.
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # If we have an iterable, recurse on the items. But not for strings, which
    # are reCURSED! (they will recurse forver).
    elif isinstance(obj, Iterable) and not isinstance(obj, str):
        return [make_encodable(item) for item in obj]
    # Otherwise, just return it.
    else:
        return obj


class JSONEncoder(_json.JSONEncoder):

    """
    An extension of the built-in JSONEncoder that can handle native PyPhi
    objects as well as NumPy arrays.

    Uses the ``json_dict`` method for PyPhi objects.
    """

    def encode(self, obj):
        """Encode using the object's ``json_dict`` method if exists, falling
        back on the built-in encoder if not."""
        try:
            return super().encode(obj.json_dict())
        except AttributeError:
            return super().encode(obj)


def dumps(obj):
    """Serialize ``obj`` to a JSON formatted ``str``."""
    # Use our encoder and compact separators.
    return _json.dumps(obj, cls=JSONEncoder, separators=(',', ':'))


class JSONDecoder(_json.JSONDecoder):

    """
    An extension of the built-in JSONDecoder that can handle native PyPhi
    objects as well as NumPy arrays.
    """
    pass


def loads(s):
    """Deserialize ``s`` (a ``str`` instance containing a JSON document) to a
    PyPhi object."""
    return _json.loads(s)
