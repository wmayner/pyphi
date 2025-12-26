# data_structures/array_like.py

from numbers import Number

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin


class ArrayLike(NDArrayOperatorsMixin):
    # Only support operations with instances of _HANDLED_TYPES.
    _HANDLED_TYPES = (np.ndarray, list, Number)

    # TODO(tpm) populate this list
    _TYPE_CLOSED_FUNCTIONS = (
        np.concatenate,
        np.stack,
        np.all,
        np.sum,
    )

    # Holds the underlying array
    _VALUE_ATTR = "value"

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = kwargs.get("out", ())
        for x in inputs + out:
            # Only support operations with instances of _HANDLED_TYPES.
            # Use ArrayLike instead of type(self) for isinstance to
            # allow subclasses that don't override __array_ufunc__ to
            # handle ArrayLike objects.
            if not isinstance(x, self._HANDLED_TYPES + (ArrayLike,)):
                return NotImplemented

        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = tuple(self._unwrap_arraylike(inputs))
        if out:
            kwargs["out"] = tuple(self._unwrap_arraylike(out))
        result = getattr(ufunc, method)(*inputs, **kwargs)

        if type(result) is tuple:
            # Multiple return values
            return tuple(type(self)(x) for x in result)
        elif method == "at":
            # No return value
            return None
        else:
            # one return value
            return type(self)(result)

    @staticmethod
    def _unwrap_arraylike(values):
        return (
            getattr(x, x._VALUE_ATTR) if isinstance(x, ArrayLike) else x for x in values
        )

    def __array_function__(self, func, types, args, kwargs):
        if func not in self._TYPE_CLOSED_FUNCTIONS:
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __array_function__ to handle MyArray objects
        if not all(issubclass(t, ArrayLike) for t in types):
            return NotImplemented
        # extract wrapped array-like objects from args
        updated_args = []

        for arg in args:
            if hasattr(arg, self._VALUE_ATTR):
                updated_args.append(arg.__getattribute__(self._VALUE_ATTR))
            else:
                updated_args.append(arg)

        # defer to NumPy implementation
        result = func(*updated_args, **kwargs)

        # cast to original wrapper if possible
        return type(self)(result) if type(result) in self._HANDLED_TYPES else result

    def __array__(self, dtype=None):
        # TODO(tpm) We should use `np.asarray` instead of accessing `.tpm`
        # whenever the underlying array is needed
        return np.asarray(self.__getattribute__(self._VALUE_ATTR), dtype=dtype)

    def __getattr__(self, name):
        return getattr(self.__getattribute__(self._VALUE_ATTR), name)
