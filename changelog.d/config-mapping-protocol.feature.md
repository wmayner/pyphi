The :data:`pyphi.config` facade now implements the Mapping protocol:
:func:`iter`, :func:`len`, ``in``, ``keys()``, ``values()``, ``items()``,
and ``get(path, default)``. Callers can enumerate all leaf settings as
dotted paths in dataclass declaration order::

    >>> for path in pyphi.config:
    ...     print(path, "=", pyphi.config[path])
    numerics.precision = 13
    formalism.iit.version = IIT_4_0_2023
    ...

The existing ``config[path]`` indexer now also accepts bare leaf keys
(``config["precision"]``) as a shortcut for the fully-qualified path
(``config["numerics.precision"]``), mirroring the routing that
``config.precision`` attribute access already used.
