``PyPhiFloat`` comparison and hash semantics fixed.

- ``__eq__`` and ``__ne__`` now return ``NotImplemented`` for non-numeric
  arguments instead of ``False`` / ``True``. Python's reflected-comparison
  protocol now kicks in correctly, letting the other operand decide
  equality. The previous behavior could mask bugs in equality dispatch.

- ``__hash__`` now snapshots ``config.PRECISION`` at construction (stored
  as a private instance attribute). A ``PyPhiFloat`` placed in a set or
  dict keeps a stable hash even if ``config.PRECISION`` is later changed;
  without this, set/dict membership lookups silently returned wrong
  answers after a precision change.

The trade-off: a ``PyPhiFloat`` constructed under one precision and
compared against one constructed under another uses each operand's own
construction-time precision for hash purposes. ``==`` continues to use
the current ``config.PRECISION``. The asymmetry is documented; in
practice ``config.PRECISION`` is set once at start-up and not mutated.

``DistanceResult`` (which subclasses ``PyPhiFloat``) was updated to
exclude underscore-prefixed instance attributes from JSON serialization,
``__repr__``, and copy operations, so the new ``_precision`` snapshot
doesn't leak into user-visible output.
