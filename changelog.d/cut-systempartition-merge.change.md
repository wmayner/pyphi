**Breaking (2.0):** ``Cut`` and ``CompleteGeneralSetPartition`` removed.

The ``Cut`` class is replaced by :class:`pyphi.models.SystemPartition`,
which adds a mandatory ``Direction`` argument as the first positional. The
two classes always represented the same mathematical object — a
unidirectional cut of a system — but the legacy ``Cut`` left direction
implicit and ``SystemPartition`` made it explicit. The 2.0 hard-break
unifies them; IIT 3.0 callers (which previously did not need a direction)
should pass ``Direction.EFFECT`` to preserve behavior — the IIT 3.0
phi-computation paths do not read the direction field, so the choice has
no semantic effect on phi values::

    # before
    Cut(from_nodes, to_nodes)
    # after
    SystemPartition(Direction.EFFECT, from_nodes, to_nodes)

``CompleteGeneralSetPartition`` is removed — it was a one-method subclass
of ``CompleteGeneralKCut`` differing only in its ``__str__`` output.
``CompleteGeneralKCut`` now serves both roles; callers needing the
"Complete\\n" prefix can format manually.

``CompleteSystemPartition`` now inherits from ``_CutBase`` (was a
standalone class — inconsistent with the rest of the cut hierarchy).
