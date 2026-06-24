"""Semantic display tones (cause / effect) for direction-aware coloring."""

_TONES = frozenset({"cause", "effect"})


def tone_of(label: object) -> str | None:
    """Map a direction-like label to a display tone.

    Accepts a :class:`~pyphi.direction.Direction` or a string like ``"CAUSE"`` /
    ``"EFFECT"``; returns ``"cause"`` / ``"effect"`` (lowercased) or ``None`` for
    anything else (e.g. ``BIDIRECTIONAL``). HTML rendering colors these; ASCII
    ignores them.
    """
    name = str(label).lower()
    return name if name in _TONES else None
