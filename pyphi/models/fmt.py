# models/fmt.py
"""Helper functions for formatting pretty representations of PyPhi models."""

from fractions import Fraction
from typing import Any

from pyphi import numerics
from pyphi.conf import config

# Unicode symbols
SMALL_PHI = "φ"
HORIZONTAL_BAR = "─"
LINE = "━"
ARROW_RIGHT = LINE * 2 + "▶"
EMPTY_SET = "∅"

NICE_DENOMINATORS = [*list(range(16)), 16, 32, 64, 128]


def labels(
    indices: tuple[int, ...], node_labels: object | None = None
) -> tuple[str, ...]:
    """Get the labels for a tuple of mechanism indices."""
    if node_labels is None:
        return tuple(map(str, indices))
    return node_labels.indices2labels(indices)  # type: ignore[attr-defined]


def fmt_number(p: Any) -> str:
    """Format a number.

    It will be printed as a fraction if the denominator isn't too big and as a
    decimal otherwise.

    If formatting fails, return the input unmodified.
    """
    try:
        formatted = format(p, f".{config.numerics.precision}f")
    except (ValueError, TypeError):
        return str(p)

    if not config.infrastructure.print_fractions:
        return formatted

    fraction = Fraction(p)
    nice = fraction.limit_denominator(128)
    return (
        str(nice)
        if (
            numerics.eq(float(fraction), float(nice))
            and nice.denominator in NICE_DENOMINATORS
        )
        else formatted
    )


def fmt_nodes(nodes: tuple[int, ...], node_labels: object | None = None) -> str:
    """Format nodes, optionally with labels."""
    return ",".join(labels(nodes, node_labels)) if nodes else EMPTY_SET


def fmt_mechanism(indices: tuple[int, ...], node_labels: object | None = None) -> str:
    """Format a mechanism or purview."""
    return "[" + fmt_nodes(indices, node_labels=node_labels) + "]"


def fmt_part(part: object, node_labels: object | None = None) -> str:
    """Format a :class:`~pyphi.models.partitions.Part`.

    The returned string looks like::

        0,1
        ───
         ∅
    """
    numer = fmt_nodes(part.mechanism, node_labels=node_labels)  # type: ignore[attr-defined]
    denom = fmt_nodes(part.purview, node_labels=node_labels)  # type: ignore[attr-defined]

    w = max(3, len(numer), len(denom))
    divider = HORIZONTAL_BAR * w
    return ("{numer:^{width}}\n{divider}\n{denom:^{width}}").format(
        numer=numer, divider=divider, denom=denom, width=w
    )


def fmt_partition_arrow(
    cut: object, direction: object | None = None, name: bool = True
) -> str:
    """Format a :class:`~pyphi.models.partitions.DirectedBipartition` as an
    arrow expression.

    The arrow always points from ``from_nodes`` to ``to_nodes`` — the
    connections the cut severs (matching :meth:`removed_edges` and the
    severed-connections grid). The causal ``direction`` the cut is evaluated
    against is annotated textually.
    """
    try:
        if name:
            name_str = cut.__class__.__name__ + " "
        else:
            name_str = ""
        direction_str = (
            "" if direction is None else f"({direction.name.lower()}) "  # type: ignore[attr-defined]
        )
        from_nodes = fmt_mechanism(cut.from_nodes, cut.node_labels)  # type: ignore[attr-defined]
        to_nodes = fmt_mechanism(cut.to_nodes, cut.node_labels)  # type: ignore[attr-defined]
        symbol = ARROW_RIGHT if direction is None else LINE * 2 + "/ /" + ARROW_RIGHT
        return f"{name_str}{direction_str}{from_nodes} {symbol} {to_nodes}"
    except AttributeError:
        return str(cut)


def fmt_directed_joint_partition(cut: object) -> str:
    """Format a :class:`~pyphi.models.partitions.DirectedJointPartition`."""
    return f"DirectedJointPartition {cut.direction}\n{cut.partition}"  # type: ignore[attr-defined]


def fmt_extended_purview(
    extended_purview: Any, node_labels: object | None = None
) -> str:
    """Format an extended purview."""
    if len(extended_purview) == 1:
        return fmt_mechanism(extended_purview[0], node_labels=node_labels)

    purviews = [
        fmt_mechanism(purview, node_labels=node_labels) for purview in extended_purview
    ]
    return "[" + ", ".join(purviews) + "]"


def fmt_transition(t: object) -> str:
    """Format a :class:`~pyphi.actual.Transition`."""
    cause = fmt_mechanism(t.cause_indices, t.node_labels)  # type: ignore[attr-defined]
    effect = fmt_mechanism(t.effect_indices, t.node_labels)  # type: ignore[attr-defined]
    return f"Transition({cause} {ARROW_RIGHT} {effect})"


def state(state: tuple[int, ...]) -> str:
    """Format a state."""
    return "(" + ",".join(map(str, state)) + ")"
