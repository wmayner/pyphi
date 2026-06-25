# pyright: strict
# labels.py
"""Helper class representing labels of substrate nodes."""

from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Sequence

import numpy as np

from . import validate
from .conf import config
from .conf import fallback
from .models import cmp


def default_label(index: int) -> str:
    """Default label for a node."""
    return f"n{index}"


def default_labels(indices: Sequence[int]) -> tuple[str, ...]:
    """Default labels for serveral nodes."""
    return tuple(default_label(i) for i in indices)


class NodeLabels(Sequence[str]):
    """Text labels for nodes in a substrate.

    Labels can either be instantiated as a tuple of strings:

        >>> NodeLabels(('A', 'IN'), (0, 1))
        NodeLabels(('A', 'IN'))

    Or, if all labels are a single character, as a string:

        >>> NodeLabels('AB', (0, 1))
        NodeLabels(('A', 'B'))
    """

    def __init__(
        self,
        labels: str | Sequence[str] | None,
        node_indices: Sequence[int],
    ) -> None:
        if labels is None:
            labels = default_labels(node_indices)

        self.labels: tuple[str, ...] = tuple(label for label in labels)
        self.node_indices: tuple[int, ...] = tuple(node_indices)

        validate.node_labels(self.labels, self.node_indices)

        # Dicts mapping indices to labels and vice versa
        self._l2i = dict(zip(self.labels, self.node_indices, strict=False))
        self._i2l = dict(zip(self.node_indices, self.labels, strict=False))

    def __len__(self) -> int:
        return len(self.labels)

    def __iter__(self) -> Iterator[str]:
        return iter(self.labels)

    def __contains__(self, x: object) -> bool:
        return x in self.labels

    def __getitem__(self, x: int | slice) -> str | tuple[str, ...]:  # type: ignore[override]
        return self.labels[x]

    def __repr__(self) -> str:
        return f"NodeLabels({self.labels})"

    @cmp.sametype
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NodeLabels):
            return NotImplemented
        return self.labels == other.labels and self.node_indices == other.node_indices

    def __hash__(self) -> int:
        return hash((self.labels, self.node_indices))

    def index2label(self, index: int) -> str:
        return self._i2l[index]

    def label2index(self, label: str) -> int:
        return self._l2i[label]

    def labels2indices(self, labels: Sequence[str]) -> tuple[int, ...]:
        """Convert a tuple of node labels to node indices."""
        return tuple(self._l2i[label] for label in labels)

    def indices2labels(self, indices: Sequence[int]) -> tuple[str, ...]:
        """Convert a tuple of node indices to node labels."""
        return tuple(self._i2l[index] for index in indices)

    def coerce_to_indices(
        self,
        nodes: Iterable[int | str | np.integer] | None,
    ) -> tuple[int, ...]:
        """Return the nodes indices for nodes, where ``nodes`` is either
        already integer indices or node labels.
        """
        if nodes is None:
            return self.node_indices

        nodes_list = list(nodes)  # Materialize to allow multiple iteration
        if all(isinstance(node, str) for node in nodes_list):
            indices = self.labels2indices(tuple(nodes_list))  # type: ignore[arg-type]
        else:
            indices = map(int, nodes_list)
        return tuple(sorted(set(indices)))

    def coerce_to_labels(
        self,
        nodes: Iterable[int | str | np.integer] | None,
    ) -> tuple[str | int, ...]:
        """Return the nodes labels for nodes, where ``nodes`` is either
        already labels or node indices.
        """
        if nodes is None:
            return self.node_indices

        nodes_list = list(nodes)  # Materialize to allow multiple iteration
        if all(isinstance(node, (int, np.integer)) for node in nodes_list):
            labels: Sequence[str | int] = self.indices2labels(
                tuple(int(n) for n in nodes_list)
            )
        else:
            # Convert any np.integer to int for type compatibility
            labels = [str(n) if isinstance(n, str) else int(n) for n in nodes_list]
        return tuple(labels)

    def label_string(
        self,
        nodes: Iterable[int | str | np.integer] | None,
        state: Sequence[int],
        sep: str | None = None,
    ) -> str:
        """Return a single string labeling the nodes."""
        separator = fallback(
            sep,
            config.infrastructure.label_separator,
        )
        assert separator is not None, "LABEL_SEPARATOR must be set in config"
        return separator.join(
            self.set_case_by_state(self.coerce_to_labels(nodes), state)
        )

    def set_case_by_state(
        self,
        labels: Sequence[str | int],
        states: Sequence[int],
    ) -> list[str]:
        """Return a list of labels with case set by the corresponding state."""
        return [
            str(label).upper() if state else str(label).lower()
            for label, state in zip(labels, states, strict=True)
        ]
