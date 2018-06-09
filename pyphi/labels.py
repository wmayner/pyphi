#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# labels.py

"""
Helper class representing labels of network nodes.
"""

import collections

from pyphi import validate
from pyphi.models import cmp


def default_label(index):
    """Default label for a node."""
    return "n{}".format(index)


def default_labels(indices):
    """Default labels for serveral nodes."""
    return tuple(default_label(i) for i in indices)


class NodeLabels(collections.Sequence):
    '''Text labels for nodes in a network.

    Labels can either be instantiated as a tuple of strings:

        >>> NodeLabels(('A', 'IN'), (0, 1))
        NodeLabels(('A', 'IN'))

    Or, if all labels are a single character, as a string:

        >>> NodeLabels('AB', (0, 1))
        NodeLabels(('A', 'B'))
    '''
    def __init__(self, labels, node_indices):
        if labels is None:
            labels = default_labels(node_indices)

        self.labels = tuple(label for label in labels)
        self.node_indices = node_indices

        validate.node_labels(self.labels, node_indices)

        # Dicts mapping indices to labels and vice versa
        self._l2i = dict(zip(self.labels, self.node_indices))
        self._i2l = dict(zip(self.node_indices, self.labels))

    def __len__(self):
        return len(self.labels)

    def __iter__(self):
        return iter(self.labels)

    def __contains__(self, x):
        return x in self.labels

    def __getitem__(self, x):
        return self.labels[x]

    def __repr__(self):
        return 'NodeLabels({})'.format(self.labels)

    @cmp.sametype
    def __eq__(self, other):
        return (self.labels == other.labels and
                self.node_indices == other.node_indices)

    def __hash__(self):
        return hash((self.labels, self.node_indices))

    def labels2indices(self, labels):
        """Convert a tuple of node labels to node indices."""
        return tuple(self._l2i[label] for label in labels)

    def indices2labels(self, indices):
        """Convert a tuple of node indices to node labels."""
        return tuple(self._i2l[index] for index in indices)

    def coerce_to_indices(self, nodes):
        """Return the nodes indices for nodes, where ``nodes`` is either
        already integer indices or node labels.
        """
        if nodes is None:
            return self.node_indices

        if all(isinstance(node, str) for node in nodes):
            indices = self.labels2indices(nodes)
        else:
            indices = map(int, nodes)
        return tuple(sorted(set(indices)))

    def to_json(self):
        return {'labels': self.labels, 'node_indices': self.node_indices}
