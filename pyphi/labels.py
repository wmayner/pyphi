from pyphi import validate


def default_label(index):
    """Default label for a node."""
    return "n{}".format(index)


def default_labels(indices):
    """Default labels for serveral nodes."""
    return tuple(default_label(i) for i in indices)


class NodeLabels:
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
        if not nodes:
            indices = ()
        elif all(isinstance(node, str) for node in nodes):
            indices = self.labels2indices(nodes)
        else:
            indices = map(int, nodes)
        return tuple(sorted(set(indices)))

    def __repr__(self):
        return 'NodeLabels({})'.format(self.labels)

    def __eq__(self, other):
        return (self.labels == other.labels and
                self.node_indices == other.node_indices)

    def __hash__(self):
        return hash((self.labels, self.node_indices))

    def to_json(self):
        return {'labels': self.labels, 'node_indices': self.node_indices}
