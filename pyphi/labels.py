from pyphi import validate


class NodeLabels:
    '''
    TODO: validate labels for duplicates
    TODO: pass in indices if defaults are generated here
    '''
    def __init__(self, labels, node_indices):
        self.labels = labels
        self.node_indices = node_indices

        validate.node_labels(labels, node_indices)

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
