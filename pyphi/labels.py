class NodeLabels:
    '''
    TODO: validate labels for duplicates
    TODO: pass in indices if defaults are generated here
    '''
    def __init__(self, labels):
        self.labels = labels
        self.node_indices = tuple(range(len(labels)))

    def labels2indices(self, labels):
        """Convert a tuple of node labels to node indices."""
        _map = dict(zip(self.labels, self.node_indices))
        return tuple(_map[label] for label in labels)

    def indices2labels(self, indices):
        """Convert a tuple of node indices to node labels."""
        _map = dict(zip(self.node_indices, self.labels))
        return tuple(_map[index] for index in indices)

    def parse_node_indices(self, nodes):
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
