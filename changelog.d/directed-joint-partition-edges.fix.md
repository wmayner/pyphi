`DirectedJointPartition.removed_edges` and `cut_matrix` no longer include
edges into mechanism nodes outside each part: severed edges now run only into
the rest of the whole purview under the partition's direction ordering,
matching the undirected `JointPartition` semantics.
