`CompositionalState` fixes: an empty (no-argument) state is fully usable; a
purview claimed in only one direction no longer raises KeyError from
`conflicts_with`; and `resolve_conflicts` ranks candidates by live conflict
counts as resolution proceeds, keeping mechanisms the stale ranking used to
discard.
