Added `ConfigSnapshot.as_overrides()`: a full-fidelity override mapping using
qualified dotted paths for all formalism fields, so
`pyphi.config.override(**snap.as_overrides())` reproduces a snapshotted
configuration exactly — including the formalism version and other fields whose
bare names collide between the IIT and actual-causation sub-namespaces.
