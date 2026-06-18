Added `pyphi.registry.InstanceRegistry`, a registry base for named instances (as
opposed to `Registry`, which stores callables). The formalism registries now subclass
it, so a registry lookup is typed as the formalism instance rather than a callable —
removing a cluster of type-checker suppressions at the dispatch sites.
