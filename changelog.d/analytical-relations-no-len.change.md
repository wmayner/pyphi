`AnalyticalRelations` no longer defines `len()`: closed-form relations
hold no relation objects (iteration already raises with guidance), and
a length would work on small systems while exceeding `len()`'s range on
large ones. Use `.num_relations()` for the exact count — the tutorials
now do. `len()` remains on the enumerable containers
(`ConcreteRelations`, `RelationSample`).
