`len()` on `AnalyticalRelations` whose closed-form relation count
exceeds `sys.maxsize` now raises an `OverflowError` that points at
`.num_relations()`, instead of the bare protocol failure; the MCP
result summary falls back accordingly.
