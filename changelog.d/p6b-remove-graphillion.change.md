Removed the `graphillion` dependency. Concrete relations enumeration is now
pure Python, so the macOS libomp source-build is no longer required and the
relations path is free-threading (no-GIL) safe. The internal
`pyphi.combinatorics.powerset_family` / `union_powerset_family` helpers were
removed.
