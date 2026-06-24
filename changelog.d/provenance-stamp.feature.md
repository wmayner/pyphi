Top-level results (`SystemIrreducibilityAnalysis`, `CauseEffectStructure`, `AcSIA`) now
carry a `provenance` record (pyphi version, source revision and dirty flag, timestamp,
wall-time, and Python/numpy/scipy/platform versions) alongside the existing `config`
snapshot, so a saved result is self-describing. A new `repr_verbosity` level (`4`)
displays it. Call `result.with_provenance(note=..., seed=...)` to record your own context
(a free-form `note` or the `seed` you controlled) on a computed result.
