Config layers now store immutable containers: tie-resolution sequences are
tuples and the per-level parallel mappings are read-only, so presets and
`ConfigSnapshot`s can no longer be corrupted through shared references, and
`IITConfig` is hashable.
