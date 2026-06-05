The `distinction_phi_normalization` cache-staleness warning now states the
value transition (e.g. `'NUM_CONNECTIONS_CUT' -> 'NONE'`) and explains why a
scoped `config.override` emits it twice (once applying the change, once
restoring the previous value on exit, which concerns Systems computed inside
the block).
