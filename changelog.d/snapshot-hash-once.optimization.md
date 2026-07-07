The config-snapshot identity hash is now computed once when the worker
function is built, instead of `hash(repr(snapshot))` (~1.3 ms) on every
mapped item. This removes a per-item overhead that dominated
scheduler-dispatched workloads with cheap items — e.g. relation-candidate
evaluation ran ~250x faster sequentially and ~5x faster in parallel after
the change.
