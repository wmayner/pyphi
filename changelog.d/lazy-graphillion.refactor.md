Defer the `graphillion` import in `pyphi.relations` and
`pyphi.combinatorics` to function bodies. `import pyphi` no longer
loads graphillion eagerly — only callers that explicitly compute
relations or set families pay the cost. This unblocks the use of
free-threaded CPython 3.13+ for workers that don't compute relations
(graphillion's `_graphillion` C extension does not declare
`PyMod_GIL_NOT_USED`, so loading it re-enables the GIL process-wide).
