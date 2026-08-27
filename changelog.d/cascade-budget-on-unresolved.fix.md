`resolve_ties.cascade` now honors its documented contract when the
escalation budget blocks a level: a lone surviving candidate resolves
instead of returning unresolved, and `on_unresolved='fail'`/`'warn'` raise
or warn on a budget-blocked tie just as they do when the cascade exhausts
its levels. The `"NONE"` tie-resolution strategy is also accepted in list
form (`["NONE"]`), matching the bare-string form instead of raising
`NotImplementedError`.
