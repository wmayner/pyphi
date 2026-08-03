`Analysis` — the type `pyphi.analyze()` returns — is now serializable. It
round-trips through `pyphi.serialize.dumps`/`loads` under every formalism and
gains `.save()` / `.load()`, so a whole analysis can be written to disk and read
back with its system, SIA, and cause-effect structure intact. Previously only
the `.ces` component could be saved.
