`pyphi-mcp install` now registers the Python interpreter it was run with
(`python -m pyphi.mcp`) instead of a `uvx` command resolving `pyphi[mcp]`. The
client starts the server from the environment PyPhi was installed into, with no
`PATH` lookup and no package resolution at startup. Pass `--from
<specification>` for the `uvx` form. Running `install` from the throwaway
environment that `uv run --with` or `uvx` builds is refused, since a client
could not launch it again.
