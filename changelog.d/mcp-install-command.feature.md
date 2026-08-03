Added `pyphi-mcp install`, which sets the MCP server up in a project: it
registers the server in the client's configuration and writes a short block of
PyPhi facts — φₛ versus Φ, little-endian states, the cost of an analysis — into
the project's `AGENTS.md`, with an `@AGENTS.md` import added to `CLAUDE.md`
because Claude Code reads that name instead. A server's `instructions` reach
only an assistant that connects to it, while a project's instruction file is
read before anything else happens. The block sits between markers so a later
install refreshes it without touching surrounding content, and
`pyphi-mcp uninstall` removes both halves. `--from`, `--scope`, `--client`,
`--print` and `--force` cover the variants; with no subcommand `pyphi-mcp`
still runs the server.
