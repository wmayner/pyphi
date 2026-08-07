Capped the `mcp` extra below version 2.0. The 2.0 release removes `mcp.server.fastmcp`, which `pyphi-mcp` is built on, so `pip install "pyphi[mcp]"` resolved to a server that failed at import.
