The welcome message is written to stderr rather than stdout. The `pyphi-mcp`
server speaks JSON-RPC over stdout, so importing PyPhi emitted the banner into
the protocol stream ahead of the first message unless the user had set
`PYPHI_WELCOME_OFF`. Scripts that captured the banner from stdout will no longer
see it there.
