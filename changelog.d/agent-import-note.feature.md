PyPhi now prints a short note to stderr when it is imported under an AI coding
agent, naming the two mistakes that most often go wrong unaided — reporting φₛ
as Φ, and the little-endian state convention — and pointing at the bundled
reference. It is written to stderr, never stdout, so it cannot corrupt the MCP
server's JSON-RPC stream. Suppress it with `PYPHI_AGENT_NOTE_OFF=1` or the new
`agent_note_off` infrastructure option; `welcome_off` controls only the welcome
message. Any harness can opt in by setting `PYPHI_AGENT`.
