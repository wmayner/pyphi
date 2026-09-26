# Use PyPhi with an AI assistant

Two tools help an AI assistant with work on IIT, and they are meant to be used
together:

- **IIT Expert** answers questions about the theory from its primary
  literature: the IIT wiki, the papers, and a glossary of the axioms,
  postulates and measures. It cites the paper, section or equation each claim
  comes from. It is hosted, so there is nothing to run locally.
- **The PyPhi MCP server** does the computing. It builds substrates, estimates
  what an analysis will cost, runs it, and plots the result. It runs locally,
  in the Python environment where PyPhi is installed.

IIT Expert is a work in progress. Its corpus and glossary are still being
checked against the sources.

## IIT Expert

IIT Expert has two parts. The connector gives the assistant the sources, and
the skill tells it to read them before answering instead of relying on what it
already believes about IIT. The plugin installs both.

**Claude Code**

```bash
claude plugin marketplace add wmayner/iit-expert-plugin
claude plugin install iit-expert@iit-expert
```

**Codex**

```bash
codex plugin marketplace add wmayner/iit-expert-plugin
codex plugin add iit-expert@iit-expert
```

**Cursor:** open Customize → From GitHub Repository and enter
`wmayner/iit-expert-plugin`.

**claude.ai and Claude Desktop** add the connector and the skill separately.
The steps are at <https://learniit.org/install>, which is also where the
instructions are kept up to date as IIT Expert changes.

If you set up the PyPhi server with `pyphi-mcp install`, it offers to run the
Claude Code and Codex commands for you.

## The PyPhi MCP server

In a uv project:

```bash
uv add "pyphi[mcp]"
uv run pyphi-mcp install
```

{doc}`mcp-server` explains what `install` writes, how to connect other
clients, and which tools the server provides.
