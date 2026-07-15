# Use PyPhi with an AI assistant (MCP server)

PyPhi includes a [Model Context Protocol](https://modelcontextprotocol.io) (MCP)
server that lets an AI assistant build substrates, run IIT analyses, inspect the
results, visualize them, and read a grounded reference on the theory — all by
talking to it in plain language. It runs locally in your own environment.

It is meant to be useful in two ways: as a calculator and interpreter for
researchers using AI assistance, and as a tutor for newcomers exploring the
theory.

## Install

Install PyPhi with the `mcp` extra. From PyPI:

```bash
pip install "pyphi[mcp]"
```

Or install the latest code straight from GitHub:

```bash
pip install "pyphi[mcp] @ git+https://github.com/wmayner/pyphi.git"
```

Plotting needs the visualization stack, so add that extra too if you want it:

```bash
pip install "pyphi[mcp,visualize]"
```

### Claude Code

Add the server with `claude mcp add` — everything after `--` is the launch
command. From PyPI:

```bash
claude mcp add pyphi -- uvx --from "pyphi[mcp]" pyphi-mcp
```

Or, for the latest code from GitHub:

```bash
claude mcp add pyphi -- uvx --from "pyphi[mcp] @ git+https://github.com/wmayner/pyphi.git" pyphi-mcp
```

This writes an entry to `.mcp.json` (or `~/.claude.json`), which you can also
create by hand:

```json
{
  "mcpServers": {
    "pyphi": {
      "command": "uvx",
      "args": ["--from", "pyphi[mcp]", "pyphi-mcp"]
    }
  }
}
```

Claude Code inherits your shell's `PATH`, so as long as `uv` is on it this needs
no absolute paths.

### Claude Desktop

Add it to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "pyphi": {
      "command": "uvx",
      "args": ["--from", "pyphi[mcp]", "pyphi-mcp"]
    }
  }
}
```

Claude Desktop is a GUI application and does not inherit your shell's `PATH`; it
launches subprocesses with a minimal system `PATH`. If `uvx` (or `pyphi-mcp`) is
not found there, give the absolute path to the executable — the one that
`which uvx` prints in your shell:

```json
{
  "mcpServers": {
    "pyphi": {
      "command": "/Users/you/.local/bin/uvx",
      "args": ["--from", "pyphi[mcp]", "pyphi-mcp"]
    }
  }
}
```

## A first conversation

Once connected, you talk to the assistant, not to the tools. Here is a typical
session and the tool calls behind it.

**You:** *What example networks does PyPhi have, and what is the "basic" one?*

The assistant calls `list_examples` and `load_example("basic")`. It gets back a
handle for the substrate plus its shape: three nodes A, B, C — a logic-gate
network where A is an OR, B a COPY, and C an XOR of the others.

**You:** *Compute Φ for it in the state where A and B are on and C is off.*

The assistant calls `analyze` with the state `(1, 1, 0)` and reports:

- **φₛ = 0.0** — the system integrated information. PyPhi's default formalism
  requires a system to do more than specify an irreducible cause–effect state:
  it must also provide itself with a repertoire of alternatives (see
  {doc}`../theory/intrinsic-information`). "Basic" is fully
  deterministic — nothing else could have happened — so it provides itself
  none, and φₛ falls to 0.
- **Φ = 1.857** — the structure integrated information, summed over the
  network's **3 distinctions** and **2 relations**.

φₛ = 0 does not mean the network has no structure — it still specifies a rich
Φ-structure. Because φₛ and Φ are different quantities — one is about
existing as a whole, the other about how much structure is specified — a good
assistant keeps them distinct rather than reporting "the phi value." The
bundled reference is what teaches it to do that (see below).

**You:** *Which mechanism contributes the most, and why?*

The assistant calls `inspect` on the result to read a specific distinction in
full — its mechanism, its cause and effect purviews, and its φ value — without
dragging the entire (potentially very large) structure back into the
conversation.

**You:** *Show me the cause-effect structure.*

The assistant calls `plot`. The Φ-structure is an interactive 3-D figure, so the
tool returns a path to a self-contained HTML file to open in a browser — it is
not shown inline, because a static snapshot of something meant to be rotated and
hovered would be misleading. The connectivity graph, transition probability
matrix, and repertoire plots are static, so those *are* returned as an inline
image.

## What it exposes

**Tools**

- `list_examples` / `load_example` — the standard networks from the IIT
  literature.
- `build_substrate` — build a substrate from a transition probability matrix.
- `describe_substrate` — inspect a substrate's nodes, connectivity, and state
  convention.
- `analyze` — compute system integrated information φₛ and the Φ-structure of a
  substrate in a state (under the default IIT 4.0 formalism, or an earlier
  version if you ask for one).
- `inspect` — drill into one part of a result (a distinction, a repertoire).
- `plot` — render PyPhi's built-in visualizations (needs the `visualize`
  extra): the cause-effect structure (`kind="ces"`), the cause and effect
  repertoires (`"repertoires"`), the causal connectivity graph
  (`"connectivity"`), or the transition probability matrix (`"tpm"`).
- `get_iit_reference` — read the bundled, citation-checked reference on the
  theory, the equations, and the common pitfalls.

**Resources** — the same reference documents, at `pyphi://theory/{topic}`.

**Prompts** — `explain_result` (narrate an analysis in plain language) and
`build_system_walkthrough` (turn a description of some units into a valid
transition probability matrix).

## Good to know

- **The assistant is taught the theory.** A citation-checked reference (the
  postulates, the key equations, and the subtleties that most often mislead)
  ships with the server as its startup instructions, as `pyphi://theory/*`
  resources, and through `get_iit_reference`. This is what keeps it from, for
  example, reading Φ = 0 as "no structure" (it means *reducible*) or confusing
  φₛ with Φ. If you are new to IIT, asking the assistant to read
  `get_iit_reference("theory")` and `get_iit_reference("gotchas")` first is
  worthwhile.
- **Big analyses are guarded.** Φ is combinatorially expensive (the practical
  ceiling for an exact analysis is about 10–12 units), so `analyze` refuses a
  full analysis of a large substrate unless you confirm it. A full Φ-structure
  can serialize to megabytes, so `analyze` returns a compact summary by default
  and `inspect` reads a specific part in full on request.

## Connect the MCP server to an arbitrary client

The server speaks MCP over stdio and is started by the `pyphi-mcp` console
script (equivalently, `python -m pyphi.mcp`). Clients launch it as a subprocess;
you rarely run it by hand.

A client launches that subprocess with its own environment rather than your
shell's, so a bare `pyphi-mcp` is found only if it is on the client's `PATH` —
which a project virtual environment usually is not. Two ways to make the launch
reliable without hardcoding a path:

- **Run it through `uv`** (recommended). `uvx` resolves an isolated environment
  and runs the entry point, so nothing needs to be installed or activated first.
  From PyPI the launch command is `uvx --from "pyphi[mcp]" pyphi-mcp`; from
  GitHub it is
  `uvx --from "pyphi[mcp] @ git+https://github.com/wmayner/pyphi.git" pyphi-mcp`.
- **Install it as a tool** with `uv tool install "pyphi[mcp]"` (or
  `pipx install "pyphi[mcp]"`), which places `pyphi-mcp` on a stable `PATH`
  entry so the bare name works everywhere.
