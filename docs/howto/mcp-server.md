# The PyPhi MCP server

PyPhi includes a [Model Context Protocol](https://modelcontextprotocol.io) (MCP)
server that lets an AI assistant build substrates, run IIT analyses, inspect the
results, visualize them, and read a grounded reference on the theory — all by
talking to it in plain language. It runs locally in your own environment.

It is meant to be useful in two ways: as a calculator and interpreter for
researchers using AI assistance, and as a tutor for newcomers exploring the
theory.

```{tip}
For questions about the theory itself, pair the server with IIT Expert, which
answers from IIT's primary literature. {doc}`ai-assistants` covers both.
```

## Set up a project with Claude Code

Four commands take you from nothing to an assistant that can compute Φ. Each
one is explained below.

```bash
uv init my-iit-project
cd my-iit-project
uv add "pyphi[mcp]"
uv run pyphi-mcp install
```

Then open the project with `claude`. Claude Code asks once whether to trust the
server this project registers; approve it, and the PyPhi tools are available for
the rest of that session and every later one. `claude mcp list` reports its
status if you want to check before starting.

### What each step does

`uv init` creates the project and `uv add` installs PyPhi into its environment.
The `mcp` extra brings in the server and its `pyphi-mcp` command. Add
`visualize` as well — `"pyphi[mcp,visualize] @ …"` — if you want the `plot` tool,
which needs the visualization stack.



`uv run` finds `pyphi-mcp` because PyPhi is now a dependency of the project you
are standing in, so the command needs no activated environment and no absolute
path. `uv run --with` and `uvx` build a throwaway environment for a single
command, so `install` refuses to write a registration from one.

Outside a uv project, run `pyphi-mcp install` with the environment holding PyPhi
activated, or call it by its full path, `/path/to/venv/bin/pyphi-mcp install`.

### What `install` writes

It registers the server in `.mcp.json` and writes a short block of PyPhi facts
into `AGENTS.md` — the two quantities φₛ and Φ, the little-endian state order,
and the cost of an analysis. A project's instruction file is read before the server connects, so the
block also reaches an assistant that drives PyPhi from a shell.

Claude Code reads `CLAUDE.md` rather than `AGENTS.md`, so `install` also adds
an `@AGENTS.md` import line to `CLAUDE.md`. Both files are safe to already
have: the block is written between markers and a later `install` replaces only
what is between them, and the import is not added twice. `pyphi-mcp uninstall`
removes both, and anything you wrote around them survives.

### Skills for your coding agent

`install` then offers two additions for every AI coding agent it finds. The
first is the `pyphi` skill, which covers the 2.0 API, state ordering, cost
estimation, and the conventions for a reproducible script. The second is the
IIT Expert plugin, which brings the `iit-expert` skill and the connector to
IIT's primary literature (see {doc}`ai-assistants`).

It probes `~/.claude`, `~/.codex` and `~/.cursor`, and asks before each:

```
Install the PyPhi skills for Claude Code, Codex? [Y/n]
Install the IIT Expert plugin (skill + connector) for Claude Code, Codex? [Y/n]
```

`--skills` and `--no-skills` answer the first question without a terminal, and
`--iit-expert` and `--no-iit-expert` answer the second, which is what a script
or a continuous-integration run needs; with no flag and no terminal, nothing is
written or run.

For Claude Code and Codex, the plugin is installed by running the agent's own
plugin commands, so the agent handles its updates from then on. If a command is
missing or fails, `install` prints the commands for you to run and carries on.
Cursor has no such command, so `install` prints the steps to follow in its
settings instead.

Cursor also reads the skills folders of Claude Code and Codex, so `install`
writes the `pyphi` skill to Cursor's own folder only when neither of those
agents is present.

`--agent NAME` reaches an agent that was not
detected, and `--agent-path DIR` reaches one PyPhi does not know about. Both
are repeatable.

Skills are written under your home directory whatever `--scope` says, because
they are about IIT and PyPhi rather than about one project, and the report
prints the full path of each one. `pyphi-mcp uninstall` removes them again; it
deletes only directories PyPhi wrote, so a skill of your own that shares a name
is left alone. It does not remove the IIT Expert plugin, which belongs to the
agent once installed; it prints the command that does.

Running `install` again refreshes the skills, which is how you update them
after upgrading PyPhi.

The registration names the Python interpreter you ran `pyphi-mcp install` with,
so the client starts the server from the same environment you installed PyPhi
into — in a uv project, that project's own `.venv`:

```json
{
  "mcpServers": {
    "pyphi": {
      "command": "/path/to/my-iit-project/.venv/bin/python",
      "args": ["-m", "pyphi.mcp"]
    }
  }
}
```

Because that is an absolute path, the client needs nothing on its `PATH` and
resolves no packages at startup. It is also specific to your machine, so someone
who clones the project runs `uv sync` and then `uv run pyphi-mcp install --force`
to point the entry at their own environment. The same command fixes it if you
move the project or rebuild its environment somewhere else.

Use `--scope user` to register for every project rather than this one,
`--client claude-desktop` for Claude Desktop, and `--print` to see what would be
written without writing it. `--from` registers a `uvx` launch command instead,
which resolves the given package specification each time the server starts
rather than using a fixed environment:

```bash
pyphi-mcp install --from "pyphi[mcp] @ git+https://github.com/wmayner/pyphi.git@main"
```

#### Registering by hand

To register the server without the instruction block, use `claude mcp add`. It
takes a name for the server and then, after `--`, the command that starts it;
both are required, and there is no list of known servers to look a bare name up
in.

```bash
claude mcp add pyphi -- /path/to/my-iit-project/.venv/bin/python -m pyphi.mcp
```

If that command opens a Claude session instead of registering anything, a shell
alias is shadowing the subcommand — `claude` aliased to `claude <some flag>`
makes `mcp` parse as a prompt rather than a command. Run the executable itself,
which `which -a claude` prints.

This writes an entry to `.mcp.json` (or `~/.claude.json`), which you can also
create by hand — it is the same JSON shown above.

## Claude Desktop

`pyphi-mcp install --client claude-desktop` writes the entry for you, run the
same way as above from the environment holding PyPhi. To do it by hand, add this
to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "pyphi": {
      "command": "/path/to/your/venv/bin/python",
      "args": ["-m", "pyphi.mcp"]
    }
  }
}
```

Claude Desktop is a GUI application and does not inherit your shell's `PATH`; it
launches subprocesses with a minimal system `PATH`. An absolute path to the
interpreter, which is what `install` writes, sidesteps that.

## A first conversation

Once connected, you talk to the assistant, not to the tools. Here is a typical
session and the tool calls behind it.

**You:** *What example networks does PyPhi have, and what is the "basic" one?*

The assistant calls `list_examples` and `load_example("basic")`. It gets back a
handle for the substrate plus its shape: three nodes A, B, C — a logic-gate
network where A is an OR, B a COPY, and C an XOR of the others.

**You:** *Compute Φ for it in the state where A and B are on and C is off.*

The assistant calls `analyze` with the state `(1, 1, 0)` and reports:

- **φₛ = 0.0** — the system integrated information. IIT requires a system to
  do more than specify an irreducible cause–effect state: it must also provide
  itself with a repertoire of alternatives (see
  {doc}`../theory/intrinsic-information`). "Basic" is fully
  deterministic — nothing else could have happened — so it provides itself
  none, and φₛ falls to 0.
- **Φ = 1.857** — the structure integrated information, summed over the
  network's **3 distinctions** and **2 relations**.

φₛ = 0 does not mean the network has no structure — it still specifies a rich
Φ-structure. Because φₛ and Φ are different quantities — one is about
existing as a whole, the other about how much structure is specified — the server's reference keeps the two distinct (see below).

**You:** *Which mechanism contributes the most, and why?*

The assistant calls `inspect` on the result to read a specific distinction in
full — its mechanism, its cause and effect purviews, and its φ value — without
dragging the entire (potentially very large) structure back into the
conversation.

**You:** *Show me the cause-effect structure.*

The assistant calls `plot`. The Φ-structure is an interactive 3-D figure, so the
tool returns a path to a self-contained HTML file to open in a browser — it is not shown inline. The connectivity graph, transition probability
matrix, and repertoire plots are static, so those *are* returned as an inline
image.

## What it exposes

**Tools**

- `list_examples` / `load_example` — the standard networks from the IIT
  literature.
- `build_substrate` — build a substrate from a transition probability matrix.
- `describe_substrate` — inspect a substrate's nodes, connectivity, state
  convention, its transition probability matrix as state-by-node rows
  (up to 256 states), and the PyPhi version serving it.
- `analyze` — compute system integrated information φₛ and the Φ-structure of a
  substrate in a state, optionally on multiple cores for that call
  (`parallel`, `workers`). `compute` narrows the work to φₛ alone (`"sia"`),
  the cause-effect structure (`"ces"`), or the distinctions alone
  (`"distinctions"`, the only one that skips the system-partition search).
- `configure_parallel` — read or persistently set the server's parallelization
  configuration; a per-call `analyze` setting takes precedence.
- `inspect` — drill into one part of a result (a distinction, a repertoire).
- `plot` — render PyPhi's built-in visualizations (needs the `visualize`
  extra): the cause-effect structure (`kind="ces"`), the cause and effect
  repertoires (`"repertoires"`), the causal connectivity graph
  (`"connectivity"`), or the transition probability matrix (`"tpm"`).
- `get_iit_reference` — read the bundled, citation-checked reference on the
  theory, the equations, and the common pitfalls; its `documentation` topic
  tells the assistant how to read this site directly (see below).

**Resources** — the same reference documents, at `pyphi://theory/{topic}`.

**Prompts** — `explain_result` (narrate an analysis in plain language),
`build_system_walkthrough` (turn a description of some units into a valid
transition probability matrix), and `migrate_code` (rewrite pre-2.0 PyPhi
code for 2.0).

## Good to know

- **The assistant is taught the theory.** A citation-checked reference (the
  postulates, the key equations, and the subtleties that most often mislead)
  ships with the server as `pyphi://theory/*` resources and through
  `get_iit_reference`. The orientation primer and the gotchas load as the
  server's startup instructions, so the mistakes that produce wrong results —
  reading φₛ = 0 as "no structure" when it means *reducible*, confusing φₛ
  with Φ — are in front of the assistant before its first tool call rather than
  waiting behind one. If you are new to IIT, asking it to read
  `get_iit_reference("theory")` as well is worthwhile.
- **Big analyses are guarded.** Φ is combinatorially expensive (the practical
  ceiling for an exact analysis is about 10–12 units), so `analyze` refuses a
  full analysis of a large substrate unless you confirm it. A full Φ-structure
  can serialize to megabytes, so `analyze` returns a compact summary by default
  and `inspect` reads a specific part in full on request. Parallelism divides
  the constants, not the exponents, so the guard applies regardless of the
  `parallel` setting; the assistant knows when multiple cores help from the
  bundled `parallelization` reference.

## Connect the MCP server to an arbitrary client

The server speaks MCP over stdio and is started by the `pyphi-mcp` console
script (equivalently, `python -m pyphi.mcp`). Clients launch it as a subprocess;
you rarely run it by hand.

A client launches that subprocess with its own environment rather than your
shell's, so a bare `pyphi-mcp` is found only if it is on the client's `PATH` —
which a project virtual environment usually is not. Three ways to make the
launch reliable:

- **Give the absolute path to the interpreter** (recommended), as
  `pyphi-mcp install` does: command `/path/to/your/venv/bin/python`, arguments
  `["-m", "pyphi.mcp"]`. `PATH` stops mattering and no packages are resolved at
  startup.
- **Install it as a tool** with `uv tool install "pyphi[mcp]"` (or
  `pipx install "pyphi[mcp]"`), which places `pyphi-mcp` on a stable `PATH`
  entry so the bare name works everywhere.
- **Run it through `uv`**, as `pyphi-mcp install --from <specification>` writes:
  `uvx --from "<specification>" pyphi-mcp` resolves an isolated environment and
  runs the entry point, so nothing needs to be installed or activated first. Be
  aware that when the resolved package provides no such executable, `uvx` falls
  back to running whatever `pyphi-mcp` is on `PATH`, which can quietly start a
  different installation than the one you named.

## Reading the documentation without the server

An assistant that cannot run the server can still read this site directly.
Three files are published for that purpose:

- `llms.txt` at the site root lists every page with a one-line summary, and
  `llms-full.txt` beside it is the whole narrative site as one markdown file
  (the generated API pages are left out; the inventory below covers them).
- Every page's MyST source is served under `_sources/`, so the markdown behind
  this page is at `_sources/howto/mcp-server.md.txt` relative to the site root.
- `objects.inv` is the intersphinx inventory: it maps every documented API name
  to its page and anchor.

On the stable site these resolve against `https://pyphi.readthedocs.io/en/stable/`.
