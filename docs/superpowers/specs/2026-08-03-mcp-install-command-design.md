# `pyphi-mcp install` — design

**Status:** draft, awaiting approval.

## The problem

An assistant working on a PyPhi project reaches for `Bash` before it reaches
for the MCP tools. `Bash` is always available, it already knows how to drive
Python, and running a script feels like the direct route. The MCP server's
`instructions` are advisory, arrive as a low-priority preamble, and — as an
external tester's session showed — do not bind: the server was connected and
the assistant still wrote and ran its own script, then reported φₛ as Φ from
its output.

The second half of that failure is worse than the first. Having decided to work
in a plain directory, the assistant wrote its own `CLAUDE.md` summarising what
it thought it had learned. The summary contained a false claim about the theory.
Because `CLAUDE.md` is loaded into context at the start of every session, the
false claim then re-entered every subsequent conversation, and the assistant had
no reason to doubt something it read in the project's own notes.

A project's instruction file is the one channel that is present *before* the
model acts. Nothing we ship reaches it today: `claude mcp add pyphi -- …`
writes `.mcp.json` and stops there.

## What the command does

```
pyphi-mcp install [--scope project|user] [--client claude-code|claude-desktop]
                  [--from <spec>] [--print] [--force]
pyphi-mcp uninstall [--scope project|user] [--client claude-code|claude-desktop]
```

`--scope` defaults to `project`: an install that touches only the directory you
ran it in is the one you can undo by deleting a file you can see.

Two files, both idempotent, neither overwritten silently.

**1. The MCP registration.** Writes the `pyphi` entry into the client's config
(`.mcp.json` for a project-scoped Claude Code install, `~/.claude.json` for a
user-scoped one, `claude_desktop_config.json` for Claude Desktop), preserving
any other servers already registered. `--from` overrides the package spec so
the main-branch and local-checkout cases stay one flag rather than a different
recipe. Re-running with an existing `pyphi` entry that differs prints the diff
and requires `--force`.

**2. The agent instructions.** Appends a delimited block to `AGENTS.md` in the
target directory, creating the file if absent, and makes sure Claude Code can
see it (below):

```markdown
<!-- pyphi:begin — managed by `pyphi-mcp install`; edits inside are overwritten -->
## PyPhi

φₛ and Φ are different quantities under IIT 4.0. `analyze(...).phi` is φₛ,
system integrated information — whether the system exists as one whole.
`.big_phi` is Φ, structure integrated information — the sum of φ over the
Φ-structure's distinctions and relations.

States are little-endian: the first node is the least-significant bit.

Analyses are superexponential in substrate size. `pyphi.cost.estimate_analysis`
is free; call it before any run over more than a handful of units.

These three are the ones that go wrong unaided; the rest of the reference is
`get_iit_reference("theory")` and `("equations")` where the MCP server is
connected, or
`python -c "from pyphi.mcp import content; print(content.load('gotchas'))"`
otherwise.

Where the server is connected, use its tools for exploration and for
interpreting results: they report which formalism produced each number, refuse
runs too large to finish, and keep φₛ and Φ distinct. The server holds results
only in memory, so anything that has to be reproducible belongs in a script —
where these same facts still apply.
<!-- pyphi:end -->
```

The delimiters make the block re-writable: a later `install` replaces what is
between them and leaves the user's own content alone. `uninstall` removes the
block and the MCP registration, leaving both files otherwise untouched and
deleting either only if the install created it and nothing else was added.
`--print` writes nothing and emits both artifacts to stdout, for people who
would rather paste.

**Reaching Claude Code.** Claude Code reads `CLAUDE.md` and does not read
`AGENTS.md`.[^1] So the block goes in `AGENTS.md`, which Codex, Cursor and
others read, and `install` then ensures `CLAUDE.md` contains the line

```markdown
@AGENTS.md
```

which is Claude Code's documented import syntax: it loads the imported file at
session start. If `CLAUDE.md` already exists and already imports `AGENTS.md`,
nothing is written. If it exists without the import, the line is added at the
top and the rest is left alone. If neither file exists, both are created, with
`CLAUDE.md` holding only the import.

A symlink (`ln -s AGENTS.md CLAUDE.md`) is the other documented option and is
what this repository itself uses, but it needs Administrator privileges or
Developer Mode on Windows, so the import is the portable default.

[^1]: <https://code.claude.com/docs/en/memory> — "Claude Code reads
`CLAUDE.md`, not `AGENTS.md`. If your repository already uses `AGENTS.md` for
other coding agents, create a `CLAUDE.md` that imports it."

## Choices worth stating

**Why a project instruction file rather than more server instructions.** We
already have server instructions and they did not hold. The difference is
ordering and authority: the project's own instruction file is context the model
treats as established fact, and it is in the window before the first tool call.
It is also client-independent, where MCP `instructions` are delivered only if
the client chooses to surface them.

**Why the block is short.** It carries the three things that actually go wrong
— the two quantities, the endianness, the cost — and a pointer to everything
else. A long block gets skimmed, and duplicating the reference here would give
us a second copy to keep correct.

**Why the facts come before the tool guidance.** The failure this exists to
prevent was a wrong claim about the theory, not a wrong choice of tool. An
assistant that reports φₛ as Φ does equal damage whether it got the number from
`analyze` or from its own script, so the facts govern both paths and belong
first. The tool guidance is the last paragraph because it is the narrower point.

**Why the facts are stated rather than linked.** An instruction to go read a
document before starting is skipped exactly when the task feels too small to
need theory — which is when these mistakes get made. The server's primer has
said "read the reference before interpreting results" since it shipped, and the
session that prompted this work had the server connected and did not. Content
that must survive that has to be present already, not one tool call away. The
same reasoning promoted `gotchas.md` into the server's always-loaded
instructions; this block is the client-independent version of it.

**Why it stops short of preferring tools over scripts in general.** The server
holds results only in memory: handles like `res1` are entries in a dictionary
that dies with the process, and no tool writes a result to disk. A researcher
building a reproducible analysis is right to write a script, and an instruction
that pushed them off that path would cost more than the mislabelling it was
meant to prevent. The division is exploration and interpretation through the
tools, durable work in scripts, with the same facts governing both.

**Why not a hook.** A `PreToolUse` hook matching pyphi invocations would be
real enforcement rather than encouragement. It is also Claude-Code-specific,
blocks tool calls, and needs `settings.json` surgery. Worth revisiting only if
the softer measures measurably fail.

## Testing

- Registration written into an empty directory, and merged into a config that
  already holds other servers, without disturbing them.
- Block appended to an absent, an empty, and an already-populated `AGENTS.md`.
- Re-running replaces the delimited block and preserves surrounding content
  verbatim.
- A conflicting existing `pyphi` entry is refused without `--force`.
- `--print` writes nothing to disk.
- `AGENTS.md` created where absent; the `@AGENTS.md` import added to an
  existing `CLAUDE.md` that lacks it, and not duplicated where it is present.
- `uninstall` removes the block and the registration and leaves surrounding
  content byte-identical; it deletes a file only where `install` created it and
  nothing else was added.

## Resolved during review

- **Which instruction file.** `AGENTS.md` for interoperability, with an
  `@AGENTS.md` import written into `CLAUDE.md`, because Claude Code does not
  read `AGENTS.md` natively.
- **Default scope.** `project`.
- **`uninstall`.** Ships alongside `install`.
