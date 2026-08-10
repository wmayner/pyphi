# Installing agent skills from `pyphi-mcp install` — design

**Status:** draft, awaiting approval.

## The problem

`pyphi-mcp install` registers the MCP server and writes a short block of facts
into the project's `AGENTS.md`. Both are project-scoped, and both help only
after someone has decided to work on PyPhi in that directory.

Neither reaches the case this design exists for: an assistant asked about
Integrated Information Theory that answers from recollection. Most published
summaries of IIT misstate it, often invert it, and a model's recollection is
built out of those summaries. The assistant has no reason to suspect it is
wrong, so it does not go looking, and the MCP server's `get_iit_reference` tool
never fires — the tool is available, but nothing prompts the model to call it.
The same is true of `learniit.org`, which will answer these questions with a
locator attached to every claim, but only if the model thinks to ask.

An agent skill is the one mechanism that fires without being called. The
client matches a skill's description against the task before the model acts, so
a skill can interrupt at the point where recollection would otherwise be used.
That is what we want to install.

There is a second, smaller problem. PyPhi 2.0 renamed most of the pre-2.0
surface with no aliases, and two result types swapped names, so code written
from memory of PyPhi 1.x either raises `ImportError` or, in the swapped case,
runs and means something else. This matters less once 2.0 has been published
for a while, but the API needs describing regardless.

## What the command does

`install` gains a third step, after the MCP registration and the `AGENTS.md`
block:

```
pyphi-mcp install [--skills | --no-skills] [--agent NAME] [--agent-path DIR]
```

1. Detect installed agents by probing for a directory under `$HOME`.
2. If none are found, print nothing further and finish.
3. `--no-skills` skips the step. `--skills` runs it without asking.
4. With neither flag: if standard input and output are both terminals and `CI`
   is unset, ask —

   ```
   Install the PyPhi skills for Claude Code, Codex? [Y/n]
   ```

   Otherwise skip, and print `pyphi-mcp install --skills` as the way to do it
   later.
5. Copy each skill directory into each target. A failure on one agent is a
   warning, not an error; the remaining agents are still written.

The step runs last so that interrupting the prompt still leaves a working MCP
registration behind. `--print` gains the skills and the paths they would be
written to, and continues to write nothing.

`uninstall` probes the same three agents, honours the same `--agent` and
`--agent-path` flags, and removes the skills it finds. It asks nothing.

### Detection

| Agent | Probe | Target |
| --- | --- | --- |
| Claude Code | `~/.claude` | `~/.claude/skills/` |
| Codex | `~/.codex` | `~/.codex/skills/` |
| Cursor | `~/.cursor` | `~/.cursor/skills/` |

`--agent NAME` installs for one of those three whether or not it was detected.
`--agent-path DIR` writes to a skills directory not in the table, for agents we
do not know about and for the case where one of these three moves. Both are
repeatable.

Skills are always written under `$HOME`, whatever `--scope` says: they teach
IIT and PyPhi, not this project. `--scope` continues to govern the MCP
registration alone, so the report prints the full path of everything written
outside the project directory.

### Delivery and removal

The skills ship inside the wheel, under `pyphi/mcp/skills/`, and `install`
copies them. The copy is a copy rather than a symlink because the source is
inside `site-packages`, which a deleted or rebuilt virtual environment takes
with it, while the skills in `$HOME` would outlive it. The cost is that an
upgraded PyPhi leaves stale skills until `install` is run again; `install`
always overwrites its own directories, so running it again is the fix.

Each installed directory gets a `.pyphi-skill` file holding the PyPhi version
that wrote it. `uninstall` removes only directories containing that file, so a
hand-written skill that happens to share a name is left alone. The version also
lets a re-install report what it refreshed, and makes a stale skill diagnosable
by reading one file.

## The skills

Two, with different triggers.

### `iit` — read before answering

Fires on any task that involves the theory: explaining it, evaluating a claim
about it, deciding whether some system is conscious on its account, or
interpreting a number PyPhi produced.

```yaml
name: iit
description: >-
  Use before explaining, discussing, evaluating, or implementing anything
  involving Integrated Information Theory — its axioms, postulates, φ, Φ,
  distinctions, relations, complexes, or whether some system is conscious on
  IIT's account — and before interpreting any number PyPhi produced. Read the
  sources named here before answering. Most published summaries of IIT
  misstate it, so answering from recollection reproduces those errors.
```

The body carries five things:

1. **Read first.** Do not answer from recollection, including for questions
   that seem too small to need a source.
2. **Where the theory is, in precedence order.** `mcp.learniit.org` where the
   connector has been added, because every claim there carries a locator;
   `get_iit_reference("theory")`, `("equations")` and `("gotchas")` where the
   PyPhi MCP server is connected; `python -c "from pyphi.mcp import content;
   print(content.load('theory'))"` otherwise, which needs only PyPhi installed;
   and the papers themselves in a PyPhi checkout. The entry for
   `mcp.learniit.org` tells the reader how to add the connector, so a model
   that cannot reach it can say what the user should do.
3. **Which formulation is current.** IIT 4.0 (Albantakis et al. 2023) for the
   mathematics, Tononi & Boly 2025 for the prose. IIT 3.0 is superseded — cite
   it for history or to answer what changed.
4. **Expound, do not survey.** State what IIT holds. Answer a standard
   objection the way IIT answers it, which the sources contain. Never invent:
   where the sources do not settle something, say that IIT has not addressed
   it, or that this is an extrapolation.
5. **The errors that recur.** φ and Φ and φₛ are three quantities. Integrated
   information is not Shannon information about the system. A system's
   behaviour does not settle its Φ. IIT is not a functionalist or
   computational account.

Where a number would settle the question, the skill points at `pyphi`.

### `pyphi` — the library and the server

Fires on any task that touches PyPhi code.

```yaml
name: pyphi
description: >-
  Use when writing, reading, or running any code that imports pyphi —
  building a Substrate or a TPM, calling analyze(), choosing a formalism,
  estimating cost before an expensive run, or saving results reproducibly.
  Covers the PyPhi 2.0 API and the MCP server's tools. PyPhi 2.0 renamed most
  of the pre-2.0 surface with no aliases, so code written from memory of older
  versions will not run.
```

The body carries:

1. **Use the server for exploration.** Its tools report which formalism
   produced each number, refuse runs too large to finish, and keep φₛ and Φ
   apart. Where it is not connected, the skill gives the command to add it.
   Durable work still belongs in a script, because the server holds results
   only in memory.
2. **The 2.0 API.** `pyphi.analyze` as the entry point, `Substrate` and
   `System`, the two result types whose names were exchanged, formalism
   presets, and `pyphi.numerics.eq` in place of `==` on φ values.
3. **A φ value means nothing without its formalism.** Pin the formalism and
   name it whenever the number is reported.
4. **States are little-endian.** The first node is the least significant bit.
5. **Estimate before running.** `pyphi.cost.estimate_analysis` is free;
   analyses are superexponential in substrate size.
6. **Building a substrate**, ending in a check that a known state round-trips,
   because an axis-order error produces a well-formed TPM that is wrong.
7. **Reproducible work.** Seeded generator instances rather than global
   seeding, `pyphi.provenance.save_json` / `save_npz` / `save_dataframe`, keys
   in the filename, no overwriting, and per-trial values saved alongside any
   summary computed from them.

### Where the depth comes from

`pyphi/mcp/content.py` describes its Markdown files as the single source of the
server's teaching material, surfaced three ways: the server's instructions, the
`pyphi://theory/*` resources, and `get_iit_reference`. The skill is a fourth
surface over the same files, not a fourth copy. `install` fills the `pyphi`
skill's `references/` from `pyphi/mcp/content/`, and `SKILL.md` indexes them,
so the client loads a document only when the task calls for it.

One topic is missing and has to be written: reproducible work, covering
seeding, the provenance writers, the filename and no-clobber conventions, and
saving raw values alongside aggregates. It becomes
`pyphi/mcp/content/reproducible-work.md` and a `get_iit_reference` topic, which
means the server gains it too rather than the material existing only inside a
skill.

The `iit` skill has no `references/`. Its job is to send the reader to
`learniit.org` or to `get_iit_reference`, and shipping a static copy of the
theory alongside those would give us a third thing to keep true.

## Testing

- Detection finds only the agents whose probe directory exists, against a
  temporary `$HOME`.
- `--skills` installs without a prompt; `--no-skills` writes nothing.
- Without either flag and without a terminal, or with `CI` set, the step is
  skipped and the follow-up command is printed.
- Re-installing overwrites the skill directories and updates the version stamp.
- `uninstall` removes only directories holding `.pyphi-skill`; a hand-written
  skill sharing a name survives.
- `--agent` reaches an agent that was not detected; `--agent-path` reaches a
  directory outside the table.
- A failure on one agent leaves the others installed and exits zero with a
  warning.
- `--print` lists the skills and their destinations and writes nothing.
- After install, the `pyphi` skill's `references/` matches
  `pyphi/mcp/content/`.
- A wheel built from the tree contains `pyphi/mcp/skills/**`, which guards the
  Hatchling include configuration.
- Every shipped `SKILL.md` has parseable front matter whose `name` matches its
  directory.

## Choices worth stating

**Why the skills ship in the wheel.** Wrangler keeps its skills in a separate
repository and fetches them over the GitHub contents API, with a day-long cache
and a retry hint for when the network fails. That machinery exists because the
skills and the tool release independently. Ours do not have to: putting them in
the wheel removes the network, the cache and the lockfile, and guarantees that
the skills describe the version of PyPhi that installed them.

**Why there is no record of the answer.** Wrangler asks once and stores the
answer, because its prompt appears after around twenty unrelated commands and
would otherwise nag. Our prompt appears only when someone types
`pyphi-mcp install`. Asking again on the next install is how a user who
declined changes their mind, so the file would cost a `--force` flag and buy
nothing.

**Why the gate ships from PyPhi rather than from `iit-reference`.**
`iit-reference` is the authority on the theory and already has a skill,
`surfaces/skill/SKILL.md`. What it does not have is a way to deliver it:
publishing there means pasting an MCP URL into a client's settings, and nothing
in that flow installs a file. `pyphi-mcp install` is the only command either
project ships that a user runs. Vendoring `iit-reference`'s skill would give
that skill two homes; pointing at `learniit.org` from a short gate of our own
keeps one authority for the theory and puts the interrupt where it can be
installed. When `learniit.org` deploys, the gate improves without changing.

**Why two skills rather than four.** Substrate construction and reproducible
work both fire on tasks that import PyPhi, so separate descriptions would not
match more precisely than one. Progressive disclosure through `references/` is
the mechanism skills already have for keeping a large body of material out of
context until it is needed.

**Why `install` writes outside the project.** A skill installed at project
scope is loaded only in that project, and the failure this addresses happens
when someone asks about IIT anywhere. The cost is that `--scope project` no
longer describes everything the command writes, which the report has to
correct by printing full paths.
