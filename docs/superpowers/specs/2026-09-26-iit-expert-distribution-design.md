# Distributing IIT Expert alongside PyPhi 2.0

Date: 2026-09-26. Status: draft, awaiting review.

## Goal

When PyPhi 2.0 ships, people using an AI assistant with PyPhi should be pointed
to IIT Expert, the lab's hosted reference on the theory, and should be able to
install it in one step. IIT Expert has two parts: a connector (the remote MCP
server at `https://mcp.learniit.org`) and the `iit-expert` skill. The skill is
the part that makes an assistant consult the connector before answering, so
both must be prominent, and installing one should bring the other.

IIT Expert is a work in progress. It ships now as a soft release, labelled as a
work in progress in every place it is advertised.

## What exists today

- **PyPhi MCP server.** Local, computes IIT quantities. Documented in
  `docs/howto/mcp-server.md`, linked from one "AI agents" card on the landing
  page and one paragraph in the README. `pyphi-mcp install` also writes two
  skills into every detected agent: `pyphi` (the software) and `iit` (a short
  instruction to read a source before answering about the theory, whose first
  choice is already `mcp.learniit.org`).
- **IIT Expert.** Source in `~/projects/iit-reference`: the skill at
  `surfaces/skill/SKILL.md`, the site and connector deployed by one Cloudflare
  Worker to `learniit.org` and `mcp.learniit.org`. The build also publishes the
  skill as `learniit.org/iit-expert-skill.zip`, and `learniit.org/install`
  explains how to add the connector and the skill.
- **The plugin repository** `~/projects/iit-expert-plugin` is scaffolded but
  has no remote, no marketplace file, a skill copy that has fallen behind the
  source, and references to an abandoned domain (`reference.iit.wiki`).

Two problems follow. An assistant with both PyPhi's `iit` skill and
`iit-expert` installed has two skills that trigger on the same questions. And
the install page tells Claude Code users to install a plugin that has not been
published.

## Decisions

1. **PyPhi retires its `iit` skill.** `iit-expert` is the one skill about the
   theory; `pyphi` remains the one skill about the software.
2. **One plugin repository serves Claude Code, Codex and Cursor.** The
   plugin bundles the skill and the connector, so installing it gives an agent
   both, and each agent's own plugin system handles updates.
3. **`pyphi-mcp install` offers the plugin by running each agent's own plugin
   commands**, rather than downloading and writing the skill itself. PyPhi
   never writes an `iit-expert` skill file, so it cannot create a duplicate of
   one the plugin installed.
4. **The documentation is organized by what the reader wants to do**:
   understanding the theory (IIT Expert) and computing with it (the PyPhi MCP
   server), presented as complementary tools meant to be used together.

## Part 1: the plugin repository

Public repository `wmayner/iit-expert-plugin`, created from the existing
scaffold.

### Layout

```
plugin.json                        Agent Plugins manifest (Codex, Cursor)
mcp.json                           Agent Plugins MCP file
.claude-plugin/plugin.json         Claude Code manifest
.claude-plugin/marketplace.json    Claude Code marketplace (the repo is its own)
.mcp.json                          Claude Code MCP file
.cursor-plugin/marketplace.json    needed for Cursor's "From GitHub Repository"
.agents/plugins/marketplace.json   Codex marketplace
skills/iit-expert/SKILL.md         the one shared skill
scripts/check_manifests.py         consistency check, run in CI
README.md, LICENSE
```

Agent Plugins (agent-plugins.org) is the manifest format Codex and Cursor both
read. Claude Code does not read it, so it gets its own `.claude-plugin/`
manifests alongside, sharing `skills/`.

The two MCP files exist because the formats reportedly differ: Agent Plugins
writes the transport as `"type": "streamable-http"` and Claude Code as
`"type": "http"`. If verification shows one file satisfies every agent, the
repository keeps one.

### Manifest contents

Every manifest carries the same `name` (`iit-expert`), `description`,
`version`, `homepage` (`https://learniit.org`), `repository`, `license`
(CC-BY-4.0) and author (the Center for Sleep and Consciousness, as today). The
description states that the plugin is a work in progress. `version` starts at
`0.1.0` and is raised on every release, because Claude Code pins installed
plugins to the manifest version and only offers an update when it changes.

`scripts/check_manifests.py` fails if the shared fields disagree across
manifests or if either MCP file names a different URL. It runs in a GitHub
Actions workflow on every push.

### Install commands the documentation will give

| Agent | Commands |
|---|---|
| Claude Code | `claude plugin marketplace add wmayner/iit-expert-plugin`, then `claude plugin install iit-expert@iit-expert-plugin` (or the `/plugin` equivalents in a session) |
| Codex | `codex plugin marketplace add wmayner/iit-expert-plugin`, then `codex plugin add iit-expert@iit-expert-plugin` |
| Cursor | Customize → From GitHub Repository → `wmayner/iit-expert-plugin` |

The marketplace names in these commands are whatever each marketplace file
declares; verification (below) confirms the exact strings before any document
prints them.

Listing in the official directories (Anthropic's, OpenAI's, Cursor's) is out of
scope for this release. Each involves review and can follow once the content is
vetted.

## Part 2: IIT Expert (`iit-reference`)

- **Sync script.** `scripts/sync_plugin.py` (or an npm script, matching the
  repo's build tooling) copies `surfaces/skill/` into a plugin checkout whose
  path is given as an argument, and bumps the plugin version. The skill's
  source of truth stays in `iit-reference`; the plugin and the zip are both
  generated from it.
- **Install page.** `learniit.org/install` leads with the plugin commands for
  Claude Code, Codex and Cursor, followed by the existing connector-plus-zip
  route for claude.ai and Claude Desktop, which cannot install plugins from a
  third-party marketplace. It adds Cursor's one-click MCP install link for
  people who want the connector alone. It names PyPhi's MCP server for
  computation and links to PyPhi's documentation.
- **Work-in-progress label** on the install page and the site's landing page.
- **Plugin README** is rewritten for `learniit.org` and describes the soft
  release.

## Part 3: `pyphi-mcp install`

All changes are in `pyphi/mcp/agents.py`, `pyphi/mcp/install.py` and
`test/mcp/`.

### Retiring the `iit` skill

- Delete `pyphi/mcp/skills/iit/`.
- Add `RETIRED = frozenset({"iit"})`. `deliver()` and `remove()` also delete a
  retired skill's directory when it holds the sentinel file, so upgrading PyPhi
  and running `install` again removes the old copy, and `uninstall` removes it
  too. A hand-written skill named `iit` is left alone, as today.
- The `pyphi` skill's pointer ("For what the theory says, use the `iit`
  skill") names `iit-expert` and says where to get it.

### Offering the plugin

A new step runs after the skills step, with the same consent rules:

```
Install the IIT Expert plugin (skill + connector) for Claude Code, Codex? [Y/n]
```

- `--iit-expert` / `--no-iit-expert` answer it without a terminal. With neither
  flag and no terminal, nothing runs and the report prints the commands.
- For each detected agent with a command-line plugin installer (`claude`,
  `codex`, found with `shutil.which`), it runs that agent's two plugin commands
  with a timeout. The commands are listed once, in a table beside `AGENTS`.
- An agent whose executable is missing, or whose command fails or times out,
  gets its commands printed instead, and the step moves on. A failure here
  never fails `install`.
- Cursor has no command-line installer, so the report prints the Cursor steps
  and the `learniit.org/install` address.
- `--print` shows the commands that would run.
- `uninstall` does not remove the plugin, since PyPhi does not own it. It
  prints the commands that would.

### Skill paths (verification first)

Two existing questions are settled by testing against the installed agents
before the code changes:

- The Codex documentation now lists `~/.agents/skills` and no longer mentions
  `~/.codex/skills`, where PyPhi writes today. If Codex 0.156 no longer reads
  `~/.codex/skills`, the Codex target moves to `~/.agents/skills`.
- Cursor reads `~/.cursor/skills`, `~/.claude/skills`, `~/.agents/skills` and
  `~/.codex/skills`. A user with both Claude Code and Cursor therefore probably
  gets the `pyphi` skill twice in Cursor. If `cursor-agent` shows the skill
  loading twice, the Cursor target is dropped whenever another target Cursor
  reads is also being written.

### The PyPhi server's instructions

The primer the server sends at startup (`pyphi/mcp/content/`) gains one
sentence: for what the theory says, prefer the IIT Expert connector where it
is connected. It does not repeat install instructions.

### Tests

- The `iit` assertions in `test/mcp/test_agents.py` are replaced: shipped
  skills are exactly `["pyphi"]`; a sentinel-marked `iit` directory is removed
  by `deliver()` and by `remove()`; an unmarked one survives.
- The plugin step is tested with `subprocess.run` and `shutil.which`
  monkeypatched: commands run for a detected agent with an executable;
  commands are printed for a missing executable, a non-zero exit and a timeout;
  `--no-iit-expert` and a non-interactive run execute nothing.
- Every test that goes through `run()` keeps the existing home-directory
  isolation fixture, and additionally must never reach a real `claude` or
  `codex` executable.

## Part 4: PyPhi documentation

The public prose is written with the `writing-naturally` skill.

- **New page `docs/howto/ai-assistants.md`**, the hub. A short section on which
  tool does what and that they are meant to be used together; then IIT Expert
  setup (the plugin commands per agent, the claude.ai and Desktop route, the
  work-in-progress label, and a link to `learniit.org/install` as the
  maintained instructions); then a pointer to `mcp-server.md` for the PyPhi
  server. Setup details that belong to IIT Expert stay brief here, so this page
  does not drift as IIT Expert changes.
- **Landing page (`docs/index.md`)**: the one "AI agents" card becomes two
  cards, one for learning the theory with IIT Expert and one for computing with
  the PyPhi MCP server. Both link to the hub page.
- **`docs/howto/mcp-server.md`**: a note near the top pointing to IIT Expert for
  questions about the theory; the "Skills for your coding agent" section
  describes the one remaining skill and the plugin offer.
- **`docs/howto/index.md`**, **`README.md`**, **`docs/whats-new-in-2.0.md`**:
  name both tools where the MCP server is mentioned today.
- **Changelog fragments**: one `change` fragment (the `iit` skill is retired and
  `install` offers the IIT Expert plugin), one `doc` fragment (the hub page).
- **ROADMAP.md**: a row for this work.

## Verification

1. `claude plugin validate` passes on the plugin repository.
2. The plugin is installed from a local checkout into Claude Code, Codex and
   `cursor-agent`, and in each the `iit-expert` skill is listed and the
   connector's tools respond. Then again from the public GitHub repository
   once it is pushed.
3. The two skill-path questions in Part 3 are answered by the same installs,
   and the answers recorded in the commit that acts on them.
4. `pyphi-mcp install` run in a scratch uv project with a scratch home
   directory: the plugin step runs the real commands for Claude Code and Codex,
   and a second run reports the plugin as already installed rather than failing.
5. `uv run pytest` with no path argument (so doctests run) is green, read from
   the summary line.
6. `just docs` builds with warnings as errors, and the new page and cards are
   checked in a browser.

## Order of work

1. Plugin repository: manifests, skill sync, check script, local verification
   in all three agents.
2. Create `wmayner/iit-expert-plugin` on GitHub and push (the user pushes, or
   approves the push).
3. `iit-reference`: sync script, install page, labels; deploy.
4. PyPhi: skill-path verification, then `install` changes with tests, then
   docs.
5. Final verification against the public repository.

## Out of scope

- Official directory listings for any agent.
- Packaging PyPhi's own skill and MCP server as a plugin. The server runs from
  a specific Python environment, which `pyphi-mcp install` already handles.
- Detecting an already-installed plugin by reading an agent's internal files.
