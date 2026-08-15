# CLAUDE.md

**AI Assistant Guide for PyPhi Development**

This document provides context and guidelines for AI assistants working on PyPhi, a Python library that implements the mathematical formalism of Integrated Information Theory (IIT).

---

## Project Overview

### Key Papers

1. **IIT 4.0 Theory** (2023):
   ```
   Albantakis L, Barbosa L, Findlay G, Grasso M, ... Tononi G. (2023)
   Integrated information theory (IIT) 4.0: formulating the properties of
   phenomenal existence in physical terms. PLoS Computational Biology 19(10): e1011465.
   https://doi.org/10.1371/journal.pcbi.1011465
   ```

2. **PyPhi Software** (2018):
   ```
   Mayner WGP, Marshall W, Albantakis L, Findlay G, Marchman R, Tononi G. (2018)
   PyPhi: A toolbox for integrated information theory.
   PLOS Computational Biology 14(7): e1006343.
   https://doi.org/10.1371/journal.pcbi.1006343

Additional key theoretical papers are in @papers.

---

## Critical Context

### Mathematical Correctness is Paramount

**This is scientific software implementing a precise mathematical formalism.**

- Small bugs can invalidate research results
- Numerical precision matters deeply (configured via `config.numerics.precision`)
- Changes to core computation logic require extreme care
- When in doubt, consult the IIT papers and existing tests

**Correctness > performance > elegance.**

### Don't defer confirmation experiments

When an audit or investigation produces a "probably no effect, worth
confirming later" claim, run the confirmation experiment as part of
the audit. Locking state (goldens, fixtures, baselines, snapshot
tests) onto an unconfirmed assumption multiplies the revalidation
cost when the assumption turns out wrong — and once the state is
committed, downstream work that builds on it inherits the
assumption silently.

*Motivating case:* an IIT 3.0 tie-resolution audit deferred a
five-minute confirmation experiment based on the structural
assumption "the path has a unique MIP by construction". The
assumption was false; four goldens were locked to buggy values for
six days before the deferred experiment finally ran in a downstream
investigation and exposed the bug.

**Approach changes conservatively**:
- Read existing code thoroughly before modifying
- Prioritize refactoring and testing over new features
- Don't assume the current implementation is optimal
- Look for inconsistencies and opportunities to improve clarity

---

## Architecture & Organization

### Key Computational Concepts

1. **Φ (Big Phi)**: Integrated information of a system
   - Computed by finding the Minimum Information Partition (MIP)
   - Combinatorially expensive: requires evaluating all partitions

2. **φ (Small Phi)**: Mechanism integration
   - How irreducible a mechanism's cause-effect repertoire is

3. **Repertoires**: Probability distributions over states
   - **Cause repertoire**: Past states that could lead to current state
   - **Effect repertoire**: Future states the system could transition to

4. **Partitions/Cuts**: Ways of disconnecting a system
   - Used to test irreducibility
   - Different partition schemes available

5. **Distinctions**: Irreducible mechanisms (IIT 4.0)
   - Concepts with cause-effect power

6. **Relations**: Dependencies between distinctions (IIT 4.0)

---

## Development Guidelines

### Before Making Changes

1. **Read the relevant code first**
   - Use [Read](file:///pyphi) to understand current implementation
   - Check tests for expected behavior
   - Consult IIT papers for theoretical grounding

2. **Understand the mathematics**
   - Don't change computation logic without understanding the theory
   - If unsure, ask the user or consult documentation

3. **Check configuration**
   - Many behaviors are configurable
   - See [pyphi_config.yml](file:///pyphi_config.yml) and [pyphi/conf/](file:///pyphi/conf/)

### Code Quality Standards

1. **Type Hints**
   - Add type hints to new code
   - Gradually add to existing code when touching it
   - Use `Optional`, `Tuple`, `Iterable` appropriately

2. **Documentation — docstring style (enforced)**

   All `pyphi/**` docstrings follow one NumPy-style standard, enforced by
   the docs build. See [pyphi/CLAUDE.md](file:///pyphi/CLAUDE.md) for the full
   rules (loads automatically when working in that directory).

3. **Testing**
   - Write tests for all new functionality
   - Use property-based testing (Hypothesis) for mathematical properties
   - Example networks are in `test/example_networks.py`

4. **Performance**
   - This code is computationally expensive by nature
   - Profile before optimizing
   - Consider caching strategies
   - Parallelization is available via Ray (optional dependency)

5. **Changelog Fragments**
   - When making user-facing changes, create a changelog fragment in `changelog.d/`
   - Fragment filename format: `<name>.<type>.md` where:
     - `<name>` is a GitHub issue number (e.g., `123`) or descriptive name (e.g., `fix-cache-bug`)
     - `<type>` is one of: `feature`, `change`, `config`, `optimization`, `fix`, `doc`, `refactor`, `misc`
   - Example: `echo "Added \`new_function()\`" > changelog.d/new-function.feature.md`
   - Use `uv run towncrier create <name>.<type>.md` for guided creation
   - See `changelog.d/README.md` for full documentation

### How to use Python, pip, etc.

Always use `uv run` for running any python development commands (for example,
`uv run python`). Use `uv pip` when pip is needed.

### Code Style

**Do not run unsafe fixes with Ruff without first getting permission from
the user.**

## Python version

**We will support only Python 3.13+ for this version.** Therefore, when writing code, **do not attempt to maintain backward compatibility with previous Python versions.**

---

## Testing Strategy

### Running tests

`just test` runs the suite and forwards extra arguments to pytest.

### Doctest scope — important

The pytest config in `pyproject.toml` sets ``testpaths = ["pyphi",
"test"]`` and ``addopts = ["--doctest-modules", "--doctest-glob=*.rst",
...]``. **CI runs `uv run pytest` with no path argument, which uses
testpaths and collects doctests in `pyphi/` source modules.** Bare-path
invocations (`pytest test/`, `pytest pyphi/specific.py`) **override
testpaths and skip the doctest sweep entirely** — local verifications
scoped this way will report green even when a doctest is broken.

When verifying a project as complete (especially renames, signature
changes, or anything touching `pyphi/` source), run `uv run pytest`
**without a path argument** at least once. The fast-lane shortcut
(`pytest test/ -m "not slow"`) is fine for inner-loop iteration but is
not a complete verification recipe.

Doctests don't run on `docs/*.rst` files either, because `docs/` isn't
in testpaths even though `--doctest-glob=*.rst` would match. Treat
`docs/*.rst` doctests as documentation that users can copy — verify by
reading, not by pytest.

### Performance regressions the φ goldens cannot see

Correctness tests are blind to cost, so two other gates carry it, and a change
that touches caching, hashing, or a full-state sweep should be checked against
both.

`test/integration/test_perf_counters.py` pins deterministic cProfile call
counts for the frames in `test/golden/perf.py::FRAMES`, which cover two
regression classes. *Redundant work* — the same operation performed more often
than necessary — is counted at PyPhi frames. *Cost per operation* — the same
operations, each more expensive — is counted at the frames dictionary collision
handling passes through (`Mapping.__eq__`, `FrozenMap.__getitem__`), because a
cache-key type whose hash stops separating distinct keys leaves every PyPhi
count identical while making each cache operation a linear scan. A new
cache-key type must also be declared in
`test/data_structures/test_hash_quality.py`, whose companion test instruments
the cache during real analyses and fails on an undeclared type.

Neither gate sees memory. Cache *occupancy* is asserted directly in
`test/cache/test_transient_repertoires.py`: a full-state sweep must admit a
number of entries that scales with the unit count, not the state count.

Fixture size is its own axis. The golden zoo tops out at four units, so costs
driven by the size of the whole system rather than of a mechanism are invisible
to it; the `specified_state` grain and the ring fixtures in
`test/golden/perf_fixtures.py` exist for that range and are perf-only (no φ
goldens, absent from `ALL_FIXTURES`).

When adding a guard here, verify it fails against the reverted defect. A pin
that cannot move is not a gate.

### Formalism pinning (tests that assert φ values)

A φ value is only meaningful relative to a formalism. Any test that asserts a
φ value must **pin its formalism explicitly** — never rely on the ambient
default. Pin with the complete preset-sourced context managers
(`IIT_3_CONFIG`, `IIT_4_CONFIG` in `test/conftest.py`, sourced from
`pyphi.conf.presets`), not a hand-listed subset of `iit.*` fields: setting
`iit.version` alone leaves the measures on the ambient default — the
partial-pin trap that silently recomputes under a different formalism when the
default changes. Tests that compute φ at module-fixture setup must pin inside
the fixture (a function-scoped autouse pin does not wrap module-fixture setup).

Exactly one test — `test_default_formalism_is_iit4_2026` — asserts the shipping
default; it is intentionally unpinned. To flip the default formalism: change
the default in `pyphi/conf/formalism.py`, update that assertion plus the two
default-dependent facade tests (`TestGlobalConfigFacade.test_layered_reads_work`
in `test/conf/test_config_layers.py` and `test_2023_omitted_metric_uses_default`
in `test/formalism/test_formalism_measure_threading.py`), and regenerate only
the `docs/` tutorial examples that demonstrate default behavior (CI doctests in
`pyphi/` compute no cap-sensitive φ).

### Running tests in parallel for faster feedback

The full suite takes a while. For faster signal, split into independent test
files and run them as parallel background jobs rather than sequentially in one
command:

- **Fast lane** (seconds-to-minute): `test_partition.py`,
  `test_subsystem_surface.py`, `test_golden_regression.py`,
  `test_invariants.py` (deterministic invariants, no Hypothesis)
- **Slow lane** (5-10 min): `test_invariants_hypothesis.py` (property
  tests with `@settings(max_examples=...)`)

Pattern: kick off the slow lane in background with
`run_in_background=true`, then run the fast lane in foreground. You'll
see the fast results in <1 min while the slow lane keeps running, and
get notified when the slow lane finishes via Monitor's `until` loop.

Don't bundle slow + fast into a single `pytest` invocation — pytest's
sequential collection means the fast result is gated on the slow one.

**Exit codes and pipes — read the summary, not the exit code.** Never pipe a
test run through `tail`/`head`/`grep` when the result matters: the pipeline's
exit code is the *last* command's, so `pytest … | tail -3` reports success
even when pytest fails or errors out. Redirect to a file instead
(`uv run pytest -q > log 2>&1`, exit code stays pytest's own), then read the
file's summary line before claiming green. This exact trap has shipped a
regression: two "exit 0" background runs each contained a real test failure
that went unread. Related: the slow lane is `uv run pytest -m slow --slow` —
the root conftest errors loudly if `--slow` is missing, so a bare `-m slow`
can never silently skip-and-pass, but only if that error is actually seen.

---

## Configuration System

### How Configuration Works

1. **Default configuration**: Defined as frozen dataclasses in
   `pyphi/conf/` — `formalism.py` (`IITConfig`, `ActualCausationConfig`),
   `infrastructure.py` (`InfrastructureConfig`), `numerics.py`
   (`NumericsConfig`).
2. **User configuration**: Loaded from `pyphi_config.yml` in working
   directory (nested format: top-level keys ``formalism`` /
   ``infrastructure`` / ``numerics``).
3. **Runtime changes**: `pyphi.config.option_name = value` (top-level
   write routes to the right layer) or `pyphi.config.numerics.override(...)`.
4. **Context managers**: `pyphi.config.override(...)` for temporary scopes.

Example:
```python
import pyphi

# Check current value
print(pyphi.config.numerics.precision)  # 13

# Change at runtime
pyphi.config.precision = 6

# Temporary change
with pyphi.config.override(precision=10):
    # Computation with higher precision
    pass
```

### Important Configuration Options

See [pyphi/conf/CLAUDE.md](file:///pyphi/conf/CLAUDE.md) for the full
option-by-option reference (loads automatically when working in that
directory).

---

## Repository Conventions

### Surfacing new functionality

When adding a feature, consider the MCP server (`pyphi/mcp/`): decide
whether the new information or functionality should be surfaced there — as
a tool in `server.py`, a resource in `resources.py`, or reference content
in `pyphi/mcp/content/` — and update those surfaces if so.

### Commit messages
Commit messages must succinctly describe what changed and why. Do not include anything related to the narrative flow of conversations with the user, or context that is irrelevant to the actual final set of changes. BAD: "User flagged an important issue. This commit fixes…". GOOD: "This commit fixes a bug where…".

### Committing specs and plans
Design specs and implementation plans (e.g. under `docs/superpowers/`) must
only be committed **after the user has explicitly approved them**. Do not
commit a spec or plan in the same breath as writing it — write it, ask the
user to review, and commit only once they sign off. The same applies to
substantive revisions of an already-approved spec/plan: re-confirm before
committing the revision.

### Using worktrees

The default is to work on whatever branch the conversation starts on. However, for significant chunks of work that require discussion and planning, you should prefer working in a git worktree (after confirming with the user).
**Create worktrees in `.claude/worktrees/`.**

---

## Quick Reference

### Key Files to Know

- [ROADMAP.md](file:///ROADMAP.md) - **Strategic 2.0 roadmap and schedule.** The single source of truth for what has landed and what remains; the Status Dashboard at the top is authoritative. Read it for current priorities, and keep it current (see ["Keeping this file up to date"](#keeping-this-file-up-to-date) below).
- [pyphi/__init__.py](file:///pyphi/__init__.py) - Main entry point
- [pyphi/system.py](file:///pyphi/system.py), [pyphi/substrate.py](file:///pyphi/substrate.py) - Core `System` / `Substrate` value types (formerly `Subsystem` / `Network`)
- [pyphi/formalism/](file:///pyphi/formalism/) - Formalism strategies: `iit3/`, `iit4/`, `actual_causation/`
- [pyphi/core/](file:///pyphi/core/) - Stateless kernel: repertoire algebra, TPM (`core/tpm/`), units
- [pyphi/conf/](file:///pyphi/conf/) - Layered configuration (formalism / infrastructure / numerics)
- [pyphi_config.yml](file:///pyphi_config.yml) - Default configuration
- [test/example_networks.py](file:///test/example_networks.py) - Test networks

### Common Commands

```bash
# Development setup
uv venv                                             # Create virtual environment
uv pip install -e ".[dev,parallel,visualize]"       # Install with dev dependencies

# Testing
just test                                            # Run tests (forwards args)
uv run pytest                                        # All tests
uv run pytest -k test_name                           # Specific test
uv run pytest --cov=pyphi                            # With coverage

# Benchmarking
just bench                                            # Quick local run (current env)
just bench-dashboard                                 # Build + serve the ASV HTML dashboard

# Documentation
just docs
open docs/_build/html/index.html

# Code quality
pre-commit run --all-files
```

---

## Keeping this file up to date

As the codebase changes, make sure to update the contents of this file as necessary.

### Keep ROADMAP.md current

[ROADMAP.md](file:///ROADMAP.md) is the strategic roadmap for the 2.0 release — the planned
refactors and features, their dependency-ordered schedule, and their status. **When you land or
change the status of any roadmapped work, update its row in the ROADMAP.md Status Dashboard in the
same change** (and any matching detail in "Remaining 2.0 Work"). The document has repeatedly drifted
— items implemented but left described as upcoming — so the dashboard is the single source of truth;
verify an item's status against the code, `changelog.d/`, and git history before trusting prose
elsewhere in the file. If you do substantial work that isn't on the roadmap, add it.

**A commit that settles a gate — a confirmation experiment, a proof, or a refutation — must
update the gated item's ROADMAP row in the same change.** A verdict that lives only in a commit
message or a `FINDINGS.md` will not be found by the next person reading the dashboard, and the
roadmap will keep describing the item as blocked or open long after the blocker is gone. This has
already happened: three theory gates were settled experimentally but left the dashboard describing
them as still-gated. When an experiment discharges (or kills) a gate, propagate the outcome to the
row, and to any wishlist entry or `PAPER-IDEAS.md` idea that cited the open question.

## graphify

This project can build a knowledge graph under graphify-out/ (god nodes, community structure, cross-file relationships) plus hand-built edges linking IIT paper concepts to the code that implements them (`implements`/`cites`), which answer "which function implements Theorem 1 / the intrinsic-difference measure / a given equation". Only the curated bridge edges are committed — `graphify-out/bridge-edges.json` (238 `implements`/`cites` edges + their endpoint nodes). The full `graph.json` and `GRAPH_REPORT.md` are large and regenerable, so they are **local-only** (gitignored) along with the rest of graphify-out/. Build them with `graphify update .`, then restore the committed bridge edges with `uv run python scripts/graphify_bridge.py inject`.

graphify is a standalone CLI, not a pyphi import dependency, so it is registered in the `[dependency-groups] dev` list (package name `graphifyy`, double-y; command is `graphify`). It installs with the rest of the dev tooling via `uv sync`, or on its own with `uv tool install 'graphifyy==0.8.44'`.

When to use it (optional — graphify is a convenience, not a required first step; it needs a local `graph.json`, which a fresh clone won't have until you run `graphify update .`):
- It earns its keep for **paper-to-code traceability** ("what implements concept X?", via `graphify path "<concept>" "<symbol>"` / `graphify explain "<concept>"`) — the bridge edges answer questions grep cannot — and for **broad orientation** across many files in code you don't already know.
- For targeted lookups in code you can already navigate, plain grep/Read are usually faster and more precise than a graph query (the bare `query` does a broad keyword sweep and can return a large, noisy neighborhood).
- Read graphify-out/GRAPH_REPORT.md (when present locally) only for broad architecture review.

Keeping it current:
- `graph.json`/`GRAPH_REPORT.md` are local. Run `graphify update .` to refresh the structural (AST) layer (cheap, deterministic, no API cost), then `uv run python scripts/graphify_bridge.py inject` to merge the committed bridge edges back into the freshly built graph.
- The committed `graphify-out/bridge-edges.json` is the version-controlled asset. The bridge edges do NOT refresh with `graphify update`; rebuild them deliberately (a focused multi-agent pass reading the IIT papers alongside their implementing modules, emitting `implements`/`cites` edges) after a release or before onboarding, then run `uv run python scripts/graphify_bridge.py extract` to refresh the committed sidecar (review the diff like a golden).

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
