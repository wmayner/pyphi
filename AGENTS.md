# CLAUDE.md

**AI Assistant Guide for PyPhi Development**

This document provides context and guidelines for AI assistants working on PyPhi, a Python library that implements the mathematical formalism of Integrated Information Theory (IIT).

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Critical Context](#critical-context)
3. [Architecture & Organization](#architecture--organization)
4. [Development Guidelines](#development-guidelines)
5. [Testing Strategy](#testing-strategy)
6. [Configuration System](#configuration-system)
7. [Common Pitfalls](#common-pitfalls)
8. [Maintenance Notes](#maintenance-notes)

---

## Project Overview

### What is PyPhi?

PyPhi is a computational implementation of **Integrated Information Theory (IIT)**, a mathematical framework for understanding consciousness and integrated information in physical systems. The library computes **Φ (phi)**, the measure of integrated information, along with related quantities.

### Scientific Context

- **Domain**: Neuroscience, consciousness studies, complex systems
- **Primary users**: Researchers, academics, computational neuroscientists
- **Computational characteristics**: Heavy numerical computation, combinatorially expensive operations
- **Theory versions**: Supports both IIT 3.0 and IIT 4.0 (currently on 4.0)

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

### Resources

- **Documentation**: https://pyphi.readthedocs.io
- **Repository**: https://github.com/wmayner/pyphi
- **User group**: https://groups.google.com/forum/#!forum/pyphi-users
- **Tutorial**: IIT 4.0 demo notebook in `docs/examples/IIT_4.0_demo.ipynb`

---

## Critical Context

### Mathematical Correctness is Paramount

**This is scientific software implementing a precise mathematical formalism.**

- Small bugs can invalidate research results
- Numerical precision matters deeply (configured via `PRECISION` setting)
- Changes to core computation logic require extreme care
- When in doubt, consult the IIT papers and existing tests

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

### Core Abstractions (pre v2.0 refactoring)

The library is built around these primary objects:

1. **`Network`** (`pyphi/network.py`)
   - Represents a system of nodes with causal relationships
   - Defined by a Transition Probability Matrix (TPM) and connectivity matrix
   - Main object on which computations are performed

2. **`Subsystem`** (`pyphi/subsystem.py`)
   - A subset of nodes from a Network in a particular state
   - Φ is computed over subsystems
   - Handles repertoire computation, mechanism evaluation

4. **TPM (Transition Probability Matrix)** (`pyphi/tpm.py`)
   - Core data structure defining system dynamics
   - Can be deterministic or probabilistic
   - Multiple representations: state-by-node, state-by-state

### Module Organization (pre v2.0 refactoring)

```
pyphi/
├── __init__.py              # Main entry point, lifts key interfaces
├── compute/                 # Main computational entry points
│   ├── network.py           # Network-level computations
│   └── subsystem.py         # Subsystem-level computations
├── models/                  # Data structures for results
│   ├── subsystem.py         # CauseEffectStructure, Concept, etc.
│   ├── mechanism.py         # RepertoireIrreducibilityAnalysis
│   ├── cuts.py              # Partition/cut representations
│   └── ...
├── metrics/                 # Distance measures for integration
│   ├── ces.py               # Cause-effect structure distances
│   └── distribution.py      # Repertoire distance measures
├── new_big_phi/             # IIT 4.0 implementation
│   └── __init__.py          # System-level analysis (Φ_s)
├── partition.py             # Partitioning schemes
├── repertoire.py            # Repertoire computation
├── parallel/                # Parallelization infrastructure
│   ├── tree.py              # Parallel tree computation
│   └── progress.py          # Progress bar management
├── cache/                   # Caching systems
│   ├── redis.py             # Redis cache backend
│   └── cache_utils.py       # Cache utilities
├── network_generator/       # Generate example networks
├── visualize/               # Visualization tools (optional dep)
└── ...
```

### IIT Version Switching (pre v2.0 refactoring)

The library supports both IIT 3.0 and IIT 4.0:

- **Config setting**: `IIT_VERSION: 4.0` in `pyphi_config.yml`
- **IIT 4.0 code**: Primarily in `pyphi/new_big_phi/`
- **IIT 3.0 code**: Distributed throughout the codebase (legacy)

The formalism differences are significant:
- IIT 3.0: Focuses on cause-effect structure (Φ)
- IIT 4.0: Adds system-level integration (Φ_s), relations, distinctions

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
   - See [pyphi_config.yml](file:///pyphi_config.yml) and [pyphi/conf.py](file:///pyphi/conf.py)

### Code Quality Standards

1. **Type Hints**
   - Add type hints to new code
   - Gradually add to existing code when touching it
   - Use `Optional`, `Tuple`, `Iterable` appropriately

2. **Documentation**
   - Docstrings should explain the *why*, not just the *what*
   - However, docstrings should NOT include any language that attempts to justify the implementation choices made in terms of the agent's planning. The sole focus should be to document. Example of what NOT to write: "This design is the optimal choice because...", etc.
   - Reference IIT concepts and terminology
   - Include mathematical formulas where relevant

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

### Development Workflow

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Install development dependencies
uv pip install -e ".[dev,parallel,visualize,graphs,emd,caching]"

# Run type checking
uv run pyright pyphi

# Run benchmarks
make benchmark

# Build documentation
make docs

# Check configuration
uv run python -c "import pyphi; print(pyphi.config)"
```

### Code Style

- **Formatting & Linting**: Project uses Ruff (configured in `pyproject.toml`)
  - **Enabled rule sets**: pycodestyle, pyflakes, isort, bugbear, comprehensions, pyupgrade, return statements, unused arguments, pathlib, pylint, performance, and Ruff-specific rules
  - **Special allowances**: Relaxed limits for scientific computing (complex functions, many arguments, magic values)
  - **Per-file ignores**: Tests allow fixtures and assertions, profiling allows print statements
  - **Do not run unsafe fixes with Ruff without first getting permission from the user.**
- **Type Checking**: pyright (configured in `pyproject.toml`)
  - Uses "standard" type checking mode
  - Better numpy type inference than mypy
  - Faster and more accurate for scientific Python code
- **Pre-commit hooks**: Configured in `.pre-commit-config.yaml`
  - Ruff linter and formatter
  - pyright type checker
  - Standard file checks (trailing whitespace, large files, etc.)

## Python version

**We will support only Python 3.12+ for this version.** Therefore, when writing code, **do not attempt to maintain backward compatibility with previous Python versions.**

---

## Testing Strategy

### Test Organization

```
pyphi/                       # Doctests
docs/                        # Doctests
test/                        # Unit, regression, and e2e tests
├── conftest.py              # Pytest configuration and fixtures
├── example_networks.py      # Reusable network definitions
├── test_*.py                # ~460 test functions across ~30 files
└── data/                    # Test data (JSON, etc.)
    ├── PQR_CES.json
    └── ...
```

**Note**: Test configuration is now in `pyproject.toml` under `[tool.pytest.ini_options]`.

### Testing Approaches

1. **Unit Tests**: Test individual functions and methods
2. **Integration Tests**: Test complete computations (e.g., full Φ calculation)
3. **Property-Based Tests**: Use Hypothesis for invariant testing
4. **Regression Tests**: Ensure results match expected values from papers
5. **Doctests**: Ensure documentation is current and documented behavior is as expected

### Example Networks

Use the pre-defined networks in `pyphi.examples`:
```python
from pyphi import examples

# Standard test networks
network = examples.basic_network()
network = examples.xor_network()
network = examples.fig1a()  # From IIT 3.0 paper
```

### Running Tests

```bash
# All tests, including doctests
uv run pytest

# Test suite
uv run pytest test/

# Specific test file
uv run pytest test/test_subsystem.py

# Specific test function
uv run pytest test/test_subsystem.py::test_cause_repertoire

# With coverage
uv run coverage run --source pyphi -m pytest
uv run coverage html

# Watch mode (with watchdog installed)
make test
```

**Note**: Coverage configuration is now in `pyproject.toml` under `[tool.coverage.*]`.

#### Doctest scope — important

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

#### Computational Behavior (``config.formalism.iit``)

- **`version`**: ``"IIT_3_0"`` / ``"IIT_4_0_2023"`` / ``"IIT_4_0_2026"``
- **`shortcircuit_sia`**: Short-circuit if reducibility detected (default: true)

#### Numerics (``config.numerics``)

- **`precision`**: Numerical precision for phi comparisons (default: 13)

#### Performance & Parallelization (``config.infrastructure``)

- **`parallel`**: Global switch for parallelization (default: false)
- **`parallel_workers`**: CPU cores to use (default: -1 = all)
- **`parallel_backend`**: ``"local"`` (ProcessPoolExecutor) or ``"auto"``
- **`parallel_*_evaluation`**: Fine-grained per-level dicts with keys
  ``parallel`` / ``chunksize`` / ``sequential_threshold`` / ``progress``
  (e.g. ``parallel_concept_evaluation``, ``parallel_complex_evaluation``,
  ``parallel_partition_evaluation``, ``parallel_purview_evaluation``,
  ``parallel_mechanism_partition_evaluation``, ``parallel_relation_evaluation``)

#### Caching (``config.infrastructure``)

- **`cache_repertoires`**: Cache repertoire computations (default: true)
- **`cache_potential_purviews`**: Cache purviews (default: true)
- **`clear_system_caches_after_computing_sia`**: Clear after each SIA (default: false)
- **`maximum_cache_memory_percentage`**: Memory limit for in-memory caches (default: 50)

#### Measures (``config.formalism.iit``)

- **`mechanism_phi_measure`**: Mechanism-level repertoire-distance measure
  (default: ``"GENERALIZED_INTRINSIC_DIFFERENCE"``)
- **`system_phi_measure`**: System-level phi measure
  (default: ``"GENERALIZED_INTRINSIC_DIFFERENCE"``; ``"INTRINSIC_INFORMATION"``
  enables the Eq. 23 cap in IIT 4.0 2026)
- **`ces_measure`**: Cause-effect-structure distance measure
  (default: ``"SUM_SMALL_PHI"``)
- **`config.formalism.actual_causation.alpha_measure`**: AC alpha measure
  (default: ``"PMI"``)

#### Partitioning (``config.formalism.iit``)

- **`mechanism_partition_scheme`**: Default ``"JOINT_PARTITION_ALL"``
- **`system_partition_scheme`**: Default ``"DIRECTED_SET_PARTITION"``

#### Debugging & Output (``config.infrastructure``)

- **`log_file`**: Log file path (default: ``"pyphi.log"``)
- **`log_file_level`**: File logging level (default: ``"INFO"``)
- **`log_stdout_level`**: Console logging level (default: ``"WARNING"``)
- **`progress_bars`**: Show progress bars (default: true)
- **`repr_verbosity`**: Detail level in ``repr()`` output (default: 2)
- **`welcome_off`**: Suppress welcome message (default: false)

---

## Common Pitfalls

### 1. Numerical Precision Issues

**Problem**: Floating-point comparisons fail due to precision
```python
# Bad
if phi == 0.0:  # May fail due to floating point error

# Good
from pyphi import utils
if utils.is_zero(phi):  # Respects config.PRECISION
```

**Key functions**:
- `utils.is_zero(x)`
- `utils.is_positive(x)`
- `utils.eq(x, y)`

### 2. Configuration Not in Working Directory

**Problem**: `pyphi_config.yml` must be in the directory where Python is executed
```bash
# This won't find config in /my/project/
cd /somewhere/else
python -c "import pyphi"  # Uses defaults!

# This will
cd /my/project
python -c "import pyphi"  # Loads ./pyphi_config.yml
```

### 3. Missing Optional Dependencies

**Problem**: Import errors for optional features

Parallel computation:
```bash
uv pip install pyphi[parallel]  # Installs Ray
```

Visualization:
```bash
uv pip install pyphi[visualize]  # matplotlib, plotly, seaborn, networkx
```

Graph analysis:
```bash
uv pip install pyphi[graphs]  # igraph, networkx
```

### 4. TPM Format Confusion

**Problem**: TPM can be in different formats

- **State-by-node**: Most common, rows are states, columns are nodes
- **State-by-state**: Rows are current states, columns are next states
- **Multidimensional**: High-dimensional array indexed by node states

**Solution**: Use conversion utilities in `pyphi.convert` or `pyphi.tpm`

### 5. Subsystem State Validation

**Problem**: Creating subsystem with invalid state

PyPhi validates that the subsystem state is consistent with the network TPM (if `VALIDATE_SUBSYSTEM_STATES` is true).

**Solution**: Use valid states or disable validation for special cases

### 6. Cache Memory Explosion

**Problem**: Large networks can fill memory with cached repertoires

**Solution**: Adjust `MAXIMUM_CACHE_MEMORY_PERCENTAGE` or disable caching:
```python
pyphi.config.CACHE_REPERTOIRES = False
```

### 7. Parallel Computation Not Working

**Problem**: Set `PARALLEL=True` but still running sequentially

**Checklist**:
1. Is `pyphi[parallel]` installed?
2. Is the specific operation's parallel flag enabled? (e.g., `PARALLEL_CONCEPT_EVALUATION.parallel`)
3. Is the problem size above `sequential_threshold`?
4. Check Ray initialization in logs

### 8. IIT Version Mismatch

**Problem**: Using IIT 3.0 code/examples with IIT 4.0 config (or vice versa)

**Solution**: Check `config.IIT_VERSION` and use appropriate functions:
- IIT 4.0: `pyphi.new_big_phi.phi_structure()`
- IIT 3.0: `pyphi.compute.ces()`, `pyphi.compute.big_phi()`

---

## Maintenance Notes

### Known Issues & Technical Debt

From [TODO.md](file:///TODO.md) and codebase inspection:

1. **Type Hints Incomplete**
   - Many functions lack type annotations
   - Gradually add when touching code

2. **API Documentation Outdated**
   - Needs regeneration after recent reorganization
   - Use `apidoc` or similar tool

3. **Redis Cache Underutilized**
   - Infrastructure exists but may not be fully leveraged
   - Consider for distributed computation scenarios

4. **Unified Partitioning Scheme Needed**
   - Multiple partition types, could be more consistent
   - See `pyphi/partition.py`

5. **IIT 3.0 Module Separation**
   - IIT 3.0 code should be isolated for clarity
   - Currently mixed throughout codebase

### File Hygiene Issues

**Untracked files to ignore**:
- `test-iit4.ipynb`, `visualize-example.ipynb` - Experimental notebooks
- `test/test_parallel2.py`, `test/test_serialization.py` - Experimental tests

### Code Organization Observations

**Strengths**:
- Clear separation of concerns (compute, models, metrics)
- Registry pattern for extensibility
- Comprehensive configuration system
- Good test coverage of core functionality

**Areas for Improvement**:
- Some modules are quite large (`subsystem.py` ~1,900 lines, `conf.py` ~1,000 lines)
- IIT 3.0 vs 4.0 code not clearly separated (except `new_big_phi/`)
- Circular import issues managed with deferred imports
- Some inconsistency in naming conventions

### Refactoring Priorities

When improving the codebase, prioritize:

1. **Add type hints** - Improves IDE support and catches bugs
2. **Improve test coverage** - Especially edge cases
3. **Separate IIT versions** - Make version differences explicit
4. **Document complex algorithms** - Especially partition evaluation
5. **Reduce code duplication** - Look for repeated patterns
6. **Performance profiling** - Identify bottlenecks before optimizing

### Testing Improvements Needed

1. **Increase coverage** of edge cases
2. **Property-based tests** for mathematical invariants
3. **Performance regression tests** via benchmarking
4. **Integration tests** for IIT 4.0 (newer code path)
5. **Parallel computation tests** (may require special setup)

---

## Working with AI Assistants

### Effective Collaboration

**Do**:
- Ask for explanations of unfamiliar IIT concepts
- Request code review before submitting changes
- Ask for help writing property-based tests
- Request refactoring suggestions with justification

**Don't**:
- Make changes to core computation logic without understanding
- Assume existing code is bug-free (it needs maintenance!)
- Skip testing because "it's just a small change"
- Ignore numerical precision requirements

### Example Workflows

#### Adding a New Feature
1. Read relevant existing code
2. Write tests first (TDD)
3. Implement feature
4. Run tests and fix failures
5. Add documentation
6. Create changelog fragment in `changelog.d/`
7. Request code review

#### Fixing a Bug
1. Write a failing test that reproduces the bug
2. Investigate root cause
3. Fix the issue
4. Verify test passes
5. Check for similar bugs elsewhere
6. Add regression test
7. Create changelog fragment in `changelog.d/`

#### Refactoring
1. Ensure tests exist for current behavior
2. Make incremental changes
3. Run tests after each change
4. Verify performance hasn't regressed
5. Update documentation if API changes

#### Commit messages
Commit messages must succinctly describe what changed and why. Do not include anything related to the narrative flow of conversations with the user, or context that is irrelevant to the actual final set of changes. BAD: "User flagged an important issue. This commit fixes…". GOOD: "This commit fixes a bug where…".

#### Committing specs and plans
Design specs and implementation plans (e.g. under `docs/superpowers/`) must
only be committed **after the user has explicitly approved them**. Do not
commit a spec or plan in the same breath as writing it — write it, ask the
user to review, and commit only once they sign off. The same applies to
substantive revisions of an already-approved spec/plan: re-confirm before
committing the revision.

#### Using worktrees

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
make test                                            # Watch mode
uv run pytest                                        # All tests
uv run pytest -k test_name                           # Specific test
uv run pytest --cov=pyphi                            # With coverage

# Benchmarking
make benchmark

# Documentation
make docs
open docs/_build/html/index.html

# Code quality
pre-commit run --all-files
```

### Getting Help

- **Documentation**: https://pyphi.readthedocs.io
- **Issues**: https://github.com/wmayner/pyphi/issues
- **Discussion**: https://groups.google.com/forum/#!forum/pyphi-users
- **IIT 4.0 Paper**: https://doi.org/10.1371/journal.pcbi.1011465
- **PyPhi Paper**: https://doi.org/10.1371/journal.pcbi.1006343

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

## Final Notes

PyPhi implements a complex mathematical theory with real-world scientific applications. Changes to this codebase can affect research results. Approach all modifications with care, test thoroughly, and when in doubt, consult the theoretical papers and existing tests.

The project needs maintenance and refactoring work, which presents an opportunity to improve code quality while preserving mathematical correctness. Incremental improvements with comprehensive testing are the best approach.

**Remember**: This is scientific software. Correctness > performance > elegance.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships. The graph also carries hand-built edges linking IIT paper concepts to the code that implements them (`implements`/`cites`), so it can answer "which function implements Theorem 1 / the intrinsic-difference measure / a given equation". `graph.json` and `GRAPH_REPORT.md` are committed and shared; everything else under graphify-out/ is gitignored local state.

graphify is a standalone CLI, not a pyphi import dependency, so it is registered in the `[dependency-groups] dev` list (package name `graphifyy`, double-y; command is `graphify`). It installs with the rest of the dev tooling via `uv sync`, or on its own with `uv tool install 'graphifyy==0.8.44'`.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- For paper-to-code traceability ("what implements concept X?"), prefer `graphify path "<concept>" "<symbol>"` and `graphify explain "<concept>"`. They give precise answers, whereas the bare `query` does a broad keyword sweep and returns a large, noisy neighborhood.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.

Keeping the graph current:
- After modifying code, run `graphify update .` to refresh the structural (AST) layer — cheap, deterministic, no API cost. Do this routinely.
- The IIT paper-to-code bridge edges do NOT refresh with `graphify update`; they go stale when implementing modules are renamed or rewritten. Rebuild them deliberately (a focused multi-agent pass reading the IIT papers alongside their implementing modules, emitting `implements`/`cites` edges) after a release or before onboarding — not on every commit.
