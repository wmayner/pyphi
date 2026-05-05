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

### Maintenance Status

**The project has not been carefully maintained recently and needs refactoring/testing work.**

Current issues (see [TODO.md](TODO.md)):
- Incomplete type hints
- API documentation needs updating
- Some code organization needs improvement
- Test coverage could be better (currently ~460 test functions, ~8,300 lines of tests)

**Approach changes conservatively**:
- Read existing code thoroughly before modifying
- Prioritize refactoring and testing over new features
- Don't assume the current implementation is optimal
- Look for inconsistencies and opportunities to improve clarity

---

## Architecture & Organization

### Core Abstractions

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

### Module Organization

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

### IIT Version Switching

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

# Run tests
make test
# or
uv run pytest test/

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

**We will support only Python 3.12+ for this version.** Therefore, when writing code, do not attempt to maintain backward compatibility with previous Python versions.

---

## Testing Strategy

### Test Organization

```
test/
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
# All tests
uv run pytest

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

---

## Configuration System

### How Configuration Works

1. **Default configuration**: Defined in `pyphi/conf.py`
2. **User configuration**: Loaded from `pyphi_config.yml` in working directory
3. **Runtime changes**: `pyphi.config.OPTION_NAME = value`
4. **Context managers**: Temporarily change settings

Example:
```python
import pyphi

# Check current value
print(pyphi.config.PRECISION)  # 13

# Change at runtime
pyphi.config.PRECISION = 6

# Temporary change
with pyphi.config.override(PRECISION=10):
    # Computation with higher precision
    pass
```

### Important Configuration Options

#### Computational Behavior

- **`IIT_VERSION`**: `"3.0"` or `"4.0"` - Theory version
- **`PRECISION`**: Numerical precision for phi comparisons (default: 13)
- **`SHORTCIRCUIT_SIA`**: Short-circuit if reducibility detected (default: true)

#### Performance & Parallelization

- **`PARALLEL`**: Global switch for parallelization (default: false)
  - Requires `pip install pyphi[parallel]` (Ray dependency)
- **`NUMBER_OF_CORES`**: CPU cores to use (default: -1 = all)
- **`PARALLEL_*_EVALUATION`**: Fine-grained parallel control
  - `parallel`: Enable for this operation
  - `chunksize`: Items per chunk
  - `sequential_threshold`: Don't parallelize below this size
  - `progress`: Show progress bars

#### Caching

- **`CACHE_REPERTOIRES`**: Cache repertoire computations (default: true)
- **`CACHE_POTENTIAL_PURVIEWS`**: Cache purviews (default: true)
- **`REDIS_CACHE`**: Use Redis for distributed caching (default: false)
- **`MAXIMUM_CACHE_MEMORY_PERCENTAGE`**: Memory limit for in-memory caches (default: 50)

#### Distance Measures

- **`REPERTOIRE_DISTANCE`**: Integration measure (default: `"GENERALIZED_INTRINSIC_DIFFERENCE"`)
  - Options: `"EMD"`, `"KLD"`, `"L1"`, `"GID"`, etc.
- **`CES_DISTANCE`**: Big Phi measure (default: `"SUM_SMALL_PHI"`)

#### Partitioning

- **`PARTITION_TYPE`**: Mechanism partition scheme (default: `"ALL"`)
- **`SYSTEM_PARTITION_TYPE`**: System partition scheme (default: `"SET_UNI/BI"`)
- **`SYSTEM_CUTS`**: Cut style for IIT 3.0 (default: `"3.0_STYLE"`)

#### Debugging & Output

- **`LOG_FILE`**: Log file path (default: `"pyphi.log"`)
- **`LOG_FILE_LEVEL`**: File logging level (default: `"INFO"`)
- **`LOG_STDOUT_LEVEL`**: Console logging level (default: `"WARNING"`)
- **`PROGRESS_BARS`**: Show progress bars (default: true)
- **`REPR_VERBOSITY`**: Detail level in `repr()` output (default: 2)
- **`WELCOME_OFF`**: Suppress welcome message (default: false)

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

---

## Quick Reference

### Key Files to Know

- [pyphi/__init__.py](file:///pyphi/__init__.py) - Main entry point
- [pyphi/compute/](file:///pyphi/compute/) - Computational entry points
- [pyphi/subsystem.py](file:///pyphi/subsystem.py) - Core subsystem logic
- [pyphi/conf.py](file:///pyphi/conf.py) - Configuration system
- [pyphi/new_big_phi/__init__.py](file:///pyphi/new_big_phi/__init__.py) - IIT 4.0 implementation
- [pyphi_config.yml](file:///pyphi_config.yml) - Default configuration
- [test/example_networks.py](file:///test/example_networks.py) - Test networks

### Common Commands

```bash
# Development setup
uv venv                                              # Create virtual environment
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

## Final Notes

PyPhi implements a complex mathematical theory with real-world scientific applications. Changes to this codebase can affect research results. Approach all modifications with care, test thoroughly, and when in doubt, consult the theoretical papers and existing tests.

The project needs maintenance and refactoring work, which presents an opportunity to improve code quality while preserving mathematical correctness. Incremental improvements with comprehensive testing are the best approach.

**Remember**: This is scientific software. Correctness > performance > elegance.
