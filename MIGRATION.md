# Migrating from Conda to uv

This guide helps existing PyPhi developers migrate from conda-based workflows to uv.

## Why We Migrated

- **3-10x faster** installation and dependency resolution
- **Better reproducibility** with deterministic lockfiles (`uv.lock`)
- **Simpler workflow** - one modern tool instead of conda+pip hybrid
- **All dependencies available** - Graphillion, igraph, and pyemd all have pre-built wheels

## Quick Start

### Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Set Up Development Environment

```bash
cd /path/to/pyphi

# Create virtual environment
uv venv

# Activate it
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows

# Install PyPhi with all dev dependencies
uv pip install -e ".[dev,parallel,visualize,graphs,emd,caching]"
```

## Command Translation

| Task                   | Conda                                       | uv                                   |
| ---------------------- | ------------------------------------------- | ------------------------------------ |
| **Create environment** | `conda create -n pyphi python=3.12`         | `uv venv`                            |
| **Activate**           | `conda activate pyphi`                      | `source .venv/bin/activate`          |
| **Install deps**       | `conda env update -f environment.yml`       | `uv pip install -e ".[dev]"`         |
| **Install package**    | `conda install -c wmayner pyphi`            | `uv pip install pyphi`               |
| **Run tests**          | `pytest`                                    | `uv run pytest`                      |
| **Run Python**         | `python script.py`                          | `uv run python script.py`            |
| **Add dependency**     | Edit `environment.yml` + `conda env update` | Edit `pyproject.toml` + `uv lock`    |
| **List packages**      | `conda list`                                | `uv pip list`                        |
| **Update packages**    | `conda update --all`                        | `uv pip install --upgrade <package>` |

## Common Workflows

### Running Tests

**Before (conda):**
```bash
conda activate pyphi
pytest test/
```

**After (uv):**
```bash
# Option 1: With activated venv
source .venv/bin/activate
pytest test/

# Option 2: Without activation
uv run pytest test/

# Option 3: Using Make
make test
```

### Installing JupyterLab

**Before (conda):**
```bash
conda activate pyphi
conda install -c conda-forge jupyterlab jupyterlab-lsp python-lsp-server ipywidgets
```

**After (uv):**
```bash
uv pip install jupyterlab jupyterlab-lsp python-lsp-server ipywidgets
```

### Building Documentation

**Before (conda):**
```bash
conda activate pyphi
cd docs
make html
```

**After (uv):**
```bash
# Makefile now uses uv automatically
make docs

# Or manually:
uv run sphinx-build docs docs/_build/html
```

### Running Benchmarks

**Before (conda):**
```bash
conda activate pyphi
cd benchmarks
asv continuous develop
```

**After (uv):**
```bash
# Makefile now uses uv automatically
make benchmark
```

## Key Differences

### 1. Virtual Environment Location

- **Conda**: Environments stored centrally (e.g., `~/miniconda3/envs/pyphi`)
- **uv**: Environment in project directory (`.venv/`)

**Benefit**: Project-specific environments are easier to manage and don't clutter global namespace.

### 2. Dependency Specification

- **Conda**: `environment.yml` file
- **uv**: `pyproject.toml` (already the source of truth for PyPhi)

**Benefit**: Single source of truth, no sync issues between conda and pip dependencies.

### 3. Lockfiles

- **Conda**: No automatic lockfile (some tools add this)
- **uv**: Automatic `uv.lock` with exact versions of all dependencies

**Benefit**: Perfect reproducibility across machines and time.

### 4. Speed

- **Conda**: Can be slow, especially for large environments
- **uv**: 3-10x faster dependency resolution and installation

**Benefit**: Less waiting, faster iteration.

## Troubleshooting

### "Command not found: uv"

After installing uv, you may need to:
1. Close and reopen your terminal
2. Or run: `source ~/.bashrc` (Linux) or `source ~/.zshrc` (macOS)

### "Package not found" errors

uv uses PyPI by default. All PyPhi dependencies are available on PyPI with pre-built wheels.

If you had conda-specific packages, you'll need to find PyPI equivalents.

### Graphillion import errors

If you get import errors with Graphillion, rebuild it from source:

```bash
uv pip uninstall graphillion
uv pip install --no-binary graphillion graphillion
```

### Want to keep using conda?

That's fine! The conda package is still available (though deprecated):

```bash
conda install -c wmayner pyphi
```

Note that the conda package may not receive updates as frequently.

## FAQ

### Do I need to uninstall conda?

No! You can keep conda installed and use both. However, for PyPhi development, we recommend using uv.

### What about environment.yml?

The `environment.yml` file has been deprecated. All dependencies are now specified in `pyproject.toml`.

For JupyterLab setup, see the "Installing JupyterLab" section above.

### Can I still use pip?

Yes! `uv pip` is a drop-in replacement for `pip`, but you can also use regular `pip` if you prefer:

```bash
# These are equivalent
uv pip install pyphi
pip install pyphi
```

However, `uv pip` is faster and integrates with `uv.lock`.

### How do I update dependencies?

Edit `pyproject.toml` and run:

```bash
uv lock              # Update lockfile
uv pip install -e .  # Install updates
```

### What about CI/CD?

CI/CD workflows are being updated to use uv. See `.github/workflows/` for examples.

### Where can I learn more about uv?

- **Documentation**: https://github.com/astral-sh/uv
- **Guide**: https://docs.astral.sh/uv/

## Need Help?

If you encounter issues during migration:

1. Check this guide first
2. Search [GitHub Issues](https://github.com/wmayner/pyphi/issues)
3. Ask in the [pyphi-users group](https://groups.google.com/forum/#!forum/pyphi-users)
4. Open a new issue if needed

## Summary

The migration from conda to uv is straightforward:

1. ✅ Install uv
2. ✅ Run `uv venv` in project directory
3. ✅ Run `uv pip install -e ".[dev,parallel,visualize,graphs,emd,caching]"`
4. ✅ Use `uv run` prefix for commands or activate `.venv`
