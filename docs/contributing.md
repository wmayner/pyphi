# Contributing

PyPhi is developed on [GitHub](https://github.com/wmayner/pyphi). Bug
reports and feature requests go to the
[issue tracker](https://github.com/wmayner/pyphi/issues); questions about
the software or the theory to the
[pyphi-users group](https://groups.google.com/forum/#!forum/pyphi-users).

## Set up

Fork the project, clone your fork, and install the runtime extras and the
development tooling with [uv](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/YOUR_USERNAME/pyphi.git
cd pyphi
uv sync --all-extras --group dev
```

The common tasks are recipes in the `justfile` (install
[just](https://github.com/casey/just)): `just test` runs the test suite,
`just docs` builds this site, and `just bench` runs the benchmarks. Run any
Python command through `uv run`.

## Tests

`uv run pytest` runs the fast suite together with the doctests in the
package; run it without a path argument before calling a change done, since
a path argument skips the doctests. Tests marked slow run separately with
`uv run pytest -m slow --slow`. A test that asserts a φ value pins its
formalism with a preset (`IIT_3_CONFIG` or `IIT_4_CONFIG` from
`test/conftest.py`) rather than relying on the default.

To add a reproduction of a published figure, follow the pattern in
`test/integration/test_paper_reproduction.py`: quote the figure and the
paper, pin the formalism the paper used, build the substrate from the
paper's weights, and check that a perturbation of the fixture changes the
value, so the test is not vacuous.

## Changelog fragments

Every user-facing change adds a file `changelog.d/<name>.<type>.md`, where
`<name>` is an issue number or a short slug and `<type>` is one of
`feature`, `change`, `config`, `optimization`, `fix`, `doc`, `refactor`, or
`misc`. The release process folds the fragments into `CHANGELOG.md`.

## Documentation

The pages under `docs/` are MyST Markdown; pages with `{code-cell}` blocks
execute during the build, and the build treats warnings as errors, so a
claim that stops being true fails the build. The pages under
`docs/getting-started/` and `docs/tutorials/` are paired with notebooks
that a pre-commit hook regenerates; stage the regenerated `.ipynb` with
your change. Build locally with `just docs` and open
`docs/_build/html/index.html`.

## Style

Code is formatted and linted with ruff and type-checked with pyright
through the pre-commit hooks (`pre-commit run --all-files`). Docstrings
follow the NumPy style. New code carries type hints.
