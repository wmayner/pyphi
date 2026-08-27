# Releasing PyPhi

The checklist for cutting a release. Every gate below has caught a real
defect at least once; skipping one is how regressions ship.

## Verification gates

1. **Full fast suite, no path argument**: `uv run pytest -q > log 2>&1`,
   then read the summary line in the log. The bare invocation uses
   `testpaths` and includes the doctest sweep over `pyphi/`; a
   path-scoped run silently skips it. Never take a piped exit code as the
   verdict.
2. **Slow lane**: `uv run pytest -m slow --slow -q > log 2>&1`, read the
   summary. This runs the paper-reproduction acceptance suite (the published
   figures) and the large-fixture tests the fast lane never sees.
3. **Executed docs build**: `just docs` (sphinx with `-W`; every code cell
   executes at build). Docs do not build in CI, so this is the only gate
   against prose-versus-output drift and cells broken by API changes.
4. **Demo notebook**: re-execute `docs/examples/IIT_4.0_demo.ipynb` end to
   end (it is excluded from the docs build; only re-execution catches
   breaks). Run it through `nbclient` inside the project venv — a bare
   `jupyter nbconvert` can silently resolve to a system Jupyter and the
   wrong kernel.
5. **Packaging**: `uv build` (both sdist and wheel — the sdist→wheel path
   has failed while `uv build --wheel` passed), then install the wheel into
   a fresh venv and `python -c "import pyphi"` with a clean `PATH`.

## Release mechanics

6. **Changelog**: consume the fragments with
   `uv run towncrier build --version <X.Y.Z>`; fold the curated
   `RELEASE-NOTES-2.0.0.md` section into `CHANGELOG.md` for the 2.0.0
   release specifically. Verify no fragments remain in `changelog.d/`.
7. **CITATION.cff**: set `version` and `date-released`.
8. **README**: rewrite the release-status blockquote and the install
   instructions — they intentionally describe the pre-release state
   (PyPI ships 1.x) until the release is actually cut.
9. **Version metadata**: tag `v<X.Y.Z>`; confirm
   `importlib.metadata.version("pyphi")` reports the tag in a build from
   the tag.
10. **Docs deploy**: confirm the readthedocs build for the tag is green.

## After tagging

11. Push the tag and the release branch only after every gate above is
    green; publish to PyPI from CI, not a laptop.
12. Verify `pip install pyphi` in a fresh environment imports and runs the
    README example.
