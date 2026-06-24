# Default: list available commands
default:
    @just --list

# Run tests
test *args:
    uv run pytest {{ args }}

# Run tests with coverage and open HTML report
coverage:
    uv run coverage run --source pyphi -m pytest
    uv run coverage html
    open htmlcov/index.html

# Build documentation
docs:
    cd docs && make html
    cp docs/_static/*.css docs/_build/html/_static
    cp docs/_static/*.png docs/_build/html/_static

# Serve documentation locally
serve-docs port="1337": docs
    cd docs/_build/html && uv run python -m http.server {{ port }}

# Run the benchmark suite once locally (current env, quick mode)
bench *args:
    cd benchmarks && uv run asv run --quick --python=same {{ args }}

# Build the ASV HTML dashboard and serve it locally (readable, left-aligned)
bench-dashboard:
    cd benchmarks && uv run asv publish && uv run asv preview

# Compare HEAD against the previous commit on one runner (builds both — slower)
bench-compare:
    cd benchmarks && uv run asv continuous --python=same HEAD~1 HEAD

# Run the deterministic perf-counter gate (the PR-blocking check)
perf-gate:
    uv run pytest test/test_perf_counters.py

# Regenerate the pinned perf call counts (review the diff like a golden)
perf-pins:
    uv run python scripts/gen_perf_counts.py

# Build and upload to PyPI
dist: clean-dist
    uv build
    uv run twine check dist/*
    uv run twine upload dist/*

# Build and upload to Test PyPI
test-dist: clean-dist
    uv build
    uv run twine check dist/*
    uv run twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# Clean build artifacts
clean:
    rm -rf dist __pycache__ **/__pycache__ **/pyphi.log htmlcov .coverage

[private]
clean-dist:
    rm -rf dist
