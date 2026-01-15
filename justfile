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

# Run benchmarks
benchmark:
    cd benchmarks && uv run asv continuous develop

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
