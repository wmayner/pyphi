"""Tests for the top-level pyphi import surface."""

import pyphi


def test_all_has_no_duplicates():
    assert len(pyphi.__all__) == len(set(pyphi.__all__))


def test_all_includes_explicit_reexports():
    for name in ("LandscapeSection", "Perturbation", "SweepResult"):
        assert name in pyphi.__all__
        assert getattr(pyphi, name) is not None


def test_all_excludes_optional_dependency_submodules():
    # `from pyphi import *` must work on a base install; submodules that
    # require optional dependencies at import time stay out of __all__.
    assert "visualize" not in pyphi.__all__
    assert "mcp" not in pyphi.__all__


def test_dir_lists_lazy_submodules():
    listing = dir(pyphi)
    for name in ("examples", "macro", "dynamics", "visualize", "substrate_generator"):
        assert name in listing


def test_star_import_names_resolve():
    for name in pyphi.__all__:
        assert getattr(pyphi, name) is not None
