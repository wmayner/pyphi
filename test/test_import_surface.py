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


def test_star_import_does_not_shadow_stdlib_names():
    """`from pyphi import *` must not rebind stdlib module names such as
    `warnings` and `types` to pyphi submodules."""
    import sys

    import pyphi

    exported = set(pyphi.__all__)
    stdlib_collisions = exported & set(sys.stdlib_module_names)
    assert not stdlib_collisions, stdlib_collisions


def test_estimate_analysis_in_all():
    import pyphi

    assert "estimate_analysis" in pyphi.__all__
    assert "AnalysisEstimate" in pyphi.__all__
