"""Tests for the top-level pyphi import surface."""

import importlib
import types

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


def test_entry_point_modules_are_modules_and_functions_stay_on_top():
    for module, function, cls in [
        ("analysis", "analyze", "Analysis"),
        ("sweeps", "sweep", "SweepResult"),
        ("optimization", "optimize", "OptimizationResult"),
    ]:
        mod = importlib.import_module(f"pyphi.{module}")
        assert isinstance(mod, types.ModuleType)
        assert getattr(mod, function) is getattr(pyphi, function)
        assert getattr(mod, cls) is getattr(pyphi, cls)


def test_star_import_does_not_rebind_common_names():
    namespace = {"log": "mine", "cache": "mine", "utils": "mine"}
    exec("from pyphi import *", namespace)
    assert namespace["log"] == "mine"
    assert namespace["cache"] == "mine"
    assert namespace["utils"] == "mine"


def test_all_lists_only_public_names():
    internal = {
        "deferred",
        "registry",
        "protocols",
        "utils",
        "node",
        "combinatorics",
        "serializable",
        "data_structures",
        "constants",
        "log",
        "cache",
        "TransitionSystem",
    }
    assert not internal & set(pyphi.__all__)
    assert all(hasattr(pyphi, name) for name in pyphi.__all__)
    for name in (
        "analyze",
        "sweep",
        "optimize",
        "Analysis",
        "SweepResult",
        "OptimizationResult",
        "Substrate",
        "System",
    ):
        assert name in pyphi.__all__


def test_models_all_has_no_private_names():
    assert not [n for n in pyphi.models.__all__ if n.startswith("_")]


def test_actual_declares_its_public_surface():
    assert "Transition" in pyphi.actual.__all__
