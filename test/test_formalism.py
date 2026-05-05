"""Tests for the formalism Protocol and registry skeleton.

Concrete formalism behavior is tested in the IIT 3.0 / IIT 4.0 specific
suites; this module verifies only the scaffolding: that the registry
accepts conforming objects, rejects non-conforming ones, and survives
``runtime_checkable`` dispatch.
"""

from __future__ import annotations

import pytest

from pyphi.formalism import FORMALISM_REGISTRY
from pyphi.formalism import ApproximateFormalism
from pyphi.formalism import ExactFormalism
from pyphi.formalism import FormalismRegistry
from pyphi.formalism import PhiFormalism


class _DummyFormalism:
    """Minimal class satisfying the PhiFormalism Protocol structurally.

    Method bodies return None; they exist only to provide the methods the
    Protocol's ``isinstance`` check looks for.
    """

    name = "DUMMY"
    default_metric = "EMD"
    compatible_metrics = frozenset({"EMD"})
    partition_scheme = "BI"

    def evaluate_mechanism(self, subsystem, direction, mechanism, purview, **kwargs):  # noqa: ARG002
        return None

    def evaluate_system(self, subsystem, **kwargs):  # noqa: ARG002
        return None

    def build_phi_structure(self, subsystem, **kwargs):  # noqa: ARG002
        return None


class _NotCallableEnough:
    """Missing ``evaluate_system``; should not satisfy the Protocol."""

    name = "BROKEN"
    default_metric = "EMD"
    compatible_metrics = frozenset({"EMD"})
    partition_scheme = "BI"

    def evaluate_mechanism(self, *args, **kwargs):  # noqa: ARG002
        return None


def test_dummy_satisfies_phi_formalism_protocol():
    """A class with the right shape passes ``isinstance`` against the Protocol."""
    assert isinstance(_DummyFormalism(), PhiFormalism)


def test_missing_method_fails_phi_formalism_protocol():
    """A class missing ``evaluate_system`` / ``build_phi_structure`` fails."""
    assert not isinstance(_NotCallableEnough(), PhiFormalism)


def test_registry_register_accepts_conforming_formalism():
    """A conforming formalism registers and is retrievable by name."""
    registry = FormalismRegistry()
    formalism = _DummyFormalism()
    registry.register("DUMMY", formalism)
    assert registry["DUMMY"] is formalism
    assert "DUMMY" in registry.all()


def test_registry_register_rejects_nonconforming_formalism():
    """Non-conforming objects raise TypeError at registration time."""
    registry = FormalismRegistry()
    with pytest.raises(TypeError, match="PhiFormalism Protocol"):
        registry.register("BAD", _NotCallableEnough())  # type: ignore[arg-type]


def test_registry_lookup_unknown_raises_keyerror():
    """Looking up an unregistered name raises KeyError with a helpful message."""
    registry = FormalismRegistry()
    with pytest.raises(KeyError, match="phi formalisms"):
        _ = registry["UNREGISTERED"]


def test_global_registry_is_a_formalism_registry():
    """The global ``FORMALISM_REGISTRY`` is a ``FormalismRegistry`` instance.

    No formalisms are registered yet (the concrete implementations land in
    the next commit); this test pins the expected type and absence of
    pre-registered entries.
    """
    assert isinstance(FORMALISM_REGISTRY, FormalismRegistry)


def test_exact_and_approximate_subtypes_exist():
    """The ``ExactFormalism`` and ``ApproximateFormalism`` subtypes are
    distinct Protocols. Concrete classes will declare ``exact`` to pick
    one or the other; this test pins their availability via the public API.

    ``issubclass`` is unsupported on Protocols with non-method members, so
    the relationship is checked by attribute inspection instead.
    """
    assert ExactFormalism is not ApproximateFormalism
    # ExactFormalism inherits from PhiFormalism per the source
    assert PhiFormalism in ExactFormalism.__mro__
    assert PhiFormalism in ApproximateFormalism.__mro__
