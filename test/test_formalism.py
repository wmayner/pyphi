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

    @property
    def config(self):
        from pyphi.conf import config as _global

        return _global.formalism

    def evaluate_mechanism(self, system, direction, mechanism, purview, **kwargs):  # noqa: ARG002
        return None

    def evaluate_mechanism_partition(self, *args, **kwargs):  # noqa: ARG002
        return None

    def evaluate_system(self, system, **kwargs):  # noqa: ARG002
        return None

    def build_phi_structure(self, system, **kwargs):  # noqa: ARG002
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
    """The global ``FORMALISM_REGISTRY`` is a ``FormalismRegistry`` instance
    and has the three concrete IIT formalisms registered."""
    assert isinstance(FORMALISM_REGISTRY, FormalismRegistry)
    assert "IIT_3_0" in FORMALISM_REGISTRY.all()
    assert "IIT_4_0_2023" in FORMALISM_REGISTRY.all()
    assert "IIT_4_0_2026" in FORMALISM_REGISTRY.all()


def test_concrete_formalisms_satisfy_protocol():
    """Each registered formalism satisfies the PhiFormalism Protocol."""
    for name in ("IIT_3_0", "IIT_4_0_2023", "IIT_4_0_2026"):
        formalism = FORMALISM_REGISTRY[name]
        assert isinstance(formalism, PhiFormalism), (
            f"{name} does not satisfy PhiFormalism"
        )
        assert formalism.name == name
        assert formalism.default_metric in formalism.compatible_metrics


def test_formalism_evaluate_system_matches_legacy_path():
    """``FORMALISM_REGISTRY['IIT_4_0_2023'].evaluate_system(s)`` produces the
    same SIA as ``pyphi.formalism.iit4.sia(s)``.

    Pins the wrapper's behavior against the underlying implementation
    before the cut-over commit moves the dispatch the other way.
    """
    from pyphi import examples
    from pyphi.formalism import iit4

    s = examples.basic_system()
    direct = iit4.sia(s)
    via_formalism = FORMALISM_REGISTRY["IIT_4_0_2023"].evaluate_system(s)
    assert direct == via_formalism, (
        f"IIT 4.0 (2023) formalism wrapper diverges from iit4.sia: "
        f"{direct.phi} vs {via_formalism.phi}"
    )


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
