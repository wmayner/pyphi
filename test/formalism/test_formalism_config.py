"""Tests that registered formalisms satisfy the PhiFormalism protocol."""

from __future__ import annotations

import pytest

from pyphi.formalism import FORMALISM_REGISTRY
from pyphi.formalism.base import PhiFormalism


@pytest.mark.parametrize("version", ["IIT_3_0", "IIT_4_0_2023", "IIT_4_0_2026"])
def test_registered_formalisms_satisfy_protocol(version):
    formalism = FORMALISM_REGISTRY[version]
    instance = formalism() if isinstance(formalism, type) else formalism
    assert isinstance(instance, PhiFormalism)
    assert instance.name == version
