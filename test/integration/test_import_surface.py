"""The top-level package imports lazily and exposes submodules on demand."""

import os
import subprocess
import sys

import pytest

import pyphi


def test_known_submodule_resolves_lazily():
    # examples is not a registrant module, so it is imported only on access.
    assert pyphi.examples.__name__ == "pyphi.examples"


def test_unknown_attribute_raises_attributeerror():
    with pytest.raises(AttributeError):
        _ = pyphi.definitely_not_a_real_submodule


def test_import_pyphi_does_not_eagerly_import_peripheral_submodules():
    # The robustness property: a broken or heavy peripheral submodule cannot
    # break `import pyphi`, because it is not imported until accessed.
    code = (
        "import sys, pyphi; "
        "leaked = [m for m in "
        "('pyphi.visualize', 'pyphi.examples', 'pyphi.macro', 'pyphi.matching') "
        "if m in sys.modules]; "
        "assert not leaked, leaked"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        env={**os.environ, "PYPHI_WELCOME_OFF": "1"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
