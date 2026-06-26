"""Regression test: the `result._inner if hasattr` unwrap pattern is retired.

The pattern ``result._inner if hasattr(result, "_inner") else result`` was a
transitional shim for code paths that needed to extract the raw array from a
``JointTPM`` wrapper.  This test locks in the retirement of that shim by
asserting that no production source file contains the pattern.

If this test ever fails it means a new shim was introduced; callers should be
updated to work directly with the typed TPM objects instead.
"""

from __future__ import annotations

import re
from pathlib import Path

# Root of the pyphi package (one level up from this test file).
_PYPHI_ROOT = Path(__file__).parent.parent.parent / "pyphi"

# Pattern that characterises the "unwrap" idiom.  We match the hasattr guard
# rather than bare ``._inner`` accesses so that the ``JointTPM`` slot
# definition itself does not trip the assertion.
_UNWRAP_PATTERN = re.compile(r"hasattr\s*\(.*,\s*['\"]_inner['\"]\)")


def test_inner_unwrap_pattern_absent() -> None:
    """No production source file contains the ``hasattr(..., '_inner')`` shim."""
    matches: list[str] = []
    for path in _PYPHI_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            if _UNWRAP_PATTERN.search(line):
                matches.append(
                    f"{path.relative_to(_PYPHI_ROOT.parent)}:{lineno}: {line.strip()}"
                )

    assert not matches, (
        "Found hasattr(_inner) unwrap shim(s) in production code — "
        "update callers to use typed TPM objects directly:\n" + "\n".join(matches)
    )
