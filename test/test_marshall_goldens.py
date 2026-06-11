"""Regression goldens from Marshall et al. 2024's committed result sets.

The expected values are parsed from the authors' verbatim
``summary.txt`` files in ``test/data/marshall2024/`` (see that
directory's README for provenance and the known upstream
discrepancies). Micro sets are reproduced with ``System`` over the
example substrates; macro sets with ``System`` over substrates built
from the authors' macro TPMs. The intrinsic-units candidate systems
(``MacroSystem``) are a different object and get project-recorded
goldens of their own.
"""

import itertools
import re
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data" / "marshall2024"

# Greek characters used in the data files, kept out of source literals.
PHI = "\u03c6"
ALPHA = "\u03b1"
BETA = "\u03b2"

_LINE = re.compile(rf"^{PHI}_s\((\S+)\) = (\S+)$")

# Per-set node-label alphabets, in node-index order.
LABELS = {
    "cg_micro": "ABCD",
    "cg_macro": ALPHA + BETA,
    "bbx_micro": "ABCDEFGH",
    "bbx_macro": ALPHA + BETA,
    "min_micro": "AB",
    "min_macro": ALPHA,
    "bu_micro": "ABC",
    "sfn_micro": "ABCD",
    "sfnn_micro": "ABCD",
    "sfs_micro": "ABCD",
}

EXPECTED_COUNTS = {
    "cg_micro": 15,
    "cg_macro": 3,
    "bbx_micro": 251,
    "bbx_macro": 3,
    "min_micro": 3,
    "min_macro": 1,
    "bu_micro": 7,
    "sfn_micro": 15,
    "sfnn_micro": 15,
    "sfs_micro": 15,
}

# The four subsystems absent from the authors' committed bbx results.
BBX_MISSING = ("ABCEFGH", "ABDEFGH", "ACDEFGH", "BCDEFGH")


def load_summary(name):
    """Parse a committed summary file into ``{subsystem_label: phi}``."""
    out = {}
    path = DATA_DIR / f"{name}.summary.txt"
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _LINE.match(line.strip())
        if match:
            out[match.group(1)] = float(match.group(2))
    return out


def indices_of(name, label):
    """Node indices of a subsystem label in a set's label alphabet."""
    return tuple(LABELS[name].index(ch) for ch in label)


class TestParser:
    @pytest.mark.parametrize("name", sorted(EXPECTED_COUNTS))
    def test_counts(self, name):
        assert len(load_summary(name)) == EXPECTED_COUNTS[name]

    @pytest.mark.parametrize("name", sorted(EXPECTED_COUNTS))
    def test_labels_map_to_indices(self, name):
        n = len(LABELS[name])
        for label in load_summary(name):
            indices = indices_of(name, label)
            assert len(set(indices)) == len(indices)
            assert all(0 <= i < n for i in indices)

    def test_bbx_missing_four_pinned(self):
        present = set(load_summary("bbx_micro"))
        expected = {
            "".join(c)
            for k in range(1, 9)
            for c in itertools.combinations("ABCDEFGH", k)
        }
        assert tuple(sorted(expected - present)) == BBX_MISSING
