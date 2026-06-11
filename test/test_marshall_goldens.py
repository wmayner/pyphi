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
import os
import re
from pathlib import Path

import pytest

from pyphi import config
from pyphi import exceptions
from pyphi.conf import presets
from pyphi.system import System
from test.test_macro_criteria import bu_substrate
from test.test_macro_criteria import min_substrate
from test.test_macro_search import dancing_couples
from test.test_macro_tpm import CG_TPM
from test.test_macro_tpm import _bbx_micro_tpm
from pyphi.substrate import Substrate

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


MICRO_SUBSTRATES = {
    "cg_micro": lambda: Substrate(CG_TPM, node_labels=tuple("ABCD")),
    "min_micro": min_substrate,
    "sfn_micro": lambda: dancing_couples(0.0),
    "sfnn_micro": lambda: dancing_couples(0.01),
    "sfs_micro": lambda: dancing_couples(0.25),
}


def _micro_cases(names):
    return [
        pytest.param(name, label, value, id=f"{name}-{label}")
        for name in names
        for label, value in load_summary(name).items()
    ]


class TestFastMicroSweeps:
    """Every committed value of the cheap micro sets, at 1e-13."""

    @pytest.mark.parametrize(
        "name,label,value",
        _micro_cases(["cg_micro", "min_micro", "sfn_micro", "sfnn_micro", "sfs_micro"]),
    )
    def test_committed_value_reproduces(self, name, label, value):
        substrate = MICRO_SUBSTRATES[name]()
        state = (0,) * substrate.size
        with config.override(**presets.iit4_2023):
            phi = System(substrate, state, indices_of(name, label)).sia().phi
        assert phi == pytest.approx(value, abs=1e-13)


class TestBuDocumentedDeviation:
    """The bu set's committed singleton zeros are stale (see the data
    README): they reproduce only under old pyphi's
    SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI default, which
    contradicts the authors' committed config and their other result
    sets. This battery pins both sides of the discrepancy so drift in
    either the upstream file or the pipeline surfaces."""

    def test_file_claims_all_small_subsystems_zero(self):
        summary = load_summary("bu_micro")
        for label, value in summary.items():
            if len(label) < 3:
                assert value == 0.0

    def test_pipeline_values_under_consistent_convention(self):
        summary = load_summary("bu_micro")
        substrate = bu_substrate()
        state = (0, 0, 0)
        with config.override(**presets.iit4_2023):
            assert System(substrate, state, (0,)).sia().phi == 1.0
            assert System(substrate, state, (1,)).sia().phi == 1.0
            with pytest.raises(exceptions.StateUnreachableForwardsError):
                System(substrate, state, (2,))
            for pair in [(0, 1), (0, 2), (1, 2)]:
                assert System(substrate, state, pair).sia().phi == pytest.approx(
                    0.0, abs=1e-13
                )
            # The full-system value is uncontested and matches the file.
            assert System(substrate, state).sia().phi == pytest.approx(
                summary["ABC"], abs=1e-13
            )


def _bbx_substrate():
    return Substrate(_bbx_micro_tpm(), node_labels=tuple("ABCDEFGH"))


def _bbx_cases(predicate):
    return [
        pytest.param(label, value, id=label)
        for label, value in load_summary("bbx_micro").items()
        if predicate(label, value)
    ]


def _assert_bbx_value(label, value):
    substrate = _bbx_substrate()
    ones = (1,) * 8
    with config.override(**presets.iit4_2023):
        phi = System(substrate, ones, indices_of("bbx_micro", label)).sia().phi
    assert phi == pytest.approx(value, abs=1e-13)


@pytest.mark.slow
class TestBbxSweepSmall:
    """Sizes 1-4: all 162 committed values (~20 s total)."""

    @pytest.mark.parametrize(
        "label,value", _bbx_cases(lambda label, value: len(label) <= 4)
    )
    def test_committed_value_reproduces(self, label, value):
        _assert_bbx_value(label, value)


@pytest.mark.slow
class TestBbxSweepLargeReducible:
    """Sizes 5-8 with committed value zero at precision (60 values).

    These short-circuit as reducible in seconds. The committed values
    include unclamped float noise (~1e-17, some negative); at 1e-13
    they are zeros.
    """

    @pytest.mark.parametrize(
        "label,value",
        _bbx_cases(lambda label, value: len(label) >= 5 and abs(value) < 1e-13),
    )
    def test_committed_value_reproduces(self, label, value):
        _assert_bbx_value(label, value)


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("PYPHI_MARSHALL_FULL_SWEEP"),
    reason="irreducible large bbx subsystems cost minutes-to-tens-of-"
    "minutes each (hours total); set PYPHI_MARSHALL_FULL_SWEEP=1 to run",
)
class TestBbxSweepLargeIrreducible:
    """Sizes 5-8 with genuinely nonzero committed values (29 values)."""

    @pytest.mark.parametrize(
        "label,value",
        _bbx_cases(lambda label, value: len(label) >= 5 and abs(value) >= 1e-13),
    )
    def test_committed_value_reproduces(self, label, value):
        _assert_bbx_value(label, value)
