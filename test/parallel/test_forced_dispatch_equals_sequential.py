"""Parallel evaluation equals sequential evaluation with dispatch forced.

Thresholds are lowered so that work really goes to workers, on the loky
process backend and the thread backend, for surfaces beyond the SIA covered by
``test_parallel_equals_sequential.py``: the 2026 cause-effect structure with
relations, ``Substrate.complexes``, k-ary substrates and actual causation.
"""

from __future__ import annotations

import numpy as np
import pytest

import pyphi
from pyphi import actual
from pyphi.conf import config
from pyphi.conf import presets
from pyphi.parallel.backends import local_process
from pyphi.parallel.backends import local_thread
from test.conftest import skip_if_no_emd_backend
from test.hypothesis_utils import random_substrate
from test.integration.test_metamorphic import TOL
from test.integration.test_metamorphic import analysis_form
from test.integration.test_metamorphic import assert_same_analysis
from test.integration.test_metamorphic import close

BACKENDS = ["local", "thread"]

ALL_LEVELS = (
    "parallel_complex_evaluation",
    "parallel_partition_evaluation",
    "parallel_distinction_evaluation",
    "parallel_purview_evaluation",
    "parallel_mechanism_partition_evaluation",
    "parallel_relation_evaluation",
)


def _forced(backend: str, levels=ALL_LEVELS, threshold: int = 2) -> dict:
    """Config override that sends every level in ``levels`` to ``backend``
    with two workers, one item per chunk and a sequential threshold low
    enough that a level with two or more items dispatches."""
    c = config.infrastructure
    forced = {"parallel": True, "sequential_threshold": threshold, "chunksize": 1}
    return {
        "parallel": True,
        "parallel_backend": backend,
        "parallel_workers": 2,
        **{k: {**getattr(c, k), **forced} for k in levels},
    }


CES_CASES = {
    "binary-52": ({"seed": 52, "n": 3}, (0, 0, 1)),
    "kary-18": ({"seed": 18, "n": 3, "alphabet": (3, 2, 2)}, (1, 1, 1)),
}


def _distinction_order(analysis):
    return [
        (tuple(d.mechanism), tuple(d.cause.purview), tuple(d.effect.purview))
        for d in analysis.ces.distinctions
    ]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("relations", ["CONCRETE", "ANALYTICAL"])
@pytest.mark.parametrize("name", list(CES_CASES))
def test_2026_ces_with_relations(name, relations, backend):
    """The 2026 analysis (SIA, distinctions with their specified states,
    relations and Φ) is the same with every level dispatched to workers."""
    kwargs, state = CES_CASES[name]
    substrate = random_substrate(**kwargs)
    identity = tuple(range(substrate.size))
    with config.override(**presets.iit4_2026, relation_computation=relations):
        seq = pyphi.analyze(substrate, state, parallel=False, progress=False)
        with config.override(**_forced(backend)):
            par = pyphi.analyze(substrate, state, progress=False)
    assert seq.phi > TOL and seq.big_phi > TOL
    assert _distinction_order(seq) == _distinction_order(par)
    assert_same_analysis(analysis_form(seq, identity), analysis_form(par, identity))


COMPLEX_CASES = [
    pytest.param("IIT_4_0_2026", {"seed": 81, "n": 4, "extreme": 0.3}, (1, 0, 1, 0)),
    pytest.param(
        "IIT_3_0",
        {"seed": 52, "n": 3},
        (0, 0, 1),
        marks=skip_if_no_emd_backend,
    ),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(("formalism", "kwargs", "state"), COMPLEX_CASES)
def test_complexes(formalism, kwargs, state, backend):
    """Substrate.complexes gives the same complexes, in the same order, with
    the candidate systems dispatched to workers."""
    substrate = random_substrate(**kwargs)
    with config.override(**presets.by_name[formalism]):
        seq = substrate.complexes(state, parallel=False, progress=False)
        with config.override(
            **_forced(backend, levels=("parallel_complex_evaluation",))
        ):
            par = substrate.complexes(state, parallel=True, progress=False)
    assert seq and seq[0].phi > TOL
    assert [c.node_indices for c in seq] == [c.node_indices for c in par]
    for a, b in zip(seq, par, strict=True):
        assert close(a.phi, b.phi)
        assert close(a.exclusion_margin, b.exclusion_margin)


def _transition(seed, alphabet=None):
    rng = np.random.default_rng(seed)
    substrate = random_substrate(seed, 3, alphabet=alphabet, extreme=0.5)
    tpm = substrate.factored_tpm
    before = tuple(int(rng.integers(k)) for k in tpm.alphabet_sizes)
    after = tuple(int(np.argmax(np.asarray(tpm.factor(j))[before])) for j in range(3))
    return actual.Transition(substrate, before, after, (0, 1, 2), (0, 1, 2))


def _account_form(account):
    return sorted(
        (str(link.direction), tuple(link.mechanism), tuple(link.purview), link.alpha)
        for link in account
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    ("seed", "alphabet"),
    [(102, None), pytest.param(312, (3, 2, 2), marks=pytest.mark.slow)],
)
def test_actual_causation(seed, alphabet, backend):
    """The transition's α, its partition and the account's links are the same
    with dispatch forced."""
    with config.override(**presets.iit4_2026):
        t = _transition(seed, alphabet)
        seq = actual.sia(t, parallel=False, progress=False)
        seq_account = _account_form(actual.account(t))
        with config.override(**_forced(backend)):
            par = actual.sia(t, parallel=True, progress=False)
            par_account = _account_form(actual.account(t))
    assert seq.alpha > TOL and seq_account
    assert close(seq.alpha, par.alpha)
    assert seq.partition == par.partition or len(seq.ties or ()) > 1
    assert [x[:3] for x in seq_account] == [x[:3] for x in par_account]
    for x, y in zip(seq_account, par_account, strict=True):
        assert close(x[3], y[3])


@pytest.mark.parametrize("backend", BACKENDS)
def test_forced_configuration_dispatches_to_workers(backend, monkeypatch):
    """The forced configuration used above really sends work to workers, for
    the CES, for complexes and for actual causation."""
    dispatched = []
    if backend == "local":
        run_parallel = local_process.LocalMapReduce._run_parallel

        def counting_run_parallel(self):
            dispatched.append(self.total)
            return run_parallel(self)

        monkeypatch.setattr(
            local_process.LocalMapReduce, "_run_parallel", counting_run_parallel
        )
    else:
        map_reduce = local_thread.LocalThreadScheduler._map_reduce

        def counting_map_reduce(self, fn, items, *more, chunking, **kwargs):
            items = list(items)
            if len(items) >= chunking.sequential_threshold:
                dispatched.append(len(items))
            return map_reduce(self, fn, items, *more, chunking=chunking, **kwargs)

        monkeypatch.setattr(
            local_thread.LocalThreadScheduler, "_map_reduce", counting_map_reduce
        )

    kwargs, state = CES_CASES["binary-52"]
    substrate = random_substrate(**kwargs)
    counts = {}
    with config.override(**presets.iit4_2026, relation_computation="CONCRETE"):
        with config.override(**_forced(backend)):
            pyphi.analyze(substrate, state, progress=False)
        counts["ces"] = len(dispatched)
        with config.override(
            **_forced(backend, levels=("parallel_complex_evaluation",))
        ):
            substrate.complexes(state, parallel=True, progress=False)
        counts["complexes"] = len(dispatched) - counts["ces"]
        with config.override(**_forced(backend)):
            actual.sia(_transition(102), parallel=True, progress=False)
        counts["actual"] = len(dispatched) - counts["ces"] - counts["complexes"]
    assert all(count > 0 for count in counts.values()), counts
