"""Metamorphic relations: transformations of the input that must not change
the result.

A permutation ``p`` reorders a substrate's units (new unit ``i`` is old unit
``p[i]``); the permuted substrate is built with numpy from the original
arrays. Every result of the permuted analysis is mapped back to the original
unit indices before it is compared. Values agree within ``TOL`` relative to
``max(1, |a|, |b|)``; a selection (MIP, purview, specified state) may differ
only where either result reports a tie or a margin below ``TOL``.
"""

import itertools

import numpy as np
import pytest

import pyphi
from pyphi import actual
from pyphi.conf import config
from pyphi.conf import presets
from pyphi.direction import Direction
from pyphi.exceptions import StateUnreachableError
from pyphi.macro import SearchBounds
from pyphi.macro import complexes as grain_complexes
from pyphi.relations import NullRelations
from test.conftest import skip_if_no_emd_backend
from test.hypothesis_utils import permute_state
from test.hypothesis_utils import permute_substrate
from test.hypothesis_utils import permute_units
from test.hypothesis_utils import random_substrate

TOL = 1e-10

IIT4 = ("IIT_4_0_2026", "IIT_4_0_2023")
IIT3 = pytest.param("IIT_3_0", marks=skip_if_no_emd_backend)
SLOW_EMD = [pytest.mark.slow, skip_if_no_emd_backend]
FORMALISMS = (*IIT4, IIT3)


def close(a, b):
    if a is None or b is None:
        return a is None and b is None
    a, b = float(a), float(b)
    return abs(a - b) <= TOL * max(1.0, abs(a), abs(b))


def _tied(margin):
    return margin is not None and abs(margin) < TOL


def _pin(formalism):
    return config.override(**presets.by_name[formalism])


def _analyze(substrate, state, subset=None, **kwargs):
    return pyphi.analyze(
        substrate, state, subset=subset, parallel=False, progress=False, **kwargs
    )


# Fixtures ----------------------------------------------------------------
#
# Each case is (generator keywords, state, subset, permutation). The seeds
# give φ_s > 0 and a structure with distinctions under every formalism.

CASES = {
    "binary-52": ({"seed": 52, "n": 3}, (0, 0, 1), None, (2, 0, 1)),
    "labelled-120": ({"seed": 120, "n": 3, "labels": True}, (1, 0, 1), None, (1, 2, 0)),
    "kary-18": (
        {"seed": 18, "n": 3, "alphabet": (3, 2, 2), "labels": True},
        (1, 1, 1),
        None,
        (2, 0, 1),
    ),
    "subset-81": (
        {"seed": 81, "n": 4, "extreme": 0.3, "labels": True},
        (1, 0, 1, 0),
        (0, 1, 2),
        (3, 1, 0, 2),
    ),
}

# Wider coverage for the slow lane: more 3-unit seeds, 4-unit substrates and
# 5-unit substrates (where three-part system partitions occur).
SLOW_CASES = {
    "binary-58": ({"seed": 58, "n": 3}, (0, 1, 1), None, (1, 2, 0)),
    "binary-98": ({"seed": 98, "n": 3}, (1, 1, 1), None, (2, 1, 0)),
    "labelled-12": ({"seed": 12, "n": 3, "labels": True}, (1, 1, 1), None, (2, 0, 1)),
    "kary-6": (
        {"seed": 6, "n": 3, "alphabet": (3, 2, 2), "labels": True},
        (2, 0, 0),
        None,
        (1, 2, 0),
    ),
    "subset-22": ({"seed": 22, "n": 4}, (1, 0, 1, 0), (0, 1, 2), (2, 3, 1, 0)),
    "kary4-17": (
        {"seed": 17, "n": 4, "alphabet": (2, 3, 2, 2), "extreme": 0.3},
        (0, 0, 1, 1),
        None,
        (3, 0, 2, 1),
    ),
    "binary4-125": (
        {"seed": 125, "n": 4, "extreme": 0.3},
        (0, 1, 1, 0),
        None,
        (2, 3, 0, 1),
    ),
    "labelled4-123": (
        {"seed": 123, "n": 4, "extreme": 0.3, "labels": True},
        (0, 0, 1, 0),
        None,
        (1, 3, 0, 2),
    ),
}

FIVE_UNIT_CASES = {
    "five-512": (
        {"seed": 512, "n": 5, "p_edge": 0.5, "extreme": 0.5},
        (0, 0, 1, 1, 1),
        None,
        (3, 0, 4, 1, 2),
    ),
}


def _build(case):
    kwargs, state, subset, p = case
    substrate = random_substrate(**kwargs)
    permuted = permute_substrate(substrate, p)
    pstate = permute_state(state, p)
    psubset = None if subset is None else permute_units(subset, p)
    return substrate, state, subset, permuted, pstate, psubset, p


def _identity(n):
    return tuple(range(n))


# Canonical forms in the original unit indices ------------------------------
#
# ``p`` maps the indices a result uses to original units: the identity for an
# analysis of the original substrate, the permutation for the permuted one.


def _units(units, p):
    return frozenset(p[i] for i in units)


def _assignment(units, state, p):
    return {p[u]: int(v) for u, v in zip(units, state, strict=True)}


def _specified(m):
    spec = getattr(m, "specified_state", None)
    state = getattr(spec, "state", spec)
    return None if state is None else tuple(np.ravel(state).tolist())


def mice_form(m, p):
    state = _specified(m)
    return {
        "phi": float(m.phi),
        "purview": _units(m.purview, p),
        "state": None if state is None else _assignment(m.purview, state, p),
        "purview_tied": bool(m.effectively_tied)
        or _tied(m.purview_margin)
        or len(m.purview_ties or ()) > 1,
        "state_tied": bool(m.effectively_tied)
        or _tied(getattr(m, "state_margin", None))
        or len(getattr(m, "state_ties", None) or ()) > 1,
    }


def sia_form(sia, p):
    partition = sia.partition
    out = {
        "phi": float(sia.phi),
        "edges": None
        if partition is None
        else frozenset((p[a], p[b]) for a, b in partition.removed_edges()),
    }
    system_state = getattr(sia, "system_state", None)
    if system_state is None:
        runner_up = getattr(sia, "runner_up", None)
        out["tied"] = len(getattr(sia, "ties", None) or ()) > 1 or (
            runner_up is not None and _tied(runner_up.phi - sia.phi)
        )
        return out
    out["tied"] = (
        bool(sia.effectively_tied)
        or bool(sia.tied_selections)
        or _tied(sia.partition_margin)
    )
    for direction in ("cause", "effect"):
        spec = getattr(system_state, direction)
        out[direction] = _assignment(spec.purview, tuple(spec.state), p)
        out[direction + "_ii"] = float(spec.intrinsic_specification)
        out[direction + "_tied"] = _tied(spec.state_margin) or len(spec.ties or ()) > 1
    return out


def ces_form(ces, p):
    out = {
        "distinctions": {
            _units(d.mechanism, p): {
                "phi": float(d.phi),
                "cause": mice_form(d.cause, p),
                "effect": mice_form(d.effect, p),
            }
            for d in ces.distinctions
        },
        "sum_phi_distinctions": float(ces.sum_phi_distinctions),
    }
    relations = getattr(ces, "relations", None)
    if relations is not None and not isinstance(relations, NullRelations):
        concrete = (
            relations.materialize() if hasattr(relations, "materialize") else relations
        )
        out["big_phi"] = float(ces.big_phi)
        out["relations"] = {
            frozenset(_units(d.mechanism, p) for d in r): float(r.phi) for r in concrete
        }
    return out


def analysis_form(analysis, p):
    return {"sia": sia_form(analysis.sia, p), "ces": ces_form(analysis.ces, p)}


def _system_state_tied(sia):
    form = sia_form(sia, _identity(max(sia.node_indices) + 1))
    return form["tied"] or form.get("cause_tied") or form.get("effect_tied")


def assert_same_sia(a, b):
    assert close(a["phi"], b["phi"]), (a["phi"], b["phi"])
    if a["phi"] > TOL and not (a["tied"] or b["tied"]):
        assert a["edges"] == b["edges"], (sorted(a["edges"]), sorted(b["edges"]))
    state_tied = False
    for direction in ("cause", "effect"):
        if direction not in a:
            continue
        assert close(a[direction + "_ii"], b[direction + "_ii"]), direction
        tied = a[direction + "_tied"] or b[direction + "_tied"]
        state_tied |= tied
        if not tied:
            assert a[direction] == b[direction], (direction, a[direction], b[direction])
    return state_tied


def assert_same_ces(a, b):
    """Compare two canonical structures; return True if no tie hid a check."""
    da, db = a["distinctions"], b["distinctions"]
    assert set(da) == set(db), (sorted(map(sorted, da)), sorted(map(sorted, db)))
    complete = True
    for mechanism in da:
        x, y = da[mechanism], db[mechanism]
        assert close(x["phi"], y["phi"]), (sorted(mechanism), x["phi"], y["phi"])
        for direction in ("cause", "effect"):
            u, v = x[direction], y[direction]
            assert close(u["phi"], v["phi"]), (sorted(mechanism), direction)
            if u["purview_tied"] or v["purview_tied"]:
                complete = False
                continue
            assert u["purview"] == v["purview"], (sorted(mechanism), direction)
            if u["state_tied"] or v["state_tied"]:
                complete = False
            else:
                assert u["state"] == v["state"], (sorted(mechanism), direction)
    assert close(a["sum_phi_distinctions"], b["sum_phi_distinctions"])
    if "big_phi" in a and complete:
        assert close(a["big_phi"], b["big_phi"]), (a["big_phi"], b["big_phi"])
        assert set(a["relations"]) == set(b["relations"])
        for relata, phi in a["relations"].items():
            assert close(phi, b["relations"][relata])
    return complete


def assert_same_analysis(a, b):
    if assert_same_sia(a["sia"], b["sia"]):
        # The distinctions depend on the tied system state.
        return False
    return assert_same_ces(a["ces"], b["ces"])


def assert_nontrivial(form):
    assert form["sia"]["phi"] > TOL
    assert form["ces"]["distinctions"]


# Unit permutation of a whole analysis ---------------------------------------


def _check_analysis_permutation(case, formalism):
    substrate, state, subset, permuted, pstate, psubset, p = _build(case)
    with _pin(formalism):
        base = analysis_form(
            _analyze(substrate, state, subset), _identity(substrate.size)
        )
        other = analysis_form(_analyze(permuted, pstate, psubset), p)
    assert_nontrivial(base)
    complete = assert_same_analysis(base, other)
    if formalism != "IIT_3_0":
        # No tie hid the comparison of relations and Φ.
        assert complete and base["ces"]["relations"]


@pytest.mark.parametrize("formalism", FORMALISMS)
@pytest.mark.parametrize("name", list(CASES))
def test_analysis_is_invariant_under_unit_permutation(name, formalism):
    """φ_s, the MIP, the system state, every distinction, every relation and
    Φ map onto the permuted substrate's analysis."""
    _check_analysis_permutation(CASES[name], formalism)


@pytest.mark.slow
@pytest.mark.parametrize("formalism", FORMALISMS)
@pytest.mark.parametrize("name", list(SLOW_CASES))
def test_analysis_is_invariant_under_unit_permutation_wide(name, formalism):
    """The permutation relation on more seeds, 4-unit and k-ary substrates."""
    _check_analysis_permutation(SLOW_CASES[name], formalism)


@pytest.mark.slow
@pytest.mark.parametrize("name", list(FIVE_UNIT_CASES))
def test_five_unit_analysis_is_invariant_under_unit_permutation(name):
    """At five units, where three-part system partitions occur."""
    _check_analysis_permutation(FIVE_UNIT_CASES[name], "IIT_4_0_2026")


# Unit permutation of the System query methods -------------------------------


def _nonempty_subsets(units):
    return [
        c for r in range(1, len(units) + 1) for c in itertools.combinations(units, r)
    ]


def _partition_form(partition, p):
    if partition is None:
        return None
    return frozenset((_units(x.mechanism, p), _units(x.purview, p)) for x in partition)


def _ria_form(ria, p):
    state = _specified(ria)
    return {
        "phi": float(ria.phi),
        "partition": _partition_form(ria.partition, p),
        "state": None if state is None else _assignment(ria.purview, state, p),
        "partition_tied": bool(ria.effectively_tied)
        or _tied(ria.partition_margin)
        or len(ria.partition_ties or ()) > 1,
        "state_tied": bool(ria.effectively_tied)
        or _tied(ria.state_margin)
        or len(ria.state_ties or ()) > 1,
    }


QUERY_CASES = [
    pytest.param("IIT_4_0_2026", "binary-52"),
    pytest.param("IIT_4_0_2026", "kary-18"),
    pytest.param("IIT_4_0_2023", "subset-81"),
    pytest.param("IIT_3_0", "binary-52", marks=skip_if_no_emd_backend),
    pytest.param("IIT_4_0_2026", "binary4-125", marks=pytest.mark.slow),
    pytest.param("IIT_3_0", "binary4-125", marks=SLOW_EMD),
]


@pytest.mark.parametrize(("formalism", "name"), QUERY_CASES)
def test_system_queries_are_invariant_under_unit_permutation(formalism, name):
    """Repertoires (with their axes permuted), find_mip, phi,
    intrinsic_specification, mic, mie and distinction map onto the permuted
    System's answers for every mechanism and purview."""
    substrate, state, subset, permuted, pstate, psubset, p = _build(
        {**CASES, **SLOW_CASES}[name]
    )
    inverse = tuple(np.argsort(p))
    iit3 = formalism == "IIT_3_0"
    identity = _identity(substrate.size)
    nontrivial = 0
    with _pin(formalism):
        measure = config.formalism.iit.specification_measure
        a = pyphi.System(substrate, state, subset=subset)
        b = pyphi.System(permuted, pstate, subset=psubset)
        units = a.node_indices
        for mechanism in _nonempty_subsets(units):
            pmechanism = permute_units(mechanism, p)
            for purview in _nonempty_subsets(units):
                ppurview = permute_units(purview, p)
                for fa, fb in (
                    (a.cause_repertoire, b.cause_repertoire),
                    (a.effect_repertoire, b.effect_repertoire),
                ):
                    ra = np.asarray(fa(mechanism, purview))
                    rb = np.transpose(np.asarray(fb(pmechanism, ppurview)), inverse)
                    assert ra.shape == rb.shape
                    np.testing.assert_allclose(ra, rb, rtol=0, atol=1e-12)
                for direction in (Direction.CAUSE, Direction.EFFECT):
                    x = _ria_form(a.find_mip(direction, mechanism, purview), identity)
                    y = _ria_form(b.find_mip(direction, pmechanism, ppurview), p)
                    where = (str(direction), mechanism, purview)
                    assert close(x["phi"], y["phi"]), where
                    nontrivial += x["phi"] > TOL
                    if x["phi"] > TOL and not (
                        x["partition_tied"] or y["partition_tied"]
                    ):
                        assert x["partition"] == y["partition"], where
                    if not (x["state_tied"] or y["state_tied"]):
                        assert x["state"] == y["state"], where
                    if iit3:
                        continue
                    sa = a.intrinsic_specification(
                        direction, mechanism, purview, specification_measure=measure
                    )
                    sb = b.intrinsic_specification(
                        direction, pmechanism, ppurview, specification_measure=measure
                    )
                    assert close(
                        sa.intrinsic_specification, sb.intrinsic_specification
                    ), where
                    if not (
                        _tied(sa.state_margin)
                        or _tied(sb.state_margin)
                        or len(sa.ties or ()) > 1
                    ):
                        assert _assignment(
                            sa.purview, tuple(sa.state), identity
                        ) == _assignment(sb.purview, tuple(sb.state), p), where
                assert close(a.phi(mechanism, purview), b.phi(pmechanism, ppurview)), (
                    mechanism,
                    purview,
                )
            for query in ("mic", "mie"):
                x = mice_form(getattr(a, query)(mechanism), identity)
                y = mice_form(getattr(b, query)(pmechanism), p)
                assert close(x["phi"], y["phi"]), (query, mechanism)
                if not (x["purview_tied"] or y["purview_tied"]):
                    assert x["purview"] == y["purview"], (query, mechanism)
                    if not (x["state_tied"] or y["state_tied"]):
                        assert x["state"] == y["state"], (query, mechanism)
            assert close(a.distinction(mechanism).phi, b.distinction(pmechanism).phi)
    assert nontrivial > 0


# Unit permutation of complexes and sweeps ------------------------------------

COMPLEX_CASES = [
    pytest.param("IIT_4_0_2026", "binary-52"),
    pytest.param("IIT_4_0_2026", "kary-18"),
    pytest.param("IIT_3_0", "labelled-120", marks=skip_if_no_emd_backend),
]


def _back(units, p):
    return tuple(sorted(p[i] for i in units))


@pytest.mark.parametrize(("formalism", "name"), COMPLEX_CASES)
def test_complexes_and_sweeps_are_invariant_under_unit_permutation(formalism, name):
    """all_sias, complexes and a sweep over every subset map onto the
    permuted substrate's."""
    substrate, state, _, permuted, pstate, _, p = _build(CASES[name])
    identity = _identity(substrate.size)
    with _pin(formalism):
        phis = {
            _back(s.node_indices, identity): s.phi for s in substrate.all_sias(state)
        }
        pphis = {_back(s.node_indices, p): s.phi for s in permuted.all_sias(pstate)}
        found = [
            (_back(c.node_indices, identity), c) for c in substrate.complexes(state)
        ]
        pfound = [(_back(c.node_indices, p), c) for c in permuted.complexes(pstate)]
        sweeps = [
            pyphi.sweep(
                s, states=[st], subsets="all", formalisms=[formalism], parallel=False
            )
            for s, st in ((substrate, state), (permuted, pstate))
        ]
    assert any(phi > TOL for phi in phis.values())
    assert found
    assert set(phis) == set(pphis)
    for units, phi in phis.items():
        assert close(phi, pphis[units]), units
    assert not any(c.effectively_tied for _, c in found)
    assert [u for u, _ in found] == [u for u, _ in pfound]
    for (_, c), (_, d) in zip(found, pfound, strict=True):
        assert close(c.phi, d.phi)
        assert close(c.exclusion_margin, d.exclusion_margin)
    rows = {tuple(r.subset): r.phi for r in sweeps[0].df.itertuples()}
    prows = {_back(r.subset, p): r.phi for r in sweeps[1].df.itertuples()}
    assert set(rows) == set(prows)
    for units, phi in rows.items():
        assert close(phi, prows[units]), units
    assert {_back(s[2], p) for s in sweeps[1].skipped} == {
        tuple(s[2]) for s in sweeps[0].skipped
    }


@pytest.mark.parametrize("compute", ["sia", "ces"])
def test_sweep_over_all_states_is_invariant_under_unit_permutation(compute):
    """A sweep over every state and subset maps onto the permuted substrate's,
    row for row, along with its list of skipped (unreachable) rows."""
    substrate, _, _, permuted, _, _, p = _build(CASES["binary-52"])
    columns = (
        ["phi"] if compute == "sia" else ["phi", "big_phi", "sum_phi_d", "sum_phi_r"]
    )

    def back_state(pstate):
        out = [None] * len(p)
        for new, old in enumerate(p):
            out[old] = pstate[new]
        return tuple(out)

    with _pin("IIT_4_0_2026"):
        kwargs = {
            "states": "all",
            "subsets": "all",
            "formalisms": ["IIT_4_0_2026"],
            "compute": compute,
            "parallel": False,
        }
        a = pyphi.sweep(substrate, **kwargs)
        b = pyphi.sweep(permuted, **kwargs)
    rows = {(tuple(r["state"]), tuple(r["subset"])): r for _, r in a.df.iterrows()}
    prows = {
        (back_state(tuple(r["state"])), _back(r["subset"], p)): r
        for _, r in b.df.iterrows()
    }
    assert set(rows) == set(prows)
    assert sum(r["phi"] > TOL for r in rows.values()) > 0
    # Φ depends on the tie-break when the system state is tied.
    tied = {
        key
        for key, result in zip(rows, a.results, strict=True)
        if compute == "ces" and _system_state_tied(result.sia)
    }
    assert len(tied) < len(rows)
    for key, row in rows.items():
        for column in ["phi"] if key in tied else columns:
            assert close(row[column], prows[key][column]), (key, column)
    assert {(tuple(s[3]), tuple(s[2])) for s in a.skipped} == {
        (back_state(tuple(s[3])), _back(s[2], p)) for s in b.skipped
    }


# Complexes against a brute-force exclusion -----------------------------------


def _phi_of_every_subset(substrate, state):
    out = {}
    for k in range(1, substrate.size + 1):
        for subset in itertools.combinations(range(substrate.size), k):
            try:
                out[subset] = float(
                    _analyze(substrate, state, subset, compute="sia").phi
                )
            except StateUnreachableError:
                out[subset] = 0.0
    return out


def _brute_force_complexes(phis):
    """Accept the candidate with the highest φ_s, drop every candidate that
    overlaps it, and repeat. Returns None when the top tier holds two
    overlapping candidates within TOL, which the tie rules resolve."""
    remaining = {s: phi for s, phi in phis.items() if phi > TOL}
    accepted = []
    while remaining:
        best = max(remaining.values())
        top = [s for s, phi in remaining.items() if phi >= best - TOL * max(1, best)]
        if any(set(a) & set(b) for a, b in itertools.combinations(top, 2)):
            return None
        accepted += [(s, remaining[s]) for s in top]
        covered = set().union(*map(set, top))
        remaining = {s: phi for s, phi in remaining.items() if not set(s) & covered}
    return accepted


BRUTE_FORCE_CASES = [
    pytest.param("IIT_4_0_2026", {"seed": 52, "n": 3}, (0, 0, 1)),
    pytest.param("IIT_4_0_2026", {"seed": 18, "n": 3, "alphabet": (3, 2, 2)}, (1, 1, 1)),
    pytest.param("IIT_4_0_2026", {"seed": 81, "n": 4, "extreme": 0.3}, (1, 0, 1, 0)),
    pytest.param(
        "IIT_3_0", {"seed": 52, "n": 3}, (0, 0, 1), marks=skip_if_no_emd_backend
    ),
    pytest.param(
        "IIT_3_0", {"seed": 120, "n": 3}, (1, 0, 1), marks=skip_if_no_emd_backend
    ),
    pytest.param(
        "IIT_4_0_2026",
        {"seed": 125, "n": 4, "extreme": 0.3},
        (0, 1, 1, 0),
        marks=pytest.mark.slow,
    ),
    pytest.param(
        "IIT_3_0", {"seed": 125, "n": 4, "extreme": 0.3}, (0, 1, 1, 0), marks=SLOW_EMD
    ),
    pytest.param(
        "IIT_3_0", {"seed": 123, "n": 4, "extreme": 0.3}, (0, 0, 1, 0), marks=SLOW_EMD
    ),
]


@pytest.mark.parametrize(("formalism", "kwargs", "state"), BRUTE_FORCE_CASES)
def test_complexes_match_a_brute_force_exclusion(formalism, kwargs, state):
    """Substrate.complexes, all_sias and maximal_complex agree with analyzing
    every subset and applying exclusion greedily by φ_s."""
    substrate = random_substrate(**kwargs)
    with _pin(formalism):
        phis = _phi_of_every_subset(substrate, state)
        found = substrate.complexes(state)
        sias = substrate.all_sias(state)
        maximal = substrate.maximal_complex(state)
    expected = _brute_force_complexes(phis)
    assert expected, "the fixture needs an untied, non-empty set of complexes"
    for sia in sias:
        units = tuple(sorted(sia.node_indices))
        assert close(sia.phi, phis[units]), units
    listed = {tuple(sorted(s.node_indices)) for s in sias}
    assert all(phi == 0 for s, phi in phis.items() if s not in listed)
    assert sorted(tuple(sorted(c.node_indices)) for c in found) == sorted(
        u for u, _ in expected
    )
    for c in found:
        assert close(c.phi, dict(expected)[tuple(sorted(c.node_indices))])
    assert [c.phi for c in found] == sorted((c.phi for c in found), reverse=True)
    assert tuple(sorted(maximal.node_indices)) == tuple(sorted(found[0].node_indices))
    assert found[0].is_maximal and not any(c.is_maximal for c in found[1:])


# Background units under IIT 3.0 ---------------------------------------------


def _conditioned_substrate(substrate, subset, state):
    """The subset's units alone, each unit's TPM conditioned on the
    background units at their current state."""
    tpm = substrate.factored_tpm
    background = [i for i in range(substrate.size) if i not in subset]
    factors = []
    for j in subset:
        factor = np.asarray(tpm.factor(j))
        index = tuple(
            (state[i] if factor.shape[i] > 1 else 0) if i in background else slice(None)
            for i in range(substrate.size)
        )
        shape = [tpm.alphabet_sizes[i] for i in subset] + [factor.shape[-1]]
        factors.append(np.broadcast_to(factor[index], shape).copy())
    return pyphi.Substrate(
        marginals=factors,
        state_space=tuple(tuple(tpm.state_space[i]) for i in subset),
        cm=np.asarray(substrate.cm)[np.ix_(subset, subset)],
    )


BACKGROUND_CASES = [
    ({"seed": 22, "n": 4}, (1, 0, 1, 0), (0, 1, 2)),
    ({"seed": 52, "n": 3}, (0, 0, 1), (0, 2)),
]


@skip_if_no_emd_backend
@pytest.mark.parametrize(("kwargs", "state", "subset"), BACKGROUND_CASES)
def test_iit3_background_is_held_at_its_current_state(kwargs, state, subset):
    """Under IIT 3.0 a subset System's analysis equals the same subset of
    Substrate.inactivate(background state), and the analysis of a substrate
    of the subset's units whose TPM is conditioned on the background state."""
    substrate = random_substrate(**kwargs)
    background = {i: state[i] for i in range(substrate.size) if i not in subset}
    identity = _identity(substrate.size)
    with _pin("IIT_3_0"):
        reference = analysis_form(_analyze(substrate, state, subset), identity)
        inactivated = analysis_form(
            _analyze(substrate.inactivate(background), state, subset), identity
        )
        conditioned = analysis_form(
            _analyze(
                _conditioned_substrate(substrate, subset, state),
                tuple(state[i] for i in subset),
            ),
            subset,
        )
    assert_nontrivial(reference)
    assert_same_analysis(reference, inactivated)
    assert_same_analysis(reference, conditioned)


def test_iit4_background_is_not_simply_conditioned():
    """Control for the IIT 3.0 relation: under IIT 4.0 the background's past
    states are weighted by their posterior, so conditioning the TPM on the
    current background state gives a different analysis."""
    kwargs, state, subset = BACKGROUND_CASES[0]
    substrate = random_substrate(**kwargs)
    with _pin("IIT_4_0_2026"):
        a = _analyze(substrate, state, subset)
        b = _analyze(
            _conditioned_substrate(substrate, subset, state),
            tuple(state[i] for i in subset),
        )
    assert a.phi > TOL
    assert not close(a.phi, b.phi) or not close(
        a.ces.sum_phi_distinctions, b.ces.sum_phi_distinctions
    )


# Unit permutation of the grain search ----------------------------------------


def _unit_key(unit, p):
    """A macro unit as (original constituents, grain, truth table over the
    sorted original constituents, background apportionment)."""
    constituents = unit.micro_constituents
    grain = unit.micro_grain
    n = len(constituents)
    old = [p[c] for c in constituents]
    order = sorted(old)
    position = [order.index(o) for o in old]
    table = []
    for index in range(2 ** (n * grain)):
        digits = [(index >> k) & 1 for k in range(n * grain)]
        native = []
        for t in range(grain):
            step = digits[t * n : (t + 1) * n]
            native.extend(step[position[i]] for i in range(n))
        table.append(unit.micro_mapping[sum(d << k for k, d in enumerate(native))])
    apportionment = tuple(sorted(p[w] for w in unit.background_apportionment))
    return (tuple(order), grain, tuple(table), apportionment)


def _units_key(units, p):
    return frozenset(_unit_key(u, p) for u in units or ())


def _grain_form(result, p):
    out = {"complexes": {}, "records": {}, "ties": set()}
    for c in result.complexes:
        out["complexes"][_back(c.node_indices, p)] = {
            "phi": float(c.phi),
            "units": _units_key(c.units, p),
            "margin": c.exclusion_margin,
            "excluded": {
                (_back(e.node_indices, p), _units_key(e.units, p)): (
                    None if e.phi is None else float(e.phi),
                    bool(e.gated),
                )
                for e in c.excluded
            },
        }
    for r in result.records:
        out["records"][_units_key(r.system.units, p)] = (r.phi, r.ii_ceiling, r.gated)
    for clique in result.ties:
        out["ties"].add(frozenset(_units_key(s.units, p) for s in clique))
    return out


GRAIN_CASES = [
    pytest.param({"seed": 52, "n": 3}, ((0, 0, 1),), (2, 0, 1), {}, id="seed52"),
    pytest.param(
        {"seed": 23, "n": 3, "extreme": 0.3}, ((1, 0, 1),), (1, 2, 0), {}, id="seed23"
    ),
    pytest.param(
        {"seed": 52, "n": 3},
        ((1, 1, 0), (0, 0, 1)),
        (2, 0, 1),
        {"max_constituents": 2, "max_update_grain": 2},
        marks=pytest.mark.slow,
        id="seed52-grain2",
    ),
    pytest.param(
        {"seed": 58, "n": 3},
        ((0, 1, 1),),
        (1, 2, 0),
        {"max_constituents": 2, "apportionment": "ENUMERATE", "max_background": 1},
        marks=pytest.mark.slow,
        id="seed58-apportioned",
    ),
]


@pytest.mark.parametrize("prune", ["off", "certified"])
@pytest.mark.parametrize(("kwargs", "history", "p", "bounds"), GRAIN_CASES)
def test_grain_search_is_invariant_under_unit_permutation(
    kwargs, history, p, bounds, prune
):
    """The grain search's complexes (units, φ_s, exclusion margins, excluded
    rivals), its evaluated records and its ties map onto the permuted
    substrate's."""
    substrate = random_substrate(**kwargs)
    with _pin("IIT_4_0_2026"):
        a = grain_complexes(
            substrate,
            history,
            bounds=SearchBounds(**bounds),
            prune=prune,
            progress=False,
        )
        b = grain_complexes(
            permute_substrate(substrate, p),
            tuple(permute_state(state, p) for state in history),
            bounds=SearchBounds(**bounds),
            prune=prune,
            progress=False,
        )
    if len(history) > 1:
        assert any(u.micro_grain > 1 for r in a.records for u in r.system.units)
    if "apportionment" in bounds:
        assert any(u.background_apportionment for r in a.records for u in r.system.units)
    x, y = _grain_form(a, _identity(substrate.size)), _grain_form(b, p)
    assert x["complexes"] and any(c["phi"] > TOL for c in x["complexes"].values())
    assert set(x["complexes"]) == set(y["complexes"])
    for footprint, c in x["complexes"].items():
        d = y["complexes"][footprint]
        assert close(c["phi"], d["phi"]), footprint
        assert c["units"] == d["units"], footprint
        assert close(c["margin"], d["margin"]), footprint
        assert set(c["excluded"]) == set(d["excluded"]), footprint
        for key, (phi, gated) in c["excluded"].items():
            assert close(phi, d["excluded"][key][0]) and gated == d["excluded"][key][1]
    assert x["ties"] == y["ties"]
    assert set(x["records"]) == set(y["records"])
    for key, (phi, ceiling, gated) in x["records"].items():
        other = y["records"][key]
        assert close(phi, other[0]) and close(ceiling, other[1]) and gated == other[2]


# Unit permutation of actual causation ---------------------------------------


def _transition_case(seed, n, alphabet=None):
    rng = np.random.default_rng(seed)
    substrate = random_substrate(seed, n, alphabet=alphabet, extreme=0.5)
    tpm = substrate.factored_tpm
    before = tuple(int(rng.integers(k)) for k in tpm.alphabet_sizes)
    # Each unit's most likely next state, so the transition is realizable.
    after = tuple(int(np.argmax(np.asarray(tpm.factor(j))[before])) for j in range(n))
    return substrate, before, after


def _links(account, p):
    out = {}
    for link in account:
        key = (str(link.direction), _back(link.mechanism, p))
        tied = len(link.purview_ties or ()) > 1
        out[key] = (float(link.alpha), _back(link.purview, p), tied)
    return out


AC_CASES = [
    (102, 3, None, (2, 0, 1)),
    (109, 3, None, (1, 2, 0)),
    (312, 3, (3, 2, 2), (2, 0, 1)),
]


@pytest.mark.parametrize(("seed", "n", "alphabet", "p"), AC_CASES)
def test_actual_causation_is_invariant_under_unit_permutation(seed, n, alphabet, p):
    """α for every mechanism's actual cause and effect, the account's links
    and the transition's α map onto the permuted transition's."""
    substrate, before, after = _transition_case(seed, n, alphabet)
    units = tuple(range(n))
    with _pin("IIT_4_0_2026"):
        t = actual.Transition(substrate, before, after, units, units)
        u = actual.Transition(
            permute_substrate(substrate, p),
            permute_state(before, p),
            permute_state(after, p),
            units,
            units,
        )
        a, b = _links(actual.account(t), units), _links(actual.account(u), p)
        for k in range(1, n + 1):
            for mechanism in itertools.combinations(units, k):
                for find in ("find_actual_cause", "find_actual_effect"):
                    x = getattr(t, find)(mechanism)
                    y = getattr(u, find)(permute_units(mechanism, p))
                    assert close(x.alpha, y.alpha), (find, mechanism)
        sa = actual.sia(t, parallel=False, progress=False)
        sb = actual.sia(u, parallel=False, progress=False)
    assert a and sa.alpha > TOL
    assert set(a) == set(b)
    for key, (alpha, purview, tied) in a.items():
        assert close(alpha, b[key][0]), key
        if not (tied or b[key][2]):
            assert purview == b[key][1], key
    assert close(sa.alpha, sb.alpha)
