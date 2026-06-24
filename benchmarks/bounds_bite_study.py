"""P13 SP2 — bite-rate study for the Zaeemzadeh (2024) certified bounds.

Question: would the theorem-certified upper bounds (the same ones wired as
runtime assertions in B1, ``pyphi/formalism/iit4/bounds.py``) let the search
*skip* a full evaluation often enough to be worth building search-integration
pruning? The roadmap gates that pruning on this measurement: build it (behind a
byte-identical shadow gate) only if the bounds bite enough; otherwise the
bounds module is the final P13 deliverable.

Two skip opportunities are measured, each in the order that maximizes pruning
(descending ceiling):

  A. ``complexes()`` / ``major_complex``: a candidate system whose size-only
     ceiling ``n(n-1)`` is below the running-best ``phi_s`` is provably
     non-maximal. Certified only for ``n >= 2`` (the bound does not cover the
     single-node self-loop-phi convention, where ``phi_s`` can be > 0 = 0).

  B. MICE search (``find_mice``): a purview whose Theorem-1 cap ``|M||Z|`` is
     below the running-best small-phi cannot win, so its partition search is
     skippable.

Run: ``uv run python benchmarks/bounds_bite_study.py``

------------------------------------------------------------------------------
Result (2026-06-13, IIT 4.0 2023 + GID, binary substrates n <= 4):

  substrate  n  n>=2 cands  max phi_s   A: valid bite   B: MICE |M||Z| bite
  basic      3  4           2.0         0/4  = 0.0       0/30  = 0.0
  xor        3  4           1.5         0/4  = 0.0       0/50  = 0.0
  rule110    3  3           1.0         0/3  = 0.0       0/98  = 0.0
  grid3      3  4           0.814       0/4  = 0.0       0/68  = 0.0
  disj_conj  4  (reducible: no positive-phi candidates)  0/14  = 0.0

CONCLUSION: the bounds do not bite in their certified domain — 0% useful prune
rate. The bounds are loose relative to realized phi: ``|M||Z|`` (in [1, 9] for
n <= 4) dwarfs actual small-phi (< 1), and ``n(n-1)`` is only valid for n >= 2,
where the smallest ceiling (size-2 -> 2) already exceeds the observed max
``phi_s`` (<= 2.0). Pruning a size-2 candidate would require a sibling with
``phi_s > 2`` (size-3: > 6); since ``phi_s`` is in practice far below the
``n(n-1)`` ceiling, the threshold is essentially never met, and this does not
improve with n. Per the roadmap gate the search-integration pruning is NOT
built; the bounds module + the B1 runtime assertions are the final P13
deliverable.

(The study also independently re-confirms B1's domain gating: the only
candidates whose ``phi_s`` exceeds ``n(n-1)`` are single nodes with self-loops
-- rule110 and grid3 above -- exactly the out-of-domain case B1's ``n < 2``
guard excludes.)
"""

from __future__ import annotations

from pyphi import System
from pyphi import examples
from pyphi import utils
from pyphi.conf import config
from pyphi.conf import presets
from pyphi.core import repertoire_algebra as _ra
from pyphi.direction import Direction
from pyphi.formalism import queries
from pyphi.substrate import all_sias

BINARY = [
    ("basic", examples.basic_substrate, (1, 0, 0)),
    ("xor", examples.xor_substrate, (0, 0, 0)),
    ("rule110", examples.rule110_substrate, (1, 0, 1)),
    ("grid3", examples.grid3_substrate, (1, 0, 0)),
    ("disj_conj", examples.disjunction_conjunction_substrate, (0, 0, 0, 0)),
]


def study_a(substrate, state) -> dict:
    """complexes()/major_complex: n(n-1) ceiling vs candidate phi_s (n>=2)."""
    sias = all_sias(substrate, state)
    cands = [(len(s.node_indices), float(s.phi)) for s in sias if s.node_indices]
    if not cands:
        return {"candidates": 0}
    global_max = max(phi for _, phi in cands)
    valid = [(n, p) for n, p in cands if n >= 2]
    skippable = sum(1 for n, _ in valid if n * (n - 1) < global_max - 1e-12)
    return {
        "candidates": len(cands),
        "n>=2_candidates": len(valid),
        "global_max_phi_s": round(global_max, 4),
        "valid_skippable_for_major": skippable,
        "valid_bite_frac": round(skippable / len(valid), 3) if valid else 0.0,
        "single_node_phi>0_out_of_domain": sum(
            1 for n, p in cands if n == 1 and p > 1e-12
        ),
    }


def study_b(substrate, state) -> dict:
    """MICE search: |M||Z| cap vs running-best small-phi, descending-cap order."""
    cs = System(substrate, state)
    total = skippable = 0
    for mechanism in utils.powerset(cs.node_indices, nonempty=True):
        for direction in (Direction.CAUSE, Direction.EFFECT):
            purviews = _ra.potential_purviews(cs, direction, mechanism, None)
            if not purviews:
                continue
            evaluated = [
                (
                    len(mechanism) * len(pv),
                    float(queries.find_mip(cs, direction, mechanism, pv).phi),
                )
                for pv in purviews
            ]
            evaluated.sort(key=lambda t: t[0], reverse=True)
            best = -1.0
            for cap, phi in evaluated:
                if best > cap + 1e-12:
                    skippable += 1
                else:
                    best = max(best, phi)
            total += len(evaluated)
    return {
        "total_purview_evals": total,
        "skippable": skippable,
        "bite_frac": round(skippable / total, 3) if total else 0.0,
    }


def main() -> None:
    config.progress_bars = False
    with config.override(**presets.iit4_2023):
        for name, factory, state in BINARY:
            substrate = factory()
            print(f"\n{name}  state={state}  (n={len(substrate.node_indices)})")
            print(f"  [A] complexes n(n-1) bite: {study_a(substrate, state)}")
            print(f"  [B] MICE |M||Z| bite:       {study_b(substrate, state)}")


if __name__ == "__main__":
    main()
