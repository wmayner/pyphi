"""Generate the IIT 4.0 (2023) pre-refactor non-regression oracle.

This script is the *reproducer* for ``test/data/iit4_2023_byte_match_oracle.json``,
which pins pre-2.0-refactor IIT 4.0 (2023) system-phi values consumed by
``test/test_cross_formalism_invariants.py::TestPreRefactorByteMatch`` (roadmap
item B5).

It MUST be run against a worktree of the oracle commit
``b3aaa3e59fbca7d63dcb532a80d8337c0fb467d3`` — it imports the pre-refactor API
(``pyphi.Network`` / ``pyphi.new_big_phi.sia``) and the flat (UPPERCASE) config,
neither of which exists on the 2.0 branch. It lives under ``scripts/`` (outside
the pytest ``testpaths``) so the current suite never tries to import it.

Regeneration workflow::

    git worktree add --detach /tmp/oracle-b3aaa3e5 b3aaa3e5
    cp scripts/gen_iit4_2023_byte_match_oracle.py /tmp/oracle-b3aaa3e5/
    cd /tmp/oracle-b3aaa3e5
    PYPHI_WELCOME_OFF=yes uv run --extra emd python \
        gen_iit4_2023_byte_match_oracle.py /path/to/repo/test/data/iit4_2023_byte_match_oracle.json

Config notes: the oracle uses the ``DIRECTED_BI`` system partition scheme (which
the 2.0 refactor *preserves* as ``DIRECTED_BIPARTITION``) and the default GID
measures at precision 13 — NOT the era's default ``SET_UNI/BI`` scheme, which
was intentionally replaced by ``DIRECTED_SET_PARTITION`` and is therefore not a
non-regression target. The recorded ``phi`` is the *raw* (un-clamped)
integration; the 2.0 code clamps it to ``max(0, raw)`` and exposes the raw value
as ``signed_phi`` (see the consuming test).
"""

import json
import sys
from pathlib import Path

import numpy as np

import pyphi
from pyphi import new_big_phi


def _logic_net_3():
    """3-node net: A=OR(B,C), B=AND(A,C), C=XOR(A,B). Fully connected."""
    tpm = np.array(
        [
            [0.0, 0.0, 0.0],  # 000
            [1.0, 0.0, 1.0],  # 100
            [1.0, 0.0, 1.0],  # 010
            [1.0, 1.0, 0.0],  # 110
            [1.0, 0.0, 1.0],  # 001
            [1.0, 1.0, 0.0],  # 101
            [1.0, 1.0, 0.0],  # 011
            [1.0, 1.0, 1.0],  # 111
        ]
    )
    cm = np.ones((3, 3), dtype=int)
    return tpm, cm


def _copy_loop_2():
    """2-node copy loop: A'=B, B'=A (deterministic, irreducible)."""
    tpm = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    cm = np.array([[0, 1], [1, 0]], dtype=int)
    return tpm, cm


def _random_nets(rng, count, n):
    return [(rng.random((2**n, n)), np.ones((n, n), dtype=int)) for _ in range(count)]


def _all_states(n):
    """All states in little-endian order (first node varies fastest)."""
    return [tuple((i >> b) & 1 for b in range(n)) for i in range(2**n)]


def build_corpus():
    """Yield (name, tpm, cm, state, nodes) for every state of each net.

    Backward-unreachable states are filtered later (in ``compute_one``).
    """
    rng = np.random.default_rng(20260614)
    nets = [
        ("logic_or_and_xor_3", *_logic_net_3()),
        ("copy_loop_2", *_copy_loop_2()),
    ]
    nets += [(f"random_2_{i}", t, c) for i, (t, c) in enumerate(_random_nets(rng, 4, 2))]
    nets += [(f"random_3_{i}", t, c) for i, (t, c) in enumerate(_random_nets(rng, 3, 3))]

    corpus = []
    for name, tpm, cm in nets:
        n = np.array(tpm).shape[1]
        for state in _all_states(n):
            sname = f"{name}__{''.join(map(str, state))}"
            corpus.append((sname, tpm, cm, state, None))
    return corpus


CONFIG = dict(
    IIT_VERSION=4.0,
    SYSTEM_PARTITION_TYPE="DIRECTED_BI",
    REPERTOIRE_DISTANCE="GENERALIZED_INTRINSIC_DIFFERENCE",
    REPERTOIRE_DISTANCE_SPECIFICATION="GENERALIZED_INTRINSIC_DIFFERENCE",
    REPERTOIRE_DISTANCE_DIFFERENTIATION="GENERALIZED_INTRINSIC_DIFFERENCE",
    PRECISION=13,
    VALIDATE_SUBSYSTEM_STATES=False,
    PROGRESS_BARS=False,
    SHORTCIRCUIT_SIA=True,
)


def compute_one(tpm, cm, state, nodes):
    network = pyphi.Network(np.array(tpm), cm=np.array(cm))
    try:
        subsystem = pyphi.Subsystem(network, tuple(state), nodes)
    except pyphi.exceptions.StateUnreachableBackwardsError:
        return None
    result = new_big_phi.sia(subsystem)
    return float(result.phi), float(result.normalized_phi)


def main(dest: Path):
    records = []
    with pyphi.config.override(**CONFIG):
        for name, tpm, cm, state, nodes in build_corpus():
            computed = compute_one(tpm, cm, state, nodes)
            if computed is None:
                print(f"{name}: SKIPPED (backward-unreachable state)")
                continue
            phi, nphi = computed
            records.append(
                {
                    "name": name,
                    "tpm": np.array(tpm).tolist(),
                    "cm": np.array(cm).tolist(),
                    "state": list(state),
                    "nodes": list(nodes) if nodes is not None else None,
                    "phi": phi,
                    "phi_hex": phi.hex(),
                    "normalized_phi": nphi,
                    "normalized_phi_hex": nphi.hex(),
                }
            )
            print(f"{name}: phi={phi!r} normalized_phi={nphi!r}")

    out = {
        "oracle_commit": "b3aaa3e59fbca7d63dcb532a80d8337c0fb467d3",
        "formalism": "IIT_4_0_2023",
        "config": CONFIG,
        "records": records,
    }
    dest.write_text(json.dumps(out, indent=2, sort_keys=True))
    nonzero = sum(1 for r in records if abs(r["phi"]) > 1e-9)
    print(f"\nWrote {len(records)} records ({nonzero} nonzero) to {dest}")


if __name__ == "__main__":
    out_path = Path(sys.argv[1] if len(sys.argv) > 1 else "iit4_2023_byte_match_oracle.json")
    main(out_path.resolve())
