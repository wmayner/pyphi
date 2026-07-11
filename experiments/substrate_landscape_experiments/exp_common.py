"""Shared helpers for the substrate-parameter exploration experiments."""

import json

import numpy as np

import pyphi
from pyphi.substrate_generator import build_substrate
from pyphi.substrate_generator import ising

# IIT 4.0 (2023) Fig 1A weights, convention weights[i, j] = i -> j.
# A<->B = +0.7, A->C = +0.2, C->B = -0.8, self A,B = -0.2, self C = +0.2.
FIG1A_WEIGHTS = np.array(
    [
        [-0.2, 0.7, 0.2],
        [0.7, -0.2, 0.0],
        [0.0, -0.8, 0.2],
    ]
)
TEMPERATURE = 1 / 4
STATE = (1, 0, 0)


def make_system(weights, state=STATE, temperature=TEMPERATURE):
    sub = build_substrate(
        [ising.probability] * weights.shape[0], weights, temperature=temperature
    )
    return sub


def sia_summary(weights, state=STATE, temperature=TEMPERATURE):
    """Full-system SIA -> dict of value + selection identity."""
    sub = make_system(weights, state, temperature)
    s = pyphi.analyze(sub, state, compute="sia")
    part = s.partition
    return {
        "phi": float(s.phi),
        "normalized_phi": float(s.normalized_phi),
        "signed_phi": float(s.signed_phi) if s.signed_phi is not None else None,
        "partition": repr_partition(part),
        "cause_state": tuple(int(x) for x in s.system_state.cause.state),
        "effect_state": tuple(int(x) for x in s.system_state.effect.state),
        "phi_cause": float(s.cause.phi) if s.cause is not None else None,
        "phi_effect": float(s.effect.phi) if s.effect is not None else None,
    }


def repr_partition(part):
    """Stable compact identity for a system partition."""
    try:
        # DirectedSetPartition: parts + cut connections
        return str(sorted(tuple(sorted(p.mechanism)) for p in part))
    except Exception:
        pass
    try:
        return str(np.argwhere(part.cut_matrix(3)).tolist())
    except Exception:
        return str(part)


def save_json(path, obj):
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=default)
