"""Approach A of the partial-distinction certified Φ bracket.

The computable core: the exact state-keyed identity (lower endpoint on Σφ_r),
the measured state-keyed certificate (upper endpoint in the complete-distinction
limit), and the wildcard extension that bounds the contribution of un-evaluated
candidate mechanisms. Validated by ``test_bracket.py`` and by the truncation
sweep in ``verify_certified_bracket.py``; promoted into
``pyphi/formalism/iit4/bounds.py`` only if that sweep confirms soundness and
usefulness.

Notation follows ``experiments/so_certificate_experiments/FINDINGS.md``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from typing import Any


def g(k: int) -> float:
    """The Eq. 14 per-o linear-program maximum weight, (2^k − 1 − k)/k."""
    return (2.0**k - 1.0 - k) / k if k > 0 else 0.0


@dataclass
class Profile:
    """The measured state-keyed incidence profile of a distinction set.

    Attributes
    ----------
    state_groups : dict
        Maps each UnitState pair ``o`` to the ascending list of densities
        ``q_d = φ_d/|purview_union_d|`` of the distinctions incident to it.
    self_sum : float
        The exact self-relation sum ``Σ_d |z*_c ∩ z*_e|·q_d``.
    """

    state_groups: dict[Any, list[float]] = field(default_factory=dict)
    self_sum: float = 0.0


def profile_from_distinctions(distinctions) -> Profile:
    groups: dict[Any, list[float]] = defaultdict(list)
    self_sum = 0.0
    for d in distinctions:
        union = set(d.purview_union)
        if not union:
            continue
        phi = float(d.phi)
        density = phi / len(union)
        inter = set(d.cause.purview_units) & set(d.effect.purview_units)
        self_sum += len(inter) * density
        for o in union:
            groups[o].append(density)
    for densities in groups.values():
        densities.sort()
    return Profile(state_groups=dict(groups), self_sum=self_sum)


def identity_cross(profile: Profile) -> float:
    total = 0.0
    for densities in profile.state_groups.values():
        k = len(densities)
        total += sum(q * (2.0 ** (k - (i + 1)) - 1.0) for i, q in enumerate(densities))
    return total


def measured_cross_certificate(profile: Profile) -> float:
    total = 0.0
    for densities in profile.state_groups.values():
        total += sum(densities) * g(len(densities))
    return total


def sum_phi_relations_lower(profile: Profile) -> float:
    return profile.self_sum + identity_cross(profile)
