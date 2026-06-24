"""Behavior-aware substrate canonicalization.

A substrate's identity is its connectivity, its per-node TPM, and its
per-node alphabet sizes. A node permutation is a substrate **automorphism**
only when it preserves all three -- so a node implementing one mechanism is
never identified with a node implementing a different one, even when their
wiring is identical.

Canonicalization is exact: the automorphism group and canonical form are
found by enumerating node permutations. This is tractable because Phi is
``O(2**n)``, so substrates on which it is computed have few nodes; the
asymptotic regime where graph-isomorphism libraries (e.g. nauty) would help
is one in which Phi itself is intractable.
"""

from __future__ import annotations

from functools import cache
from itertools import permutations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyphi.substrate import Substrate

# Decimal places used when byte-keying TPM arrays for cross-substrate
# comparison, to make canonical-form equality robust to float round-off.
_ROUND = 12


def _relabel_joint(arr: np.ndarray, perm: tuple[int, ...]) -> np.ndarray:
    """Relabel a joint TPM array under ``perm`` (destination ``i`` <- source
    ``perm[i]``).

    ``arr`` has shape ``(*alphabet_sizes, n_nodes, max_alphabet)``: the first
    ``n`` axes are input-state axes, axis ``-2`` is the output-node axis, and
    axis ``-1`` is the per-node next-state distribution (which travels with its
    node). Permuting the input axes and reindexing the node axis relabels the
    nodes.
    """
    n = len(perm)
    return arr.transpose((*perm, n, n + 1))[..., list(perm), :]


def _candidate_perms(substrate: Substrate) -> tuple[tuple[int, ...], ...]:
    """Node permutations preserving connectivity and alphabet sizes.

    These are the only permutations that can be substrate automorphisms or
    isomorphisms; pruning here also avoids comparing arrays of mismatched
    shape (a permutation across differing alphabets reshapes the TPM).
    """
    cm = np.asarray(substrate.cm)
    alphabet = substrate.tpm.alphabet_sizes
    n = len(alphabet)
    out = []
    for perm in permutations(range(n)):
        if any(alphabet[perm[i]] != alphabet[i] for i in range(n)):
            continue
        if not np.array_equal(cm[np.ix_(perm, perm)], cm):
            continue
        out.append(perm)
    return tuple(out)


def substrate_automorphisms(
    substrate: Substrate,
) -> tuple[tuple[int, ...], ...]:
    """All node permutations preserving connectivity, TPM, and alphabet sizes.

    Always contains the identity permutation.
    """
    arr = substrate.tpm.to_joint()
    return tuple(
        perm
        for perm in _candidate_perms(substrate)
        if np.array_equal(_relabel_joint(arr, perm), arr)
    )


def _serialization(substrate: Substrate, perm: tuple[int, ...]) -> tuple:
    """A relabeling-applied, byte-comparable key for ``substrate`` under
    ``perm``.

    Rounding makes cross-substrate equality robust to float round-off; the
    ``+ 0.0`` normalizes ``-0.0`` to ``0.0`` so equal values compare equal
    byte-for-byte.
    """
    cm = np.asarray(substrate.cm)
    alphabet = substrate.tpm.alphabet_sizes
    n = len(alphabet)
    arr = _relabel_joint(substrate.tpm.to_joint(), perm)
    cm_p = np.ascontiguousarray(cm[np.ix_(perm, perm)])
    arr_p = np.ascontiguousarray(np.round(arr, _ROUND)) + 0.0
    alpha_p = tuple(alphabet[perm[i]] for i in range(n))
    return (alpha_p, cm_p.tobytes(), arr_p.tobytes())


@cache
def _canonical(
    substrate: Substrate,
) -> tuple[tuple, tuple[tuple[int, ...], ...]]:
    """Return ``(canonical_key, achievers)``.

    ``canonical_key`` is the lexicographically smallest serialization over
    candidate permutations; ``achievers`` is every permutation attaining it
    (the set mapping ``substrate`` to its canonical form).
    """
    best_key = None
    achievers: list[tuple[int, ...]] = []
    for perm in _candidate_perms(substrate):
        key = _serialization(substrate, perm)
        if best_key is None or key < best_key:
            best_key, achievers = key, [perm]
        elif key == best_key:
            achievers.append(perm)
    # The identity permutation is always a candidate, so the loop always runs.
    assert best_key is not None
    return best_key, tuple(achievers)


def substrate_canonical_form(
    substrate: Substrate,
) -> tuple[Substrate, tuple[int, ...]]:
    """Return ``(canonical_substrate, canonical_permutation)``.

    ``canonical_substrate`` is the lexicographically smallest relabeling of
    ``substrate``; ``canonical_permutation`` is the smallest permutation
    attaining it (unique by construction).
    """
    from pyphi.substrate import Substrate

    _, achievers = _canonical(substrate)
    perm = min(achievers)
    arr = _relabel_joint(substrate.tpm.to_joint(), perm)
    cm = np.asarray(substrate.cm)[np.ix_(perm, perm)]
    state_space = substrate.state_space
    permuted_state_space = tuple(state_space[perm[i]] for i in range(len(perm)))
    canonical = Substrate(tpm=arr, cm=cm, state_space=permuted_state_space)
    return canonical, perm


def are_substrates_isomorphic(s1: Substrate, s2: Substrate) -> bool:
    """Whether some node permutation maps ``s1``'s connectivity, TPM, and
    alphabet sizes onto ``s2``'s."""
    if sorted(s1.tpm.alphabet_sizes) != sorted(s2.tpm.alphabet_sizes):
        return False
    return _canonical(s1)[0] == _canonical(s2)[0]


def canonical_state(substrate: Substrate, state: tuple[int, ...]) -> tuple[int, ...]:
    """Map ``state`` into canonical coordinates, reduced over the automorphism
    orbit.

    For substrates related by a node permutation, corresponding states'
    canonical-coordinate images agree only up to an automorphism, so the
    permutation-invariant identity of ``state`` is the lexicographically
    smallest image over every permutation that carries ``substrate`` to its
    canonical form.
    """
    _, achievers = _canonical(substrate)
    return min(tuple(state[perm[i]] for i in range(len(perm))) for perm in achievers)
