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

from itertools import permutations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyphi.substrate import Substrate


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
