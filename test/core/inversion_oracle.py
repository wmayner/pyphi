"""Dense reference for the cause inversion (IIT 4.0 Eq. 4).

The production implementation is a greedy sum-product contraction; this
oracle materializes the full joint likelihood over all substrate units
(a^N) and computes the same quantities directly. Only usable for small
substrates, which is exactly its job: an independent implementation the
reduced path is cross-validated against.
"""

from __future__ import annotations

import numpy as np


def dense_cause_marginal_reference(factored, state, node_indices):
    """Return ``{unit index: cause factor}`` for units in ``node_indices``."""
    n = factored.n_nodes
    alphabet_sizes = factored.alphabet_sizes
    all_indices = tuple(range(n))
    system_indices = tuple(sorted(node_indices))
    background_indices = tuple(sorted(set(all_indices) - set(system_indices)))

    pr_joint = np.ones(alphabet_sizes, dtype=np.float64)
    for i in all_indices:
        pr_joint = pr_joint * factored.factor(i)[..., state[i]]

    if system_indices:
        pr_bg = pr_joint.sum(axis=system_indices, keepdims=True)
    else:
        pr_bg = pr_joint.copy()

    norm = pr_joint.sum()
    assert norm > 0.0, "oracle: unreachable state"
    weight = pr_bg / norm

    out = {}
    for i in node_indices:
        weighted = factored.factor(i) * weight[..., np.newaxis]
        if background_indices:
            weighted = weighted.sum(axis=background_indices, keepdims=True)
        out[i] = weighted
    return out
