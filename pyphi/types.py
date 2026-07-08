# pyright: strict
"""Type aliases for PyPhi.

Centralized type aliases used throughout the PyPhi codebase, for consistency
and readability. Import domain-specific types from this module rather than
using raw primitive types or numpy types directly.
"""

import numpy as np
from numpy.typing import NDArray

# =============================================================================
# Node and State Types
# =============================================================================

type NodeIndex = int
"""Index of a single node in a substrate."""

type NodeIndices = tuple[NodeIndex, ...]
"""Tuple of node indices representing a set of nodes."""

type State = tuple[int, ...]
"""State of a substrate or system, one integer per node.

Each entry is an index into that node's alphabet: ``0`` or ``1`` for a
binary node, and ``0..k-1`` for a node with a ``k``-valued alphabet.
"""

type Mechanism = tuple[NodeIndex, ...]
"""A mechanism is a set of nodes, represented as a tuple of node indices."""

type Purview = tuple[NodeIndex, ...]
"""A purview is a set of nodes over which a repertoire is defined."""

# =============================================================================
# NumPy Array Types
# =============================================================================

type TPMArray = NDArray[np.float64]
"""Transition Probability Matrix as a numpy array of float64 values.

The TPM defines the causal structure of a substrate by specifying the probability
of each node being ON given the state of its inputs.
"""

type ConnectivityMatrix = NDArray[np.int_]
"""Connectivity matrix defining which nodes are connected.

A binary matrix where ``cm[i, j] == 1`` indicates node ``i`` has a causal
effect on node ``j`` (i.e. ``i`` is an input to ``j``).
"""

type Repertoire = NDArray[np.float64]
"""Probability distribution over states (cause or effect repertoire).

Repertoires represent the cause or effect power of a mechanism over a purview.
"""

type StateArray = NDArray[np.int_]
"""Array representation of a state (as opposed to tuple representation)."""

# =============================================================================
# Phi Types
# =============================================================================

type Phi = float
"""Integrated information value (φ).

Can represent small phi (mechanism integration) or big phi (system integration).
"""

type SmallPhi = float
"""Small phi (φ) - integrated information of a mechanism."""

type BigPhi = float
"""Big phi (Φ) - integrated information of a system."""

# =============================================================================
# Distance Measure Types
# =============================================================================

type Distance = float
"""Distance between two probability distributions or structures."""

# =============================================================================
# Configuration Types
# =============================================================================

type Precision = int
"""Numerical precision for floating point comparisons."""

# =============================================================================
# General Utility Types
# =============================================================================

# Note: Additional type aliases can be added here as the typing effort progresses.
# For example:
# - Partition types (once pyphi/partition.py is typed)
# - Cut types (once pyphi/models/cuts.py is fully typed)
# - Cache types (once pyphi/cache is typed)
