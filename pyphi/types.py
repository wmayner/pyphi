"""Type aliases for PyPhi.

This module provides centralized type aliases used throughout the PyPhi codebase.
These aliases improve consistency, readability, and make refactoring easier.

All new code should import domain-specific types from this module rather than
using raw primitive types or numpy types directly.
"""

from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

# =============================================================================
# Node and State Types
# =============================================================================

NodeIndex: TypeAlias = int
"""Index of a single node in a network."""

NodeIndices: TypeAlias = tuple[NodeIndex, ...]
"""Tuple of node indices representing a set of nodes."""

State: TypeAlias = tuple[int, ...]
"""State of a network or subsystem as a tuple of binary values (0 or 1)."""

Mechanism: TypeAlias = tuple[NodeIndex, ...]
"""A mechanism is a set of nodes, represented as a tuple of node indices."""

Purview: TypeAlias = tuple[NodeIndex, ...]
"""A purview is a set of nodes over which a repertoire is defined."""

# =============================================================================
# NumPy Array Types
# =============================================================================

TPMArray: TypeAlias = NDArray[np.float64]
"""Transition Probability Matrix as a numpy array of float64 values.

The TPM defines the causal structure of a network by specifying the probability
of each node being ON given the state of its inputs.
"""

ConnectivityMatrix: TypeAlias = NDArray[np.int_]
"""Connectivity matrix defining which nodes are connected.

A binary matrix where cm[i, j] = 1 indicates node j has a causal effect on node i.
"""

Repertoire: TypeAlias = NDArray[np.float64]
"""Probability distribution over states (cause or effect repertoire).

Repertoires represent the cause or effect power of a mechanism over a purview.
"""

StateArray: TypeAlias = NDArray[np.int_]
"""Array representation of a state (as opposed to tuple representation)."""

# =============================================================================
# Phi Types
# =============================================================================

Phi: TypeAlias = float
"""Integrated information value (φ).

Can represent small phi (mechanism integration) or big phi (system integration).
"""

SmallPhi: TypeAlias = float
"""Small phi (φ) - integrated information of a mechanism."""

BigPhi: TypeAlias = float
"""Big phi (Φ) - integrated information of a system."""

# =============================================================================
# Distance Measure Types
# =============================================================================

Distance: TypeAlias = float
"""Distance between two probability distributions or structures."""

# =============================================================================
# Configuration Types
# =============================================================================

Precision: TypeAlias = int
"""Numerical precision for floating point comparisons."""

# =============================================================================
# General Utility Types
# =============================================================================

# Note: Additional type aliases can be added here as the typing effort progresses.
# For example:
# - Partition types (once pyphi/partition.py is typed)
# - Cut types (once pyphi/models/cuts.py is fully typed)
# - Cache types (once pyphi/cache is typed)
