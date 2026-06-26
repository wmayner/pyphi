# pyright: strict
# constants.py
"""Package-wide constants."""

import pickle
from pathlib import Path

#: The protocol used for pickling objects.
PICKLE_PROTOCOL: int = pickle.HIGHEST_PROTOCOL

DISK_CACHE_LOCATION: Path = Path("__pyphi_cache__")

#: Node states
OFF: tuple[int, ...] = (0,)
ON: tuple[int, ...] = (1,)


# Probability value below which we issue a warning about precision.
TPM_WARNING_THRESHOLD: float = 1e-10
