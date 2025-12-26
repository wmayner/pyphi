# constants.py
"""Package-wide constants."""

import pickle
from pathlib import Path
from typing import Tuple

#: The protocol used for pickling objects.
PICKLE_PROTOCOL: int = pickle.HIGHEST_PROTOCOL

DISK_CACHE_LOCATION: Path = Path("__pyphi_cache__")

#: Node states
OFF: Tuple[int, ...] = (0,)
ON: Tuple[int, ...] = (1,)


# Probability value below which we issue a warning about precision.
# TODO(4.0)
TPM_WARNING_THRESHOLD: float = 1e-10
