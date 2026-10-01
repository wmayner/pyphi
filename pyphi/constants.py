# pyright: strict
# constants.py
"""Package-wide constants."""

import pickle
from pathlib import Path

#: The protocol used for pickling objects.
PICKLE_PROTOCOL: int = pickle.HIGHEST_PROTOCOL

DISK_CACHE_LOCATION: Path = Path("__pyphi_cache__")

#: Where the guide to porting pre-2.0 code is published.
MIGRATION_GUIDE_URL: str = (
    "https://pyphi.readthedocs.io/en/latest/migration/migration-2.0.html"
)

#: Node states
OFF: tuple[int, ...] = (0,)
ON: tuple[int, ...] = (1,)


# Probability value below which we issue a warning about precision.
TPM_WARNING_THRESHOLD: float = 1e-10
