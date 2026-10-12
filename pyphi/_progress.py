"""The progress bar PyPhi uses."""

import warnings
from typing import Any

from tqdm import TqdmWarning

with warnings.catch_warnings():
    # In a notebook without ipywidgets, tqdm.auto warns and falls back to the
    # console bar; the console bar is the right choice there, so say nothing.
    warnings.simplefilter("ignore", TqdmWarning)
    from tqdm.auto import tqdm as _tqdm

#: Seconds of work before a bar is drawn. A computation that finishes sooner
#: prints nothing.
DELAY: float = 1.0


class tqdm(_tqdm):
    """``tqdm.auto.tqdm`` that appears only once work has run for ``DELAY``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("delay", DELAY)
        super().__init__(*args, **kwargs)
