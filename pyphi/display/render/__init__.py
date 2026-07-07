"""Display backends and the backend registry."""

from pyphi.display.description import Description
from pyphi.display.render import ascii as _ascii
from pyphi.display.render import html as _html

_BACKENDS = {"ascii": _ascii.render, "html": _html.render}


def render(description: Description, backend: str = "ascii", verbosity: int = 2) -> str:
    """Render ``description`` with the backend registered under ``backend``."""
    return _BACKENDS[backend](description, verbosity)
