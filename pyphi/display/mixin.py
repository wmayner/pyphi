"""The Displayable mixin: the single site that reads repr_verbosity."""

from __future__ import annotations

from pyphi.conf import config
from pyphi.display.description import Description
from pyphi.display.render import render

LOW = 0
MEDIUM = 1
HIGH = 2


def _verbosity() -> int:
    return config.infrastructure.repr_verbosity


class Displayable:
    """Provides ``repr`` / ``str`` / HTML from a subclass ``_describe()`` hook."""

    def _describe(self, verbosity: int) -> Description:
        raise NotImplementedError

    def __repr__(self) -> str:
        verbosity = _verbosity()
        description = self._describe(verbosity)
        if verbosity == LOW and description.compact is not None:
            return description.compact
        return render(description, backend="ascii", verbosity=verbosity)

    __str__ = __repr__

    def _repr_html_(self) -> str:
        verbosity = _verbosity()
        return render(self._describe(verbosity), backend="html", verbosity=verbosity)

    def _repr_mimebundle_(self, **kwargs) -> dict[str, str]:  # noqa: ARG002
        return {"text/plain": str(self), "text/html": self._repr_html_()}
