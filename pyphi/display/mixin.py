"""The Displayable mixin: the single site that reads repr_verbosity."""

from __future__ import annotations

from pyphi.conf import config
from pyphi.display.description import Description
from pyphi.display.render import render

LOW = 0
MEDIUM = 1
HIGH = 2
FULL = 3  # everything HIGH shows, plus exhaustive extras (e.g. RIA/MICE cut grids)


def _verbosity() -> int:
    return config.infrastructure.repr_verbosity


class Displayable:
    """Provides ``repr`` / ``str`` / HTML from a subclass ``_describe()`` hook.

    Verbosity policy (read from ``config.infrastructure.repr_verbosity``):

    - ``LOW`` (0): the one-line compact form only.
    - ``MEDIUM`` (1): the full card minus expensive embedded TPM grids
      (e.g. a ``Substrate``'s TPM, a ``System``'s conditioned cause/effect
      TPMs — which require computing marginals).
    - ``HIGH`` (2, the default): everything in the standard card.
    - ``FULL`` (3): the card plus exhaustive extras (e.g. the cut-matrix grid
      for an RIA/MICE partition).

    ``_describe(verbosity)`` honors this: it returns a compact-only
    ``Description`` at ``LOW`` (so no heavy sections are built) and gates
    embedded TPM grids behind ``HIGH``.
    """

    def _describe(self, verbosity: int) -> Description:
        raise NotImplementedError

    def __repr__(self) -> str:
        verbosity = _verbosity()
        description = self._describe(verbosity)
        if verbosity == LOW and description.compact is not None:
            return description.compact
        return render(description, backend="ascii", verbosity=verbosity)

    __str__ = __repr__

    def _compact_repr(self) -> str:
        """A one-line summary suitable for embedding in another object's repr.

        Describes at ``LOW`` so the compact form is produced without building
        the (potentially expensive) full card. Returns the description's
        ``compact`` form when defined, otherwise the title.
        """
        description = self._describe(LOW)
        return (
            description.compact if description.compact is not None else description.title
        )

    def _repr_html_(self) -> str:
        verbosity = _verbosity()
        description = self._describe(verbosity)
        if verbosity == LOW and description.compact is not None:
            # Match __repr__: a compact leaf, not a full card, at LOW.
            description = Description(
                title=description.title, compact=description.compact
            )
        return render(description, backend="html", verbosity=verbosity)

    def _repr_mimebundle_(self, **kwargs) -> dict[str, str]:  # noqa: ARG002
        return {"text/plain": str(self), "text/html": self._repr_html_()}
