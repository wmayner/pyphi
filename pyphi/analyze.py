"""One high-level entry point for a single system's IIT analysis.

``analyze`` takes a substrate and a state, builds the candidate system, runs
the analysis under the active (or a named) formalism, and returns an
:class:`Analysis` — a small bundle exposing the system irreducibility analysis,
the cause-effect structure, and the scalar Φ uniformly across formalisms. A
``compute`` argument selects a cheaper or custom result instead of the bundle;
a ``grains`` argument runs the bounded intrinsic-unit search over the whole
substrate instead, returning its complexes.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import pandas as pd

from pyphi.conf import config
from pyphi.conf import presets
from pyphi.display import FULL
from pyphi.display import Description
from pyphi.display import Displayable
from pyphi.display.numbers import format_value
from pyphi.system import System


@dataclass(frozen=True, repr=False)
class Analysis(Displayable):
    """A single system's analysis: its SIA, its CES, and the scalar Φ.

    Uniform across formalisms: under IIT 4.0 the cause-effect structure embeds
    its own SIA, while under IIT 3.0 the CES is the bare set of distinctions and
    the SIA is computed separately; either way ``sia`` / ``ces`` / ``phi`` are
    populated.
    """

    system: System
    sia: Any
    ces: Any

    @property
    def phi(self) -> float:
        return float(self.sia.phi)

    def _describe(self, verbosity: int) -> Description:
        # Reuse the cause-effect structure's flat rich card. Under IIT 4.0 it
        # already folds in the embedded SIA; under IIT 3.0 the CES is bare
        # Distinctions, so append the separately-computed SIA's sections flat
        # (capped at FULL) so the card still leads with Φ.
        desc = self.ces._describe(verbosity)
        sections = list(desc.sections)
        if getattr(self.ces, "sia", None) is None:
            sections.extend(self.sia._describe(min(verbosity, FULL)).sections)
        return Description(
            title="Analysis",
            sections=tuple(sections),
            compact=f"Analysis(Φ={format_value(self.phi)})",
        )

    def to_pandas(self) -> pd.DataFrame:
        # IIT 4.0: ces carries .distinctions and .relations.
        # IIT 3.0: ces is the Distinctions sequence itself (no relations).
        distinctions = getattr(self.ces, "distinctions", self.ces)
        relations = getattr(self.ces, "relations", None)
        sum_phi_r = float(relations.sum_phi()) if relations is not None else float("nan")
        return pd.DataFrame(
            [
                {
                    "phi": float(self.sia.phi),
                    "normalized_phi": float(
                        getattr(self.sia, "normalized_phi", float("nan"))
                    ),
                    "n_distinctions": len(distinctions),
                    "sum_phi_r": sum_phi_r,
                }
            ]
        )


def analyze(
    substrate: Any,
    state: tuple[int, ...],
    *,
    subset: Any = None,
    formalism: str | None = None,
    compute: Any = None,
    grains: Any = None,
    parallel_kwargs: dict | None = None,
) -> Analysis | Any:
    """Analyze one candidate system over ``substrate`` in ``state``, or run
    a grain search over the whole substrate.

    Parameters
    ----------
    substrate
        The substrate to analyze.
    state : tuple[int, ...]
        The state of the substrate's nodes. When ``grains`` admits update
        grains above 1, a sequence of micro states (oldest first) of the
        required length instead — see
        :func:`pyphi.macro.complexes`.
    subset : optional
        Node indices of the candidate system; ``None`` uses the whole
        substrate. Incompatible with ``grains``.
    formalism : str or None, optional
        ``None`` uses the active config formalism; a version name
        (``"IIT_3_0"`` / ``"IIT_4_0_2023"`` / ``"IIT_4_0_2026"``) applies that
        formalism for this call only.
    compute : optional
        ``None`` returns an :class:`Analysis` bundle; ``"sia"`` or ``"ces"``
        returns the raw result object; a callable returns ``compute(system)``.
        Incompatible with ``grains``.
    grains : optional
        ``None`` analyzes the single candidate system. ``True`` runs the
        bounded intrinsic-unit search with default
        :class:`~pyphi.macro.SearchBounds`; a
        :class:`~pyphi.macro.SearchBounds` instance runs it with those
        bounds. The search returns its
        :class:`~pyphi.macro.ComplexesResult`.
    parallel_kwargs : dict or None, optional
        Forwarded to :func:`pyphi.macro.complexes`; meaningful only with
        ``grains``.

    Returns
    -------
    Analysis
        The full analysis bundle; the raw result object when ``compute``
        selects one; or the grain search's
        :class:`~pyphi.macro.ComplexesResult` when ``grains`` is set.

    Raises
    ------
    ValueError
        If ``formalism`` is not a known version name; if ``compute`` is not
        ``"sia"``, ``"ces"``, a callable, or ``None``; if ``grains`` is not
        ``True``, a :class:`~pyphi.macro.SearchBounds`, or ``None``; if
        ``grains`` is combined with ``subset`` or ``compute``; or if
        ``parallel_kwargs`` is given without ``grains``.
    """
    if formalism is not None and formalism not in presets.by_name:
        valid = ", ".join(sorted(presets.by_name))
        raise ValueError(f"unknown formalism {formalism!r}; expected one of: {valid}")

    bounds: Any = None
    if grains is None:
        if parallel_kwargs is not None:
            raise ValueError(
                "parallel_kwargs applies only to a grain search; pass grains="
            )
    else:
        from pyphi.macro.search import SearchBounds

        if subset is not None:
            raise ValueError(
                "grains cannot be combined with subset: the grain search "
                "assembles candidate systems over the whole universe"
            )
        if compute is not None:
            raise ValueError(
                "grains cannot be combined with compute: the grain search "
                "returns its ComplexesResult"
            )
        if grains is True:
            bounds = SearchBounds()
        elif isinstance(grains, SearchBounds):
            bounds = grains
        else:
            raise ValueError(
                f"grains must be True or a SearchBounds instance; got {grains!r}"
            )

    ctx = (
        config.override(**presets.by_name[formalism])
        if formalism is not None
        else nullcontext()
    )
    result: Any = None
    with ctx:
        if bounds is not None:
            from pyphi.macro.search import complexes as grain_complexes

            result = grain_complexes(
                substrate, state, bounds, parallel_kwargs=parallel_kwargs
            )
        else:
            indices = substrate.node_indices if subset is None else subset
            system = System.from_substrate(substrate, state, indices)
            if callable(compute):
                result = compute(system)
            elif compute == "sia":
                result = system.sia()
            elif compute == "ces":
                result = system.ces()
            elif compute is not None:
                raise ValueError(
                    f"unknown compute: {compute!r}; expected 'sia', 'ces', a "
                    "callable, or None for the full Analysis bundle"
                )
            else:
                ces = system.ces()
                sia = getattr(ces, "sia", None)
                if sia is None:
                    sia = system.sia()
                result = Analysis(system=system, sia=sia, ces=ces)
    return result
