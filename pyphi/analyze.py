"""One high-level entry point for IIT analysis.

``analyze`` takes a substrate and a state, builds the candidate system, runs
the analysis, and returns an :class:`Analysis` — a small bundle exposing the
system irreducibility analysis, the cause-effect structure, and their scalar
values. A ``compute`` argument selects a cheaper or custom result instead of
the bundle; a ``grains`` argument runs the bounded intrinsic-unit search over
the whole substrate instead, returning its complexes.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from dataclasses import replace
from typing import Any

import pandas as pd

from pyphi import numerics
from pyphi.conf import config
from pyphi.conf import presets
from pyphi.conf.formalism import IITConfig
from pyphi.display import FULL
from pyphi.display import Description
from pyphi.display import Displayable
from pyphi.display import Row
from pyphi.display import Section
from pyphi.display import system_phi_label
from pyphi.display.numbers import format_value
from pyphi.serializable import Serializable
from pyphi.system import System


@dataclass(frozen=True, repr=False)
class Analysis(Displayable, Serializable):
    """A single system's analysis: its SIA, its CES, and their scalar values.

    The cause-effect structure embeds the system irreducibility analysis, and
    ``sia``, ``ces``, and ``phi`` are always populated.

    Notes
    -----
    ``phi`` and ``big_phi`` are different quantities. ``phi`` is φₛ, the system
    integrated information, which decides whether the system exists as one
    whole; ``big_phi`` is Φ, the structure integrated information, the sum of φ
    over the Φ-structure's distinctions and relations. A system can have
    φₛ = 0 — it is reducible — while its distinctions still sum to a nonzero Φ.

    For results computed under an earlier version of IIT (see
    :doc:`/howto/earlier-versions`): under IIT 3.0 the CES is the bare set of
    distinctions and the SIA is computed separately, the system-level quantity
    ``phi`` is that version's Φ, and ``big_phi`` is not defined.
    """

    system: System
    sia: Any
    ces: Any

    @property
    def phi(self) -> float:
        """float: φₛ, the system integrated information."""
        return float(self.sia.phi)

    @property
    def formalism(self) -> str:
        """str: The version of IIT that produced this analysis, read from the
        configuration snapshot the system irreducibility analysis carries."""
        return self.sia.config.formalism.iit.version

    @property
    def _phi_label(self) -> str:
        # Under IIT 3.0 the CES is a bare distinction sequence carrying no
        # config snapshot; the SIA always carries one.
        config = getattr(self.ces, "config", None) or getattr(self.sia, "config", None)
        return system_phi_label(config)

    @property
    def big_phi(self) -> float:
        """float: Φ, the structure integrated information — the sum of φ over
        the Φ-structure's distinctions and relations.

        Raises
        ------
        AttributeError
            For a result computed under IIT 3.0, which has no relations and so
            no structure integrated information. That version's Φ is
            :attr:`phi`.
        """
        if self._phi_label == "Φ":
            raise AttributeError(
                "IIT 3.0 has no structure integrated information; its Φ is the "
                "system-level value on `.phi`."
            )
        return float(self.ces.big_phi)

    def _describe(self, verbosity: int) -> Description:
        # Reuse the cause-effect structure's flat rich card. Under IIT 4.0 it
        # already folds in the embedded SIA; under IIT 3.0 the CES is bare
        # Distinctions, so append the separately-computed SIA's sections flat
        # (capped at FULL) so the card still leads with the system-level value.
        desc = self.ces._describe(verbosity)
        sections = list(desc.sections)
        # A result computed under an earlier version of IIT closes its summary
        # with that version. The version is always recorded on ``.formalism``
        # and in serialized results.
        if self.formalism != IITConfig.version:
            first = sections[0]
            sections[0] = replace(
                first, rows=(*first.rows, Row("Formalism", self.formalism))
            )
        if getattr(self.ces, "sia", None) is None:
            sections.extend(self.sia._describe(min(verbosity, FULL)).sections)
        elif verbosity < FULL and getattr(self.sia, "partition", None) is not None:
            # Below FULL the structure card does not embed the SIA; show the
            # system-level facts a reader needs to interpret φₛ.
            sections.append(Section(label="System", rows=self._system_rows()))
        phi_label = self._phi_label
        return Description(
            title="Analysis",
            sections=tuple(sections),
            compact=f"Analysis({phi_label}={format_value(self.phi)})",
        )

    def _system_rows(self) -> tuple[Row, ...]:
        from pyphi.models.partitions import concise_partition

        rows = [
            Row("Units", self.sia._system_label()),
            Row("Current state", self.sia.current_state),
        ]
        state = getattr(self.sia, "system_state", None)
        if state is not None:
            if state.cause is not None:
                rows.append(Row("Specified cause state", state.cause.state))
            if state.effect is not None:
                rows.append(Row("Specified effect state", state.effect.state))
        rows.append(Row("MIP", concise_partition(self.sia.partition)))
        ii = getattr(self.sia, "intrinsic_information", None)
        if ii is not None:
            rows.append(Row("ii(s)", ii))
            findings = {f.kind: f for f in self.sia.explain().findings}
            binding = findings.get("requirement_binding")
            if binding is not None:
                direction = dict(binding.detail)["direction"]
                rows.append(Row("Requirement binds", f"{binding.value} ({direction})"))
            # An irreducible system whose φₛ is zero only because it has no
            # intrinsic information: say so, since nothing else on the card does.
            sides = dict(getattr(findings.get("binding_direction"), "detail", ()))
            if (
                binding is not None
                and binding.value == "differentiation"
                and numerics.is_zero(ii)
                and sides
                and numerics.is_positive(min(sides.values()))
            ):
                rows.append(
                    Row(f"Why {self._phi_label} = 0", "no repertoire of alternatives")
                )
        return tuple(rows)

    def to_pandas(self) -> pd.DataFrame:
        # IIT 4.0: ces carries .distinctions and .relations.
        # IIT 3.0: ces is the Distinctions sequence itself (no relations).
        distinctions = getattr(self.ces, "distinctions", self.ces)
        relations = getattr(self.ces, "relations", None)
        sum_phi_r = float(relations.sum_phi()) if relations is not None else float("nan")
        sum_phi_d = float(distinctions.sum_phi())
        # ``phi`` is φₛ; ``big_phi`` is Φ. See the class Notes.
        try:
            big_phi = self.big_phi
        except AttributeError:
            big_phi = float("nan")
        return pd.DataFrame(
            [
                {
                    "phi": float(self.sia.phi),
                    "normalized_phi": float(
                        getattr(self.sia, "normalized_phi", float("nan"))
                    ),
                    "big_phi": big_phi,
                    "n_distinctions": len(distinctions),
                    "sum_phi_d": sum_phi_d,
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
        The state of the substrate's nodes. When the bounds' maximum micro
        grain (``max_update_grain ** max_depth``) exceeds 1, a sequence of
        that many micro states (oldest first) instead — see
        :func:`pyphi.macro.complexes`.
    subset : optional
        Node indices of the candidate system; ``None`` uses the whole
        substrate. Incompatible with ``grains``.
    formalism : str or None, optional
        Leave unset to compute IIT under the active configuration. Pass an
        earlier version's name (``"IIT_4_0_2023"`` or ``"IIT_3_0"``) only to
        reproduce published results; it applies to this call only. The name
        a preset is exported under (``"iit3"``) and the preset itself
        (``pyphi.iit3``) select the same version. See
        :doc:`/howto/earlier-versions`.
    compute : optional
        ``None`` returns an :class:`Analysis` bundle; ``"sia"``, ``"ces"``,
        or ``"distinctions"`` returns the raw result object; a callable
        returns ``compute(system)``. Incompatible with ``grains``.
        ``"distinctions"`` skips the system-partition search, which is the
        whole cost of a cause-effect structure over a sparse substrate; see
        :meth:`~pyphi.system.System.distinctions`.
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
        ``"sia"``, ``"ces"``, ``"distinctions"``, a callable, or ``None``; if
        ``grains`` is not ``True``, a :class:`~pyphi.macro.SearchBounds`, or
        ``None``; if ``grains`` is combined with ``subset`` or ``compute``;
        or if ``parallel_kwargs`` is given without ``grains``.
    """
    formalism = presets.canonical(formalism)

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
            elif compute == "distinctions":
                result = system.distinctions(congruent=True)
            elif compute is not None:
                raise ValueError(
                    f"unknown compute: {compute!r}; expected 'sia', 'ces', "
                    "'distinctions', a callable, or None for the full "
                    "Analysis bundle"
                )
            else:
                ces = system.ces()
                sia = getattr(ces, "sia", None)
                if sia is None:
                    sia = system.sia()
                result = Analysis(system=system, sia=sia, ces=ces)
    return result
