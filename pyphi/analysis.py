"""One high-level entry point for IIT analysis.

``analyze`` takes a substrate and a state, builds the candidate system, runs
the analysis, and returns an :class:`Analysis` — a small bundle exposing the
system irreducibility analysis, the cause-effect structure, and their scalar
values. A ``compute`` argument selects a cheaper or custom result instead of
the bundle; a ``grains`` argument runs the bounded intrinsic-unit search over
the whole substrate instead, returning its complexes.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Any

import pandas as pd

from pyphi import numerics
from pyphi.conf import presets
from pyphi.conf.formalism import IITConfig
from pyphi.display import FULL
from pyphi.display import Description
from pyphi.display import Displayable
from pyphi.display import Row
from pyphi.display import Section
from pyphi.display import system_phi_label
from pyphi.display.numbers import format_value
from pyphi.models.ces import CauseEffectStructure
from pyphi.relations import NullRelations
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
    :doc:`/howto/earlier-versions`): under IIT 3.0 the CES has no relations,
    the system-level quantity ``phi`` is that version's Φ, and ``big_phi`` is
    not defined.
    """

    system: System
    sia: Any
    ces: CauseEffectStructure

    @property
    def phi(self) -> float:
        """float: φₛ, the system integrated information."""
        return float(self.sia.phi)

    @property
    def formalism(self) -> str:
        """str: The version of IIT that produced this analysis:
        ``"IIT_4_0_2026"`` (the default), ``"IIT_4_0_2023"``, or ``"IIT_3_0"``.
        Read from the configuration snapshot the system irreducibility analysis
        carries."""
        return self.sia.config.formalism.iit.version

    @property
    def _phi_label(self) -> str:
        return system_phi_label(self.ces.config)

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
        return float(self.ces.big_phi)

    def _describe(self, verbosity: int) -> Description:
        # Reuse the cause-effect structure's flat rich card, which folds in
        # the embedded SIA at FULL verbosity.
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
        if verbosity < FULL and getattr(self.sia, "partition", None) is not None:
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

        rows = [Row("Units", self.sia._system_label())]
        if self.sia.current_state is not None:
            rows.append(Row("Current state", self.sia.current_state))
        state = getattr(self.sia, "system_state", None)
        if state is not None:
            if state.cause is not None:
                rows.append(Row("Specified cause state", state.cause.state))
            if state.effect is not None:
                rows.append(Row("Specified effect state", state.effect.state))
        rows.append(Row("MIP", concise_partition(self.sia.partition)))
        # A null result records why φₛ is zero without evaluating partitions.
        rows.extend(
            Row(f"Why {self._phi_label} = 0", reason.description)
            for reason in self.sia.reasons or ()
        )
        # ii(s) enters φₛ only where the intrinsic-information requirement
        # applies (Mayner et al. 2026, Eq. 23); IIT 4.0 as published in 2023
        # does not use it.
        ii = getattr(self.sia, "intrinsic_information", None)
        if ii is not None and self.sia._applies_requirement():
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
        """A one-row DataFrame that identifies the analysis and gives its
        values.

        The identifiers are ``formalism`` (the version name, as in a sweep's
        rows), ``subset`` (the system's sorted unit indices) and
        ``current_state`` (the system's state), as in the SIA's own row. The
        values are φₛ (``phi``), ``normalized_phi``, Φ (``big_phi``),
        ``n_distinctions``, ``sum_phi_d`` and ``sum_phi_r``, the same columns
        as a sweep's rows for ``compute="ces"`` or ``compute=None``.
        """
        record = {
            "formalism": self.formalism,
            "subset": tuple(int(i) for i in self.system.node_indices),
            "current_state": tuple(int(x) for x in self.system.proper_state),
            **_structure_record(self.sia, self.ces),
        }
        return pd.DataFrame([record])


def _structure_record(sia: Any, ces: CauseEffectStructure) -> dict[str, Any]:
    """The scalar summary of one system's analysis, keyed by column name.

    IIT 3.0 defines no relations and no structure integrated information, so
    there ``sum_phi_r`` and ``big_phi`` are NaN rather than zero.
    """
    relations = ces.relations
    iit3 = system_phi_label(ces.config) == "Φ"
    return {
        "phi": float(sia.phi),
        "normalized_phi": float(getattr(sia, "normalized_phi", float("nan"))),
        "big_phi": float("nan") if iit3 else float(ces.big_phi),
        "n_distinctions": len(ces.distinctions),
        "sum_phi_d": float(ces.distinctions.sum_phi()),
        "sum_phi_r": (
            float("nan")
            if isinstance(relations, NullRelations)
            else float(relations.sum_phi())
        ),
    }


_COMPUTE_NAMES = ("sia", "ces", "distinctions")


def _check_compute(compute: Any) -> None:
    """Raise ``ValueError`` unless ``compute`` is in the shared vocabulary:
    ``None``, a name in ``_COMPUTE_NAMES``, or a callable."""
    if compute is None or callable(compute) or compute in _COMPUTE_NAMES:
        return
    raise ValueError(
        f"unknown compute: {compute!r}; expected 'sia', 'ces', 'distinctions', "
        "a callable, or None for the full Analysis bundle"
    )


def _run_compute(system: System, compute: Any) -> Any:
    """Compute what ``compute`` selects for ``system``; see :func:`analyze`."""
    _check_compute(compute)
    if callable(compute):
        return compute(system)
    if compute == "sia":
        return system.sia()
    if compute == "ces":
        return system.ces()
    if compute == "distinctions":
        return system.distinctions(congruent=True)
    ces = system.ces()
    return Analysis(system=system, sia=ces.sia, ces=ces)


def analyze(
    substrate: Any,
    state: tuple[int, ...],
    *,
    subset: Any = None,
    formalism: str | None = None,
    compute: Any = None,
    grains: Any = None,
    parallel: bool | None = None,
    progress: bool | None = None,
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
        (``pyphi.iit3``) select the same version; a modified copy of a preset
        raises ``ValueError``, since this argument only selects published
        versions (apply modified settings with ``pyphi.config.override`` and
        leave ``formalism`` unset). ``"IIT_4_0_2026"`` names the
        default; passing it applies that version's complete preset, which
        differs from leaving ``formalism`` unset only if the active
        configuration has been customized. See :doc:`/howto/earlier-versions`.
    compute : optional
        ``None`` returns an :class:`Analysis` bundle; ``"sia"``, ``"ces"``,
        or ``"distinctions"`` returns the raw result object; a callable
        returns ``compute(system)``. Incompatible with ``grains``.
        ``"distinctions"`` skips the system-partition search, which is the
        whole cost of a cause-effect structure over a sparse substrate; see
        :meth:`~pyphi.system.System.distinctions`.
    grains : optional
        ``None`` or ``False`` analyzes the single candidate system. ``True``
        runs the bounded intrinsic-unit search with default
        :class:`~pyphi.macro.SearchBounds`; a
        :class:`~pyphi.macro.SearchBounds` instance runs it with those
        bounds. The search returns its
        :class:`~pyphi.macro.ComplexesResult`.
    parallel : bool or None, optional
        ``None`` follows the configuration. ``True`` runs this call in
        parallel: a single system at the system-partition, purview and
        distinction levels, a grain search over its candidate systems.
        ``False`` runs it sequentially. Finer control is through the
        per-level options in ``config.infrastructure``; see
        :doc:`/howto/parallel`.
    progress : bool or None, optional
        Show progress bars; ``None`` follows
        ``config.infrastructure.progress_bars``.

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
        ``grains`` is not a bool, a :class:`~pyphi.macro.SearchBounds`, or
        ``None``; or if ``grains`` is combined with ``subset`` or
        ``compute``.
    """
    from pyphi.conf import config
    from pyphi.conf._helpers import parallel_overrides

    formalism = presets.canonical(formalism)
    _check_compute(compute)

    bounds: Any = None
    if grains is not None and grains is not False:
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
                f"grains must be True, False, or a SearchBounds instance; got {grains!r}"
            )

    from pyphi.conf._helpers import check_progress_flag

    progress = check_progress_flag(progress)
    progress_override = {} if progress is None else {"progress_bars": progress}
    with presets.use(formalism), config.override(**progress_override):
        if bounds is not None:
            from pyphi.macro.search import complexes as grain_complexes

            return grain_complexes(
                substrate, state, bounds=bounds, parallel=parallel, progress=progress
            )
        with config.override(**parallel_overrides(parallel)):
            return _run_compute(System(substrate, state, subset=subset), compute)
