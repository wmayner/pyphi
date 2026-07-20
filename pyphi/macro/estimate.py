"""Analytic pre-flight for the bounded intrinsic-unit search.

Counts the candidate systems a :func:`pyphi.macro.complexes` sweep would
evaluate, without constructing macro TPMs or computing φₛ. The counting
walk mirrors the search's own control flow and drives the search's own
enumeration functions with lightweight stand-in units, so the counts
reflect the implementation exactly.

Because the search prunes adaptively — only decompositions that pass the
intrinsic-unit criteria spawn mapped variants — counts beyond the first
macroing level assume every candidate passes: an exact worst case. At
``max_depth=0`` no judgment happens and the enumeration is exact. Even
then, a candidate whose state is unreachable under its own TPM is
discarded at run time, which the enumeration cannot predict, so counts
bound the candidates *enumerated*, and ``ComplexesResult.records`` can be
smaller.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from pyphi.cost import _PARTITION_COUNT_CAP
from pyphi.cost import _Counter
from pyphi.cost import _LimitReached
from pyphi.cost import _partition_counts
from pyphi.display import Description
from pyphi.display import Displayable
from pyphi.display import Row
from pyphi.display import Section
from pyphi.macro.search import SearchBounds
from pyphi.macro.search import _apportionments
from pyphi.macro.search import _assemble_systems
from pyphi.macro.search import _decompositions
from pyphi.macro.search import _require_iit4
from pyphi.macro.search import candidate_mappings
from pyphi.models.pandas import ToPandasMixin


@dataclass(frozen=True)
class _StandIn:
    """One unit shape standing for every mapped variant that shares it.

    Carries exactly the attributes the search's enumeration functions
    read, plus the number of concrete units it represents.
    """

    micro_constituents: tuple[int, ...]
    micro_grain: int
    background_apportionment: tuple[int, ...]
    multiplicity: int = 1


def _shape_key(unit):
    return (
        unit.micro_constituents,
        unit.micro_grain,
        unit.background_apportionment,
    )


def _combo_count(combo) -> int:
    total = 1
    for unit in combo:
        total *= unit.multiplicity
    return total


def _merged(units) -> list[_StandIn]:
    """Merge stand-ins sharing a shape key, summing multiplicities."""
    acc: dict[tuple, _StandIn] = {}
    for unit in units:
        key = _shape_key(unit)
        prior = acc.get(key)
        if prior is None:
            acc[key] = unit
        else:
            acc[key] = _StandIn(
                unit.micro_constituents,
                unit.micro_grain,
                unit.background_apportionment,
                prior.multiplicity + unit.multiplicity,
            )
    return list(acc.values())


def _variant_standins(V, W, bounds: SearchBounds, base: int) -> list[_StandIn]:
    """Stand-ins for the mapped and grained variants ``base`` concrete
    decompositions of shape ``V`` would spawn, mirroring the search's
    variant construction (a single-constituent decomposition is pure
    grain raising, so its variants start at update grain 2)."""
    size = len(V)
    grain = V[0].micro_grain
    footprint = tuple(sorted(set().union(*(set(u.micro_constituents) for u in V))))
    min_grain = 2 if size == 1 else 1
    out = []
    for update_grain in range(min_grain, bounds.max_update_grain + 1):
        num_mappings = len(candidate_mappings(size, update_grain, bounds))
        if num_mappings:
            out.append(_StandIn(footprint, update_grain * grain, W, base * num_mappings))
    return out


@dataclass(frozen=True)
class SearchEstimate(Displayable, ToPandasMixin):
    """The size and shape of a grain search, before running it.

    All quantities are exact worst cases under the assumption that every
    judged decomposition passes the intrinsic-unit criteria; the search
    itself can only do less. See the module docstring for what "exact"
    can and cannot promise.

    Attributes
    ----------
    n : int
        Micro universe size the estimate was computed for.
    bounds : SearchBounds
        The bounds the estimate describes.
    judgments_by_level : tuple[int, ...]
        Candidate decompositions judged per macroing level (the first
        level's count is exact; deeper levels are worst-case).
    worst_case_pool_by_level : tuple[int, ...]
        Concrete units in the pool after each level, all-pass.
    assemblies_upper_bound : int
        Candidate systems the final sweep assembles (Eq. 18).
    distinct_systems_upper_bound : int
        Deduplicated candidate systems across judgment candidates,
        judgment competitors, and the final sweep — the number to compare
        with ``len(ComplexesResult.records)``.
    systems_by_unit_count : dict[int, int]
        Final-sweep candidates bucketed by macro unit count m.
    partitions_by_unit_count : dict[int, int]
        Partitions per system irreducibility analysis at each m, under
        the active system partition scheme.
    partition_sweeps_upper_bound : int
        Σ over m of systems × partitions — the SIA-cost axis.
    construction_keys_upper_bound : int
        Distinct (footprint, grain) construction keys — the Θ(τ·4ⁿ)
        construction-cost axis.
    is_exact : bool
        Whether the enumeration is exact rather than an upper bound
        (``max_depth == 0`` and not truncated).
    truncated : bool
        The counting hit its ``limit``; counts are then lower bounds of
        the bound.
    partitions_capped : bool
        At least one bucket has a macro unit count m above the
        partition-count cap and so has no partition count;
        ``partition_sweeps_upper_bound`` then covers only the counted
        buckets. Independent of ``truncated`` and ``is_exact``.
    """

    n: int
    bounds: SearchBounds
    judgments_by_level: tuple[int, ...]
    worst_case_pool_by_level: tuple[int, ...]
    assemblies_upper_bound: int
    distinct_systems_upper_bound: int
    systems_by_unit_count: dict[int, int]
    partitions_by_unit_count: dict[int, int]
    partition_sweeps_upper_bound: int
    construction_keys_upper_bound: int
    is_exact: bool
    truncated: bool
    partitions_capped: bool

    def _qualifier(self) -> str:
        if self.truncated:
            return "≥"
        return "=" if self.is_exact else "≤"

    def _pandas_record(self) -> dict:
        return {
            "n": self.n,
            "distinct_systems_upper_bound": self.distinct_systems_upper_bound,
            "assemblies_upper_bound": self.assemblies_upper_bound,
            "partition_sweeps_upper_bound": self.partition_sweeps_upper_bound,
            "construction_keys_upper_bound": self.construction_keys_upper_bound,
            "is_exact": self.is_exact,
            "truncated": self.truncated,
            "partitions_capped": self.partitions_capped,
        }

    def _describe(self, verbosity: int) -> Description:  # noqa: ARG002
        q = self._qualifier()
        sweeps_q = "≥" if self.partitions_capped else q
        rows = [
            Row("Candidate systems", f"{q} {self.distinct_systems_upper_bound}"),
            Row("Assembly sweep", f"{q} {self.assemblies_upper_bound}"),
            Row("Partition sweeps", f"{sweeps_q} {self.partition_sweeps_upper_bound}"),
            Row("Construction keys", f"{q} {self.construction_keys_upper_bound}"),
            Row("Judgments by level", str(self.judgments_by_level)),
            Row("Pool by level", str(self.worst_case_pool_by_level)),
            Row("Exact", self.is_exact),
            Row("Truncated", self.truncated),
        ]
        bucket_rows = tuple(
            Row(
                f"m = {m}",
                f"{self.systems_by_unit_count[m]} systems × "
                f"{self.partitions_by_unit_count.get(m, '>cap')} partitions",
            )
            for m in sorted(self.systems_by_unit_count)
        )
        sections = [Section(rows=tuple(rows))]
        if bucket_rows:
            sections.append(Section(label="Systems by unit count", rows=bucket_rows))
        return Description(
            title="SearchEstimate",
            subtitle=f"{self.n} micro units",
            sections=tuple(sections),
            compact=(
                f"SearchEstimate(n={self.n}, systems "
                f"{q} {self.distinct_systems_upper_bound})"
            ),
        )


def estimate_search(
    bounds: SearchBounds, n: int, limit: int = 1_000_000
) -> SearchEstimate:
    """Count the candidate systems a grain search over ``n`` micro units
    would evaluate under ``bounds``, without running it.

    Notes
    -----
    ``limit`` bounds the *reported counts* and stops the walk at
    enumeration-phase granularity (``truncated=True``): one
    ``_assemble_systems`` call completes before the limit is checked, so
    wall time is not strictly bounded by ``limit``.

    ``mappings="EXHAUSTIVE"`` above ``exhaustive_cap`` raises a
    ``ValueError``. The search itself raises only when such a
    decomposition passes judgment; the estimate assumes every
    decomposition passes, so it raises unconditionally.
    """
    _require_iit4()
    counter = _Counter(limit)
    truncated = False

    # Global registry of distinct unit-combos (the memo's worst case).
    # A key re-recorded with a larger count supersedes: concrete variants
    # only accumulate, so the later set contains the earlier one.
    combos: dict[tuple, int] = {}

    def record_combo(combo) -> None:
        key = tuple(sorted(_shape_key(u) for u in combo))
        count = _combo_count(combo)
        prior = combos.get(key, 0)
        if count > prior:
            counter.charge(count - prior)
            combos[key] = count

    indices = tuple(range(n))
    pool: list[_StandIn] = [_StandIn((i,), 1, ()) for i in indices]

    judgments_by_level: list[int] = []
    pool_by_level: list[int] = []
    # Shape-level judgment ledger: concrete decompositions already judged
    # per (decomposition shape, W). A shape re-encountered after its
    # members gained variants is re-judged for the new concretes only.
    judged: dict[tuple, int] = {}
    min_size = 1 if bounds.max_update_grain > 1 else 2
    max_size = min(n, bounds.max_constituents)
    assemblies = 0
    sweep_buckets: dict[int, int] = {}

    judged_this_level = 0
    level_open = False
    try:
        for unit in pool:
            record_combo((unit,))
        for _level in range(bounds.max_depth):
            pool_prev = tuple(pool)
            judged_this_level = 0
            level_open = True
            emitted_any = False
            for size in range(min_size, max_size + 1):
                pool_at_class_start = tuple(pool)
                new_units: list[_StandIn] = []
                for footprint in itertools.combinations(indices, size):
                    for V in _decompositions(
                        footprint,
                        pool_prev,
                        allow_singleton=bounds.max_update_grain > 1,
                    ):
                        inherited = set().union(
                            *(set(u.background_apportionment) for u in V)
                        )
                        for W in _apportionments(n, footprint, inherited, bounds):
                            key = (
                                tuple(sorted(_shape_key(u) for u in V)),
                                W,
                            )
                            base = _combo_count(V)
                            delta = base - judged.get(key, 0)
                            if delta <= 0:
                                continue
                            judged[key] = base
                            counter.charge(delta)
                            judged_this_level += delta
                            record_combo(V)
                            fp = set(footprint)
                            allowed = set(W)
                            members = [
                                u
                                for u in pool_at_class_start
                                if set(u.micro_constituents) < fp
                                and set(u.background_apportionment) <= allowed
                            ]
                            for combo in _assemble_systems(
                                members, bounds.max_background
                            ):
                                record_combo(combo)
                            new_units.extend(_variant_standins(V, W, bounds, delta))
                pool = _merged([*pool, *new_units])
                emitted_any = emitted_any or bool(new_units)
            judgments_by_level.append(judged_this_level)
            pool_by_level.append(sum(u.multiplicity for u in pool))
            level_open = False
            if not emitted_any:
                break
        for combo in _assemble_systems(pool, bounds.max_background):
            record_combo(combo)
            count = _combo_count(combo)
            counter.charge(count)
            assemblies += count
            m = len(combo)
            sweep_buckets[m] = sweep_buckets.get(m, 0) + count
    except _LimitReached:
        truncated = True
        # Flush the partial level's tallies so a mid-level cutoff still
        # reports the judgments and pool it charged before stopping.
        if level_open:
            judgments_by_level.append(judged_this_level)
            pool_by_level.append(sum(u.multiplicity for u in pool))

    partitions = _partition_counts(sorted(sweep_buckets))
    partitions_capped = any(m > _PARTITION_COUNT_CAP for m in sweep_buckets)
    partition_sweeps = sum(
        count * partitions[m] for m, count in sweep_buckets.items() if m in partitions
    )
    construction_keys = len({(u.micro_constituents, u.micro_grain) for u in pool})
    return SearchEstimate(
        n=n,
        bounds=bounds,
        judgments_by_level=tuple(judgments_by_level),
        worst_case_pool_by_level=tuple(pool_by_level),
        assemblies_upper_bound=assemblies,
        distinct_systems_upper_bound=sum(combos.values()),
        systems_by_unit_count=sweep_buckets,
        partitions_by_unit_count=partitions,
        partition_sweeps_upper_bound=partition_sweeps,
        construction_keys_upper_bound=construction_keys,
        is_exact=bounds.max_depth == 0 and not truncated,
        truncated=truncated,
        partitions_capped=partitions_capped,
    )
