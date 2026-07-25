"""Compare what a shard costs against what the cost model charged it.

Runs shards through the campaign's own execution path and records, per
shard: the planned unit charge, the partition evaluations actually
performed, CPU seconds, and cache hit/miss/eviction counts. The two
diagnostics separate the two ways the model can be wrong —

* ``evaluations / units`` differing between shard forms means the model
  miscounts operations;
* ``cpu_s / evaluations`` differing means it counts operations correctly
  but their cost is not constant.

Three arms:

``plan``
    Every shard of a real plan, both payload kinds as the ladder produced
    them. Reproduces the production comparison, mechanism size and payload
    kind confounded exactly as they are there.
``payload``
    One mechanism run as a ``mechanisms`` shard and as ``purview_range``
    shards over the same purviews. Isolates the payload path.
``multiplicity``
    A fixed unit total carried by K distinct mechanisms, for a range of K.
    Isolates how many distinct mechanisms a shard packs from how large they
    are.

Usage
-----
    uv run python experiments/units_runtime_model/measure_shard_cost.py \
        --units 16 --arm plan --units-per-job 2e5 --seed 1
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
from measure_pair_cost import ring_ising  # type: ignore[import-not-found]
from measure_pair_cost import unique_path  # type: ignore[import-not-found]

RESULTS = Path(__file__).parent / "results"


class Instrument:
    """Count partition evaluations and cache traffic around a shard run."""

    def __init__(self) -> None:
        from pyphi.formalism import queries

        self._queries = queries
        self._real = queries.evaluate_partition
        self.evaluations = 0

    def __enter__(self) -> Instrument:
        from pyphi.cache import registry

        registry.clear_all()

        def counting(*args: Any, **kwargs: Any) -> Any:
            self.evaluations += 1
            return self._real(*args, **kwargs)

        self._queries.evaluate_partition = counting
        self.wall = time.perf_counter()
        self.cpu = time.process_time()
        return self

    def __exit__(self, *exc: Any) -> None:
        from pyphi.cache import registry

        self.cpu_s = time.process_time() - self.cpu
        self.wall_s = time.perf_counter() - self.wall
        self._queries.evaluate_partition = self._real
        infos = registry.info().values()
        self.hits = sum(i.hits for i in infos)
        self.misses = sum(i.misses for i in infos)
        self.evictions = sum(i.evictions for i in infos)

    def row(self) -> dict:
        return {
            "cpu_s": self.cpu_s,
            "wall_s": self.wall_s,
            "evaluations": self.evaluations,
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "cache_evictions": self.evictions,
        }


def run_spec(
    spec: Any,
    substrate: Any,
    system: Any,
    scope: Any,
    formalism: str,
    ceiling_bytes: int | None = None,
) -> dict:
    import psutil

    from pyphi.campaign import CESShardTask
    from pyphi.campaign.runner import _run_ces_shard

    task = CESShardTask(
        task_id=0,
        kind="ces_shard",
        substrate_label="s",
        state=tuple(system.state),
        subset=tuple(system.node_indices),
        scope=scope,
        config_overrides={"maximum_cache_memory_bytes": ceiling_bytes},
        formalism=formalism,
        spec=spec,
        ordering=None,
    )
    process = psutil.Process()
    rss_before = process.memory_info().rss
    with Instrument() as instrument:
        entries, failed = _run_ces_shard(task, {"s": substrate})
    row = instrument.row()
    row.update(
        ceiling_bytes=ceiling_bytes,
        rss_before=rss_before,
        rss_after=process.memory_info().rss,
    )
    row.update(
        payload_kind=spec.payload_kind,
        units=spec.units,
        n_mechanisms=len(spec.mechanisms) if spec.mechanisms else 1,
        n_purviews=len(spec.purviews) if spec.purviews else None,
        memory_bytes=spec.memory_bytes,
        n_entries=len(entries),
        failed=failed,
    )
    return row


def scoped(purview_order: int, mechanism_order: int) -> Any:
    from pyphi.campaign.scope import AxisScope
    from pyphi.campaign.scope import CESScope

    return CESScope(
        mechanisms=AxisScope(max_order=mechanism_order),
        cause_purviews=AxisScope(max_order=purview_order),
        effect_purviews=AxisScope(max_order=purview_order),
    )


def arm_plan(
    system: Any,
    substrate: Any,
    scope: Any,
    formalism: str,
    units_per_job: float,
    max_shards: int,
    rng: Any,
) -> list[dict]:
    """Run every shard the ladder produces (a sample, if there are many)."""
    from pyphi.campaign.shards import plan_ces_shards

    specs = plan_ces_shards(system, scope, units_per_job)
    by_kind: dict[str, list] = {}
    for spec in specs:
        by_kind.setdefault(spec.payload_kind, []).append(spec)
    print({k: len(v) for k, v in by_kind.items()})
    chosen: list = []
    for kind, group in sorted(by_kind.items()):
        take = min(max_shards, len(group))
        picks = rng.choice(len(group), size=take, replace=False)
        chosen.extend(group[int(i)] for i in sorted(picks))
        print(f"  {kind}: running {take} of {len(group)}")
    rows = []
    for spec in chosen:
        row = run_spec(spec, substrate, system, scope, formalism)
        row["arm"] = "plan"
        rows.append(row)
        print(
            f"  {row['payload_kind']:>16} units={row['units']:>12,.0f} "
            f"evals={row['evaluations']:>10,} cpu={row['cpu_s']:>8.2f}s "
            f"us/unit={row['cpu_s'] / row['units'] * 1e6:>7.2f}"
        )
    return rows


def arm_payload(
    system: Any, substrate: Any, scope: Any, formalism: str, mechanism_order: int
) -> list[dict]:
    """The same mechanism as one ``mechanisms`` shard and as purview ranges."""
    from pyphi.campaign.shards import ShardSpec
    from pyphi.cost import mechanism_workloads
    from pyphi.direction import Direction

    workloads = mechanism_workloads(
        substrate, subset=system.node_indices, scope=scope, limit=10**9
    )
    mechanism = max(
        (m for m in workloads if len(m) == mechanism_order),
        key=lambda m: workloads[m].units,
    )
    workload = workloads[mechanism]
    rows = []
    row = run_spec(
        ShardSpec(
            payload_kind="mechanisms",
            mechanisms=(mechanism,),
            units=float(workload.units),
        ),
        substrate,
        system,
        scope,
        formalism,
    )
    row.update(arm="payload", mechanism=list(mechanism))
    rows.append(row)
    for direction in (Direction.CAUSE, Direction.EFFECT):
        axis = scope.purview_axis(direction, mechanism)
        purviews = tuple(
            axis.select(
                system.potential_purviews(
                    direction, mechanism, max_order=axis.order_bound()
                )
            )
        )
        if not purviews:
            continue
        from pyphi.cost import partition_sweep_count

        units = sum(1 + partition_sweep_count(len(mechanism), len(p)) for p in purviews)
        row = run_spec(
            ShardSpec(
                payload_kind="purview_range",
                mechanism=mechanism,
                direction=direction.name,
                purviews=purviews,
                units=float(units),
            ),
            substrate,
            system,
            scope,
            formalism,
        )
        row.update(arm="payload", mechanism=list(mechanism), direction=direction.name)
        rows.append(row)
    for row in rows:
        print(
            f"  {row['payload_kind']:>16} {row.get('direction', 'both'):>6} "
            f"units={row['units']:>12,.0f} evals={row['evaluations']:>10,} "
            f"cpu={row['cpu_s']:>8.2f}s us/unit={row['cpu_s'] / row['units'] * 1e6:>7.2f}"
        )
    return rows


def arm_size(
    system: Any,
    substrate: Any,
    scope: Any,
    formalism: str,
    targets: list[float],
    rng: Any,
) -> list[dict]:
    """The same shard composition at increasing unit totals.

    Production shards are two orders of magnitude larger than anything the
    other arms reach. Cost per unit against shard size catches anything that
    degrades as one process accumulates work — cache growth, the live-object
    count the cyclic collector rescans, retained tie sets — none of which a
    small shard exposes.
    """
    from pyphi.campaign.shards import ShardSpec

    pool = [m for order in (3, 4, 5, 6) for m in _windows(system.size, order)]
    rng.shuffle(pool)
    charges = [(m, mechanism_units(system, scope, m)) for m in pool]
    rows = []
    for target in sorted(targets):
        packed: list = []
        total = 0.0
        for mechanism, units in charges:
            if total >= target:
                break
            packed.append(mechanism)
            total += units
        if total < target * 0.5:
            print(f"  target {target:,.0f}: only {total:,.0f} units available, skipped")
            continue
        row = run_spec(
            ShardSpec(
                payload_kind="mechanisms", mechanisms=tuple(packed), units=float(total)
            ),
            substrate,
            system,
            scope,
            formalism,
        )
        row.update(arm="size", target_units=target)
        rows.append(row)
        print(
            f"  {len(packed):>4} mechanisms units={total:>12,.0f} "
            f"cpu={row['cpu_s']:>8.1f}s us/unit={row['cpu_s'] / total * 1e6:>7.2f} "
            f"miss/unit={row['cache_misses'] / total:>6.3f} "
            f"evict={row['cache_evictions']:>8,} rss={row['rss_after'] >> 20:>5}MiB"
        )
    return rows


def arm_multiplicity(
    system: Any,
    substrate: Any,
    scope: Any,
    formalism: str,
    target_units: float,
    rng: Any,
) -> list[dict]:
    """A fixed unit total spread over increasingly many distinct mechanisms.

    Shards are built by taking mechanisms of one order until the unit total
    is reached, for each of several orders. A high order reaches the target
    with few mechanisms, a low order needs many, so cost per unit against
    mechanism count separates packing multiplicity from mechanism size.
    """
    from pyphi.campaign.shards import ShardSpec
    from pyphi.cost import mechanism_workloads

    workloads = mechanism_workloads(
        substrate, subset=system.node_indices, scope=scope, limit=10**9
    )
    by_order: dict[int, list] = {}
    for mechanism, workload in workloads.items():
        by_order.setdefault(len(mechanism), []).append((mechanism, workload.units))
    rows = []
    for order in sorted(by_order):
        pool = sorted(by_order[order])
        rng.shuffle(pool)
        packed: list = []
        total = 0.0
        for mechanism, units in pool:
            if total >= target_units:
                break
            packed.append(mechanism)
            total += units
        if total < target_units * 0.5:
            print(f"  order {order}: only {total:,.0f} units available, skipped")
            continue
        row = run_spec(
            ShardSpec(
                payload_kind="mechanisms",
                mechanisms=tuple(packed),
                units=float(total),
            ),
            substrate,
            system,
            scope,
            formalism,
        )
        row.update(arm="multiplicity", mechanism_order=order)
        rows.append(row)
        print(
            f"  order {order}: {len(packed):>4} mechanisms units={total:>12,.0f} "
            f"evals={row['evaluations']:>10,} cpu={row['cpu_s']:>8.2f}s "
            f"us/unit={row['cpu_s'] / total * 1e6:>7.2f} "
            f"us/eval={row['cpu_s'] / max(1, row['evaluations']) * 1e6:>7.2f}"
        )
    return rows


def mechanism_units(system: Any, scope: Any, mechanism: tuple) -> float:
    """One mechanism's unit charge, without walking the whole scope.

    The same sum :func:`pyphi.cost.mechanism_workloads` computes, for a
    single mechanism — a whole-scope walk over a 21-unit substrate costs
    minutes and these arms need a handful of mechanisms.
    """
    from pyphi.cost import partition_sweep_count
    from pyphi.direction import Direction

    total = 0.0
    for direction in (Direction.CAUSE, Direction.EFFECT):
        axis = scope.purview_axis(direction, mechanism)
        for purview in axis.select(
            system.potential_purviews(direction, mechanism, max_order=axis.order_bound())
        ):
            total += 1 + partition_sweep_count(len(mechanism), len(purview))
    return total


def _windows(units: int, order: int) -> list[tuple]:
    """Every contiguous window of ``order`` units on the ring.

    A scattered mechanism of order 5 or more has no order-3 purview on a
    radius-2 ring — no purview that small reaches all of its units, so
    ``potential_purviews`` prunes every candidate and the mechanism carries
    no work at all. Contiguous windows are the mechanisms a low purview cap
    actually admits, which is what makes the production scope's locality
    constraint necessary.
    """
    return [tuple(sorted((i + d) % units for d in range(order))) for i in range(units)]


def _pack_mechanisms(
    system: Any, scope: Any, pool: list[tuple], target: float
) -> tuple[tuple, float]:
    packed: list = []
    total = 0.0
    for mechanism in pool:
        if total >= target:
            break
        packed.append(mechanism)
        total += mechanism_units(system, scope, mechanism)
    return tuple(packed), total


def _purview_range(
    system: Any, scope: Any, mechanism: tuple, target: float
) -> tuple[Any, float]:
    from pyphi.cost import partition_sweep_count
    from pyphi.direction import Direction

    axis = scope.purview_axis(Direction.EFFECT, mechanism)
    purviews = list(
        axis.select(
            system.potential_purviews(
                Direction.EFFECT, mechanism, max_order=axis.order_bound()
            )
        )
    )
    kept: list = []
    total = 0.0
    for purview in purviews:
        if total >= target:
            break
        kept.append(purview)
        total += 1 + partition_sweep_count(len(mechanism), len(purview))
    return tuple(kept), total


def arm_cache(
    system: Any,
    substrate: Any,
    scope: Any,
    formalism: str,
    target_units: float,
    pack_order: int,
    split_order: int,
    ceilings: list[int | None],
    rng: Any,
) -> list[dict]:
    """Both shard forms at matched units, each under several cache ceilings.

    The forms are matched on the planner's unit charge, so any difference in
    cost per unit is the model's error. Running each under an unlimited and
    a binding ceiling separates compulsory miss cost — which a shard packing
    many distinct mechanisms pays whatever the ceiling — from starvation.
    """
    from pyphi.campaign.shards import ShardSpec
    from pyphi.direction import Direction

    pool = _windows(system.size, pack_order)
    rng.shuffle(pool)
    big = max(
        _windows(system.size, split_order),
        key=lambda m: mechanism_units(system, scope, m),
    )
    # Match the two forms on the unit charge, capped by whichever has less
    # work available: an unmatched comparison would confound shard form with
    # shard size.
    available = min(
        sum(mechanism_units(system, scope, m) for m in pool),
        _purview_range(system, scope, big, float("inf"))[1],
    )
    target = min(target_units, available)
    packed, packed_units = _pack_mechanisms(system, scope, pool, target)
    purviews, range_units = _purview_range(system, scope, big, target)
    if not purviews or not packed:
        raise SystemExit(
            f"no work: {len(packed)} packed mechanisms, {len(purviews)} purviews "
            f"for {big}; raise --purview-order or lower --split-order"
        )
    print(
        f"  mechanisms: {len(packed)} of order {pack_order}, {packed_units:,.0f} units\n"
        f"  purview_range: {big} over {len(purviews)} purviews, "
        f"{range_units:,.0f} units"
    )
    specs = [
        ShardSpec(
            payload_kind="mechanisms",
            mechanisms=packed,
            units=float(packed_units),
        ),
        ShardSpec(
            payload_kind="purview_range",
            mechanism=big,
            direction=Direction.EFFECT.name,
            purviews=purviews,
            units=float(range_units),
        ),
    ]
    rows = []
    for ceiling in ceilings:
        for spec in specs:
            row = run_spec(spec, substrate, system, scope, formalism, ceiling)
            row.update(arm="cache", pack_order=pack_order, split_order=split_order)
            rows.append(row)
            per_unit = row["cpu_s"] / max(1.0, row["units"]) * 1e6
            traffic = row["cache_hits"] + row["cache_misses"]
            print(
                f"  ceiling={'none' if ceiling is None else f'{ceiling >> 20}MiB':>8} "
                f"{row['payload_kind']:>14} units={row['units']:>11,.0f} "
                f"cpu={row['cpu_s']:>8.1f}s us/unit={per_unit:>7.2f} "
                f"miss={row['cache_misses'] / max(1, traffic):>6.2%} "
                f"miss/unit={row['cache_misses'] / row['units']:>6.3f} "
                f"evict={row['cache_evictions']:>8,} "
                f"rss={row['rss_after'] >> 20:>5}MiB"
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=int, default=16)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--purview-order", type=int, default=3)
    parser.add_argument("--mechanism-order", type=int, default=6)
    parser.add_argument(
        "--arm",
        choices=("plan", "payload", "multiplicity", "cache", "size"),
        default="multiplicity",
    )
    parser.add_argument("--pack-order", type=int, default=4)
    parser.add_argument("--size-targets", type=float, nargs="*", default=[1e6, 5e6, 2e7])
    parser.add_argument("--split-order", type=int, default=7)
    parser.add_argument(
        "--ceiling-mib",
        type=int,
        nargs="*",
        default=[],
        help="Cache ceilings in MiB to run each form under; an unlimited arm "
        "is always run first.",
    )
    parser.add_argument("--units-per-job", type=float, default=2e5)
    parser.add_argument("--target-units", type=float, default=2e5)
    parser.add_argument("--max-shards", type=int, default=4)
    parser.add_argument("--formalism", default="IIT_4_0_2026")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-label", default="")
    args = parser.parse_args()

    from pyphi.conf import config
    from pyphi.conf import presets
    from pyphi.system import System

    rng = np.random.default_rng(args.seed)
    substrate = ring_ising(args.units, args.radius, args.temperature)
    scope = scoped(args.purview_order, args.mechanism_order)

    with config.override(
        **presets.by_name[args.formalism], progress_bars=False, parallel=False
    ):
        system = System.from_substrate(substrate, (0,) * args.units)
        if args.arm == "plan":
            rows = arm_plan(
                system,
                substrate,
                scope,
                args.formalism,
                args.units_per_job,
                args.max_shards,
                rng,
            )
        elif args.arm == "payload":
            rows = arm_payload(
                system, substrate, scope, args.formalism, args.mechanism_order
            )
        elif args.arm == "size":
            rows = arm_size(
                system, substrate, scope, args.formalism, args.size_targets, rng
            )
        elif args.arm == "cache":
            rows = arm_cache(
                system,
                substrate,
                scope,
                args.formalism,
                args.target_units,
                args.pack_order,
                args.split_order,
                [None, *(m << 20 for m in args.ceiling_mib)],
                rng,
            )
        else:
            rows = arm_multiplicity(
                system, substrate, scope, args.formalism, args.target_units, rng
            )

    RESULTS.mkdir(parents=True, exist_ok=True)
    label = f"_{args.run_label}" if args.run_label else ""
    name = (
        f"shard_cost_{args.arm}_n{args.units}_o{args.purview_order}"
        f"_m{args.mechanism_order}_seed{args.seed}{label}.json"
    )
    path = unique_path(RESULTS / name)
    path.write_text(
        json.dumps(
            {
                "params": vars(args),
                "platform": {
                    "machine": platform.machine(),
                    "python": platform.python_version(),
                },
                "rows": rows,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
