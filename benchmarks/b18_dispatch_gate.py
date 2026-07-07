"""Measure the parallel dispatch gate across the below-chunksize regime.

Compares two implementations of ``LocalMapReduce._should_run_parallel``:
the chunksize-boundary gate (sequential whenever ``total <= chunksize``,
so the chunker's ``num_workers`` chunk-count floor never applies below
that boundary) and the chunk-count gate (parallel whenever the chunker
would produce more than one chunk, so any workload at or above
``sequential_threshold`` fans out). The study that decided between them
ran both gates across workload sizes spanning the chunksize boundary, at
the realistic per-item costs of six map-reduce sites:

- **relations** (chunksize 4096, threshold 1024): ``Relation``
  construction over candidate distinction combinations, exactly as
  :func:`pyphi.relations.all_relations` dispatches it. Per-item cost is
  on the order of microseconds (relation phi is lazy), so this level
  measures dispatch overhead against near-free items.
- **mech_partitions** (chunksize 4096, threshold 1024): one
  ``evaluate_partition`` call per mechanism partition, as in the
  mechanism MIP search. Per-item cost ~0.1 ms. The real site
  short-circuits on the first zero-phi partition; that is disabled here
  so workload size is controlled.
- **purviews** (chunksize 256, threshold 64): one ``find_mip`` call per
  purview, as in MICE search. Per-item cost ~10 ms.
- **sys_partitions** (chunksize 4096, threshold overridden to the
  proposed 64): one system-level ``evaluate_partition`` per system
  partition, as in the IIT 4.0 SIA. Per-item cost ~10 ms.
- **sys_partitions_shortcircuit**: the same map on a reducible system
  with short-circuiting enabled, bounding the parallel arm's wasted
  work when the sequential arm would exit at the first zero-phi
  partition.
- **complexes** (chunksize 64, threshold 16): one full SIA per candidate
  system, as in ``complexes()``. Per-item cost ~0.1-10 s.

Pool state: the loky executor is pre-warmed before the timed trials, so
the headline numbers are **warm-pool** (steady state). A separate
cold-pool section times the flipped gate's first call after a worker
shutdown, since below-chunksize workloads currently never pay that cost.

Item subsets are sampled with a seeded RNG and shared between the two
gate arms within a trial (paired measurements). Progress bars are off.

Usage::

    uv run python benchmarks/b18_dispatch_gate.py \
        [--seed 1810] [--trials 5] [--levels relations mech_partitions purviews] \
        [--run-label LABEL]

Results (raw per-trial wall times, medians, the seed, and machine info)
are written to ``benchmarks/b18_dispatch_gate_results/b18_dispatch_gate_
seed<seed>_trials<trials>[_<label>].json``; existing files are never
overwritten (a ``_v2``/``_v3`` suffix is added instead).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import multiprocessing
import platform
import random
import statistics
import time
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent / "b18_dispatch_gate_results"

# Workload sizes as fractions of the level's chunksize. Sizes at or below
# 1.0x differ between the gates (current: sequential; flipped: parallel);
# sizes above 1.0x are controls where both gates parallelize identically.
SIZE_FRACTIONS = (0.25, 0.5, 0.9, 1.0, 1.1, 2.0)

COLD_TRIALS = 3


def rule_ring(rule: int, n: int):
    """A ring of ``n`` binary cells following elementary CA ``rule``."""
    import pyphi
    from pyphi import Substrate

    tpm = np.zeros((2**n, n), dtype=int)
    for i in range(2**n):
        state = pyphi.convert.le_index2state(i, n)
        for j in range(n):
            left, self_, right = state[(j - 1) % n], state[j], state[(j + 1) % n]
            tpm[i, j] = (rule >> (left * 4 + self_ * 2 + right)) & 1
    cm = np.zeros((n, n), dtype=int)
    for j in range(n):
        cm[(j - 1) % n, j] = cm[j, j] = cm[(j + 1) % n, j] = 1
    return Substrate(tpm, cm=cm)


def chunksize_boundary_gate(self) -> bool:
    """The pre-change gate: sequential at or below one chunksize."""
    if self.total is None:
        return True
    if self.total < self.sequential_threshold:
        return False
    return not (self.chunksize and self.total <= self.chunksize)


def flipped_gate(self) -> bool:
    """The chunk-count gate: parallelize iff the chunker yields more than
    one chunk (the shipped behavior since this study landed)."""
    from pyphi.parallel import get_num_processes

    if self.total is None:
        return True
    if self.total < self.sequential_threshold:
        return False
    if not self.chunksize:
        return self.total > 1
    k = max(math.ceil(self.total / self.chunksize), get_num_processes())
    return min(k, self.total) > 1


GATES = {"current": chunksize_boundary_gate, "flipped": flipped_gate}


class gate:
    """Context manager installing a ``_should_run_parallel`` implementation.

    Both gate implementations are pinned here so the two arms stay
    comparable regardless of which one the library currently ships.
    """

    def __init__(self, which: str):
        self.which = which

    def __enter__(self):
        from pyphi.parallel.backends.local_process import LocalMapReduce

        self._original = LocalMapReduce._should_run_parallel
        LocalMapReduce._should_run_parallel = GATES[self.which]
        return self

    def __exit__(self, *_exc):
        from pyphi.parallel.backends.local_process import LocalMapReduce

        LocalMapReduce._should_run_parallel = self._original
        return False


def setup_relations():
    """Relation construction over real distinctions (all_relations site shape).

    Distinctions come from the mechanisms of size 1 and 2 of a dense Ising
    network (dense connectivity gives wide purviews, so the candidate
    combination stream is deep, as in large-CES relations enumeration).
    """
    import pyphi
    from pyphi import System
    from pyphi.relations import Relation
    from pyphi.relations import _combinations_with_nonempty_congruent_overlap
    from pyphi.relations import _relation_size_func
    from pyphi.substrate_generator import build_substrate

    n = 7
    weight_rng = np.random.default_rng(1810)
    weights = weight_rng.normal(1.0, 0.5, size=(n, n))
    substrate = build_substrate("ising", weights, temperature=0.25)
    system = System(substrate, (0,) * n)
    mechanisms = [c for k in (1, 2) for c in itertools.combinations(range(n), k)]
    distinctions = [d for d in map(system.distinction, mechanisms) if d]
    chunksize = pyphi.config.infrastructure.parallel_relation_evaluation["chunksize"]
    pool_size = int(2.5 * chunksize * max(SIZE_FRACTIONS))
    pool = list(
        itertools.islice(
            _combinations_with_nonempty_congruent_overlap(distinctions), pool_size
        )
    )
    size_func = _relation_size_func([d.purview_union for d in distinctions])

    def worker(combination):
        return Relation(distinctions[i] for i in combination)

    def run(items):
        from pyphi import conf
        from pyphi.parallel import map_reduce

        pkwargs = conf.parallel_kwargs(
            pyphi.config.infrastructure.parallel_relation_evaluation
        )
        return map_reduce(worker, items, size_func=size_func, **pkwargs)

    return {
        "pool": pool,
        "run": run,
        "option": "parallel_relation_evaluation",
        # Pin the boundary regime under study so the arms stay comparable
        # if the shipped default threshold changes.
        "option_overrides": {"sequential_threshold": 1024, "chunksize": 4096},
        "network": (
            f"dense ising n={n} (seed 1810, T=0.25), distinctions from "
            "mechanisms of size 1-2"
        ),
        "notes": "worker constructs a lazy Relation, as at the real site",
    }


def setup_mech_partitions():
    """One evaluate_partition call per mechanism partition (MIP-search shape)."""
    import pyphi
    from pyphi import System
    from pyphi.direction import Direction
    from pyphi.formalism.queries import evaluate_partition
    from pyphi.measures.distribution import resolve_mechanism_measure
    from pyphi.partition import mechanism_partitions

    n = 6
    system = System(rule_ring(154, n), (0,) * n)
    mechanism = purview = tuple(range(n))
    alphabet_sizes = system.substrate.factored_tpm.alphabet_sizes
    mechanism_measure = resolve_mechanism_measure(
        pyphi.config.formalism.iit.mechanism_phi_measure, alphabet_sizes
    )
    specification_measure = resolve_mechanism_measure(
        pyphi.config.formalism.iit.specification_measure
    )
    repertoire = system.repertoire(Direction.CAUSE, mechanism, purview)
    specified_state = system.intrinsic_information(
        Direction.CAUSE,
        mechanism,
        purview,
        specification_measure=specification_measure,
    ).ties[0]
    chunksize = pyphi.config.infrastructure.parallel_mechanism_partition_evaluation[
        "chunksize"
    ]
    pool_size = int(2.5 * chunksize * max(SIZE_FRACTIONS))
    pool = list(itertools.islice(mechanism_partitions(mechanism, purview), pool_size))

    def worker(partition):
        return evaluate_partition(
            system,
            Direction.CAUSE,
            mechanism,
            purview,
            partition,
            repertoire=repertoire,
            state=specified_state,
            mechanism_measure=mechanism_measure,
        )

    def run(items):
        from pyphi import conf
        from pyphi.parallel import map_reduce

        pkwargs = conf.parallel_kwargs(
            pyphi.config.infrastructure.parallel_mechanism_partition_evaluation
        )
        return map_reduce(worker, items, **pkwargs)

    return {
        "pool": pool,
        "run": run,
        "option": "parallel_mechanism_partition_evaluation",
        "option_overrides": {"sequential_threshold": 1024, "chunksize": 4096},
        "network": f"rule154 ring n={n}, full mechanism x full purview, CAUSE",
        "notes": (
            "short-circuit on zero-phi partitions disabled so workload size "
            "is controlled; the real site short-circuits"
        ),
    }


def setup_purviews():
    """One find_mip call per purview (MICE-search shape)."""
    import pyphi
    from pyphi import System
    from pyphi.direction import Direction
    from pyphi.substrate_generator import build_substrate

    n = 8
    rng = np.random.default_rng(1810)
    weights = rng.normal(1.0, 0.5, size=(n, n))
    substrate = build_substrate("ising", weights, temperature=0.25)
    state = (0,) * n
    mechanism = (0, 1)
    pool = list(System(substrate, state).potential_purviews(Direction.EFFECT, mechanism))

    def run(items):
        from pyphi import conf
        from pyphi.parallel import map_reduce

        # Fresh System per run so repertoire caches start cold in both
        # arms: per-item costs are first-computation costs, as in MICE
        # search over distinct purviews.
        system = System(substrate, state)

        def worker(purview):
            return system.find_mip(Direction.EFFECT, mechanism, purview)

        pkwargs = conf.parallel_kwargs(
            pyphi.config.infrastructure.parallel_purview_evaluation
        )
        return map_reduce(
            worker,
            items,
            total=len(items),
            size_func=lambda purview: 2 ** len(purview),
            **pkwargs,
        )

    return {
        "pool": pool,
        "run": run,
        "option": "parallel_purview_evaluation",
        "option_overrides": {"sequential_threshold": 64, "chunksize": 256},
        "network": f"dense ising n={n} (seed 1810, T=0.25), mechanism (0, 1), EFFECT",
        "notes": "fresh System per run keeps repertoire caches cold in both arms",
    }


def _sys_partition_workload(substrate_weights_seed=None, coupling_scale=1.0):
    """A system + the map_kwargs of the IIT 4.0 SIA partition map.

    With ``substrate_weights_seed`` set, a dense Ising network is used;
    ``coupling_scale`` scales the couplings between the two halves of the
    network, so a zero scale gives two decoupled blocks — a reducible
    system with zero-phi partitions. Otherwise the rule-154 ring is used.
    """
    import pyphi
    from pyphi import System
    from pyphi.direction import Direction
    from pyphi.formalism.iit4 import intrinsic_differentiation_value
    from pyphi.formalism.iit4 import system_intrinsic_information
    from pyphi.measures.distribution import resolve_mechanism_measure
    from pyphi.measures.distribution import resolve_system_measure
    from pyphi.partition import system_partitions
    from pyphi.substrate_generator import build_substrate

    n = 6
    if substrate_weights_seed is not None:
        rng = np.random.default_rng(substrate_weights_seed)
        weights = rng.normal(1.0, 0.5, size=(n, n))
        half = n // 2
        within_block = np.zeros((n, n), dtype=bool)
        within_block[:half, :half] = True
        within_block[half:, half:] = True
        weights[~within_block] *= coupling_scale
        substrate = build_substrate("ising", weights, temperature=0.25)
    else:
        substrate = rule_ring(154, n)
    system = System(substrate, (0,) * n)
    system_measure = resolve_system_measure(
        pyphi.config.formalism.iit.system_phi_measure
    )
    specification_measure = resolve_mechanism_measure(
        pyphi.config.formalism.iit.specification_measure
    )
    system_state = system_intrinsic_information(
        system, specification_measure=specification_measure
    )
    directions = tuple(Direction.both())
    map_kwargs = {
        "system": system,
        "system_state": system_state,
        "system_measure": system_measure,
        "directions": directions,
        "intrinsic_differentiation": {
            d: intrinsic_differentiation_value(d, system) for d in directions
        },
    }
    pool = list(
        itertools.islice(
            system_partitions(system.node_indices, node_labels=system.node_labels),
            4096,
        )
    )
    return pool, map_kwargs


def setup_sys_partitions():
    """One system-level evaluate_partition per system partition (SIA shape).

    The current sequential_threshold (1024) assumes cheap items; system
    partitions cost tens of ms each, so this level runs with the proposed
    threshold of 64 to measure whether parallelizing 64-2048 partitions
    pays. Short-circuiting is not passed (as for an irreducible system).
    """
    import pyphi
    from pyphi.formalism.iit4 import evaluate_partition

    pool, map_kwargs = _sys_partition_workload()

    def run(items):
        from pyphi import conf
        from pyphi.parallel import map_reduce

        pkwargs = conf.parallel_kwargs(
            pyphi.config.infrastructure.parallel_partition_evaluation
        )
        return map_reduce(evaluate_partition, items, map_kwargs=map_kwargs, **pkwargs)

    return {
        "pool": pool,
        "run": run,
        "option": "parallel_partition_evaluation",
        "option_overrides": {"sequential_threshold": 64, "chunksize": 4096},
        "size_fractions": (1 / 64, 1 / 8, 1 / 2),
        "cold": False,
        "network": "rule154 ring n=6, IIT 4.0 system partitions",
        "notes": (
            "sequential_threshold overridden to the proposed 64 in both "
            "arms; no short-circuit (irreducible-system shape)"
        ),
    }


def setup_sys_partitions_shortcircuit():
    """System partition map on a reducible system, with short-circuiting.

    The real SIA site short-circuits on the first zero-phi partition.
    On a reducible system the sequential arm exits almost immediately;
    a parallel dispatch pays pool round-trips and per-chunk work until
    each worker hits a falsy item. This bounds the flipped gate's
    regression on reducible systems.
    """
    import pyphi
    from pyphi import utils
    from pyphi.formalism.iit4 import evaluate_partition

    pool, map_kwargs = _sys_partition_workload(
        substrate_weights_seed=1810, coupling_scale=0.0
    )
    first_falsy = next(
        (i for i, p in enumerate(pool[:256]) if not evaluate_partition(p, **map_kwargs)),
        None,
    )
    if first_falsy is None:
        raise RuntimeError("system is not reducible; no zero-phi partition found")

    def run(items):
        from pyphi import conf
        from pyphi.parallel import map_reduce

        pkwargs = conf.parallel_kwargs(
            pyphi.config.infrastructure.parallel_partition_evaluation
        )
        return map_reduce(
            evaluate_partition,
            items,
            map_kwargs=map_kwargs,
            shortcircuit_func=utils.is_falsy,
            **pkwargs,
        )

    return {
        "pool": pool,
        "run": run,
        "option": "parallel_partition_evaluation",
        "option_overrides": {"sequential_threshold": 64, "chunksize": 4096},
        "size_fractions": (1 / 8, 1 / 2),
        "cold": False,
        "network": (
            "two decoupled dense ising blocks n=6 (seed 1810, cross "
            f"couplings zeroed), first zero-phi partition at index {first_falsy}"
        ),
        "notes": (
            "short-circuit on zero-phi enabled as at the real SIA site; "
            "items are randomly sampled so falsy density is realistic for "
            "this reducible system"
        ),
    }


def setup_complexes():
    """One SIA per candidate system (complexes-sweep shape)."""
    import pyphi
    from pyphi import examples
    from pyphi.substrate import _resolved_sia
    from pyphi.substrate import possible_complexes

    substrate = examples.rule154_substrate()
    state = (0,) * 5
    pool = list(possible_complexes(substrate, state))
    sia_fn, map_kwargs = _resolved_sia()
    map_kwargs.setdefault("progress", False)

    def run(items):
        from pyphi import conf
        from pyphi.parallel import map_reduce

        pkwargs = conf.parallel_kwargs(
            pyphi.config.infrastructure.parallel_complex_evaluation
        )
        return map_reduce(
            sia_fn, items, total=len(items), map_kwargs=map_kwargs, **pkwargs
        )

    return {
        "pool": pool,
        "run": run,
        "option": "parallel_complex_evaluation",
        "option_overrides": {"sequential_threshold": 16, "chunksize": 64},
        "size_fractions": (0.25, len(pool) / 64),
        "cold": False,
        "network": "rule154 example substrate (n=5), all candidate systems",
        "notes": "worker computes a full SIA per candidate, as in complexes()",
    }


SETUPS = {
    "relations": setup_relations,
    "mech_partitions": setup_mech_partitions,
    "purviews": setup_purviews,
    "sys_partitions": setup_sys_partitions,
    "sys_partitions_shortcircuit": setup_sys_partitions_shortcircuit,
    "complexes": setup_complexes,
}


def prewarm_pool():
    """Spawn the loky workers so timed trials see a warm pool."""
    from joblib.externals.loky import get_reusable_executor

    from pyphi.parallel import get_num_processes

    executor = get_reusable_executor(max_workers=get_num_processes())
    list(executor.map(abs, range(get_num_processes())))


def shutdown_pool():
    from joblib.externals.loky import get_reusable_executor

    from pyphi.parallel import get_num_processes

    get_reusable_executor(max_workers=get_num_processes()).shutdown(kill_workers=True)


def run_level(name: str, seed: int, trials: int) -> dict:
    import pyphi

    setup = SETUPS[name]()
    option = {
        **getattr(pyphi.config.infrastructure, setup["option"]),
        **setup.get("option_overrides", {}),
    }
    chunksize = option["chunksize"]
    threshold = option["sequential_threshold"]
    pool = setup["pool"]
    run = setup["run"]
    size_fractions = setup.get("size_fractions", SIZE_FRACTIONS)

    sizes = sorted(
        {
            max(1, round(frac * chunksize))
            for frac in size_fractions
            if round(frac * chunksize) <= len(pool)
        }
    )
    if len(pool) < 2 * chunksize:
        print(
            f"  [{name}] pool has {len(pool)} items (< 2x chunksize); "
            f"sizes capped at {max(sizes)}"
        )
    rng = random.Random(seed)

    cells = []
    cold_size = min(max(1, round(0.5 * chunksize)), len(pool))
    cold: list[float] = []
    with pyphi.config.override(**{setup["option"]: {**option, "parallel": True}}):
        prewarm_pool()
        # Untimed shakeout of both arms (JIT-ish warmup: imports, caches
        # in worker processes).
        with gate("current"):
            run(pool[: min(len(pool), chunksize + 1)])
        with gate("flipped"):
            run(pool[: min(len(pool), threshold)])

        for size in sizes:
            raw = {"current": [], "flipped": []}
            for trial in range(trials):
                indices = rng.sample(range(len(pool)), size)
                items = [pool[i] for i in indices]
                order = (
                    ("current", "flipped")
                    if trial % 2 == 0
                    else (
                        "flipped",
                        "current",
                    )
                )
                for arm in order:
                    with gate(arm):
                        start = time.perf_counter()
                        run(items)
                        raw[arm].append(time.perf_counter() - start)
            medians = {arm: statistics.median(ts) for arm, ts in raw.items()}
            cell = {
                "size": size,
                "fraction_of_chunksize": size / chunksize,
                "raw_seconds": raw,
                "median_seconds": medians,
                "speedup_current_over_flipped": (
                    medians["flipped"] / medians["current"]
                    if medians["current"]
                    else None
                ),
            }
            cells.append(cell)
            print(
                f"  [{name}] size={size} ({size / chunksize:.2f}x chunksize): "
                f"current={medians['current']:.4f}s flipped={medians['flipped']:.4f}s "
                f"-> flipped/current={medians['flipped'] / medians['current']:.2f}x"
            )

        # Cold-pool: flipped gate's first call after worker shutdown, at
        # the middle of the affected regime. The current gate runs these
        # sizes sequentially, so pool state is irrelevant to it.
        for _trial in range(COLD_TRIALS if setup.get("cold", True) else 0):
            indices = rng.sample(range(len(pool)), cold_size)
            items = [pool[i] for i in indices]
            shutdown_pool()
            with gate("flipped"):
                start = time.perf_counter()
                run(items)
                cold.append(time.perf_counter() - start)
        prewarm_pool()
        if cold:
            print(
                f"  [{name}] cold-pool flipped @ size={cold_size}: "
                f"median {statistics.median(cold):.4f}s"
            )

    return {
        "level": name,
        "option": setup["option"],
        "chunksize": chunksize,
        "sequential_threshold": threshold,
        "network": setup["network"],
        "notes": setup["notes"],
        "pool_size": len(pool),
        "cells": cells,
        "cold_pool_flipped": (
            {"size": cold_size, "raw_seconds": cold} if cold else None
        ),
    }


def unique_path(directory: Path, stem: str, suffix: str) -> Path:
    path = directory / f"{stem}{suffix}"
    version = 2
    while path.exists():
        path = directory / f"{stem}_v{version}{suffix}"
        version += 1
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1810)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument(
        "--levels", nargs="+", choices=sorted(SETUPS), default=sorted(SETUPS)
    )
    parser.add_argument("--run-label", default=None)
    args = parser.parse_args()

    import pyphi
    from pyphi.parallel import get_num_processes

    pyphi.config.progress_bars = False
    pyphi.config.parallel = True

    record = {
        "seed": args.seed,
        "trials": args.trials,
        "pool_state": "warm (pre-warmed loky executor); cold-pool section separate",
        "num_workers": get_num_processes(),
        "cpu_count": multiprocessing.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pyphi_version": getattr(pyphi, "__version__", None),
        "iit_version": pyphi.config.formalism.iit.version,
        "levels": [],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"b18_dispatch_gate_seed{args.seed}_trials{args.trials}"
    if args.run_label:
        stem += f"_{args.run_label}"
    path = unique_path(RESULTS_DIR, stem, ".json")

    # Save after every level so a crash in a later level cannot lose
    # earlier levels' raw data.
    for name in args.levels:
        print(f"level: {name}")
        record["levels"].append(run_level(name, args.seed, args.trials))
        path.write_text(json.dumps(record, indent=2))
    path.write_text(json.dumps(record, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
