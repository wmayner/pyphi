# Do campaign work units predict shard runtime?

`pyphi.cost.mechanism_workloads` charges each mechanism a number of work
units, and `plan_ces_shards` packs mechanisms into shards until each reaches
`units_per_job`. A production campaign reported shards of nearly equal units
differing several-fold in runtime by which rung of the planning ladder
produced them, which would make `units_per_job` useless as a runtime bound.
These scripts measure whether that holds.

## Scripts

| Script | What it measures |
|---|---|
| `measure_pair_cost.py` | CPU time of one (mechanism, direction, purview) MIP search against the units charged for it, plus the partition evaluations actually performed and the number of tied specified states |
| `measure_shard_cost.py` | CPU time, cache traffic, and resident memory of whole shards run through the campaign's own execution path. Arms: `plan`, `payload`, `multiplicity`, `size`, `cache` |
| `analyze.py` | Reads every result file and regresses observed cost on the charge |

Results land in `results/` keyed by their parameters, and are never
overwritten.

## Findings

Measured on a periodic Ising ring (radius 2, temperature ¼) under
`IIT_4_0_2026`, at 14, 16, and 21 units, purview order ≤ 3.

**The operation count is exact.** Across 46 (mechanism, purview) pairs
spanning mechanism orders 1–6, the number of `evaluate_partition` calls
equalled the charged partition count in every case — ratio 1.000, no
exceptions. Neither of the two mechanisms that could have broken this fired:
the zero-φ short-circuit never triggered (every sampled mechanism was
irreducible), and every pair had exactly one tied specified state, so the
per-state repetition of the partition sweep in `_find_mip_iit4` never
multiplied the work.

The larger mechanisms specifically — a suspected undercount, since
`JOINT_PARTITION_ALL` grows fast in mechanism order — are right on both
counts. Re-enumerating `mechanism_partitions` confirms the seeded counts for
(6, 3) = 18,306 and (7, 3) = 102,671, and a size-6 mechanism against an
order-3 purview costs 41.85 µs per unit, indistinguishable from size-5's
41.77 and size-4's 40.07. Neither the count nor the cost per partition rises
with mechanism order.

**Cost per unit is very nearly constant, and the model's one real error is
the relative weight of the two axes.** Regressing per-pair CPU time on
partition count gives 524 µs per purview evaluation and 41.8 µs per
partition: a purview evaluation costs about **12.5 partitions**, where the
model charged 1. That is the whole discrepancy the pair data shows, and it
matters only for scopes whose purviews are small enough that few partitions
amortize the fixed cost — cost per unit runs 105 µs at (|m|=1, |p|=1) against
a 41 µs asymptote by (|m|=5, |p|=3). `PURVIEW_EVALUATION_UNITS = 12` now
carries that weight.

**Neither shard form is dearer per unit.** At 21 units, matched on the unit
charge, one order-7 mechanism split across purviews (`purview_range`,
1,046,988 units) cost 44.8 µs/unit and 21 packed order-4 mechanisms
(`mechanisms`, 995,484 units) cost 42.1 µs/unit — 1.06× apart, the
`purview_range` form marginally the slower. The payload path is not the
variable.

**Packing many distinct mechanisms is not dearer per unit either.** At a
fixed unit total, shards carrying 66, 24, 6, and 4 distinct mechanisms cost
45.7, 41.8, 41.2, and 43.4 µs/unit. Cache miss rate does vary strongly with
packing — 9.6% for the 66-mechanism shard against 2.1% for the 4-mechanism
one — but the misses are too cheap relative to the partition sweep they feed
for that to show in the total. Units alone predict CPU time across these
shards to within 5.3%; adding a per-miss term does not improve the fit (its
coefficient comes out negative).

**Starving the cache costs about 20%, not several-fold.** Under a ceiling
below the process's own baseline resident memory — so nothing is admitted at
all — misses per unit rose 13× for the `mechanisms` form and 19× for
`purview_range`, and cost per unit rose only 1.17× and 1.09×. This is the
paired test of the locality hypothesis, and it refutes it: recomputing a
repertoire is cheap beside the sweep it serves.

**Shard size drifts upward mildly.** At 21 units, the same shard composition
at 2.0 M, 5.5 M, and 20.1 M units cost 44.00, 45.02, and 50.65 µs/unit — about
1.15× per decade of shard size, with resident memory growing 202 → 413 MiB and
no evictions. (The largest run shared the machine with a test suite, so part of
that 1.15× may be memory-bandwidth contention rather than scale.) Extrapolated
to production's ~100 M units this is worth perhaps 1.25×, not 4×.

**So the production report is not reproduced by anything in the model.** Its
`purview_range` shard (70.4 M units in 63 min, 53.7 µs/unit) agrees with
these measurements to 1.2×. Its `mechanisms` shards (99.8 M units, ≥4 h 55 m
CPU, ≥177 µs/unit) do not, and no controlled variation here — shard form,
mechanism multiplicity, substrate size, shard size, cache starvation — moves
cost per unit by more than about 1.2× each, or 1.35× compounded. What remains
untested is outside the cost model: the execute nodes the two forms landed on.
`CampaignTaskOutput.metrics` now records per-shard CPU time and cache traffic,
so the next campaign answers this from its own outputs.

## Reproducing

```bash
uv run python experiments/units_runtime_model/measure_pair_cost.py --units 14 --seed 1
uv run python experiments/units_runtime_model/measure_shard_cost.py --units 16 --arm multiplicity --seed 1
uv run python experiments/units_runtime_model/measure_shard_cost.py --units 21 --arm cache --ceiling-mib 400 --seed 1
uv run python experiments/units_runtime_model/analyze.py
```

The 21-unit arms spend several minutes building the substrate before any
measurement starts.
