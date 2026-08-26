# What's new in PyPhi 2.0

PyPhi 2.0 is a comprehensive rework of the library around IIT 4.0: new core
value types; first-class formalisms covering IIT 3.0, both IIT 4.0
formulations, and actual causation; multi-valued units; closed-form
relations; analysis across spatiotemporal scales; and rebuilt configuration,
serialization, display, and parallel execution. This page tours the
highlights; the complete list of changes is in the
[changelog](https://github.com/wmayner/pyphi/blob/main/CHANGELOG.md).

## Upgrading from 1.x

Three changes affect existing code and results:

- **The core types follow the IIT 4.0 paper's vocabulary.** `Network` is now
  `Substrate`, `Subsystem` is now `System`, `Concept` is now `Distinction`,
  and the partition and cut names follow the paper throughout.
- **The default formalism is IIT 4.0 (2026)**, which adds
  [the intrinsic-information requirement](theory/intrinsic-information.md)
  to system integrated information; under it, **deterministic systems
  compute φ_s = 0**. The 2023 formulation and IIT 3.0 remain fully supported
  and reproduce published values (see the "Published results reproduce"
  section below for the two documented exceptions).
- **Other breaking changes:** Python 3.13+ is required, configuration is
  restructured into layered namespaces, and logging is off by default —
  PyPhi no longer writes a `pyphi.log` file into the working directory; opt
  in with `pyphi.enable_logging()`.

The [migration guide](migration/migration-2.0.md) covers all of this with
before-and-after examples for code written against 1.x or the
`feature/iit-4.0` branch.

## Intrinsic units: analysis across scales

What is the right spatiotemporal grain at which to analyze the causal powers of
a system? IIT provides a principled answer to this question. PyPhi 2.0
implements the macro-unit framework of Marshall, Findlay, Albantakis & Tononi
(2024) in `pyphi.macro`: macro units are defined by coarse-graining or
blackboxing constituents at a finer grain — over space, over update steps, or
both — and the four-step macro TPM construction turns a set of macro units into
a `MacroSystem` that behaves exactly like a micro `System` in the standard IIT
4.0 pipeline. Both of the paper's worked examples are reproduced at the
published precision.

The framework comes with its judgments attached. The intrinsic unit
criteria (`pyphi.macro.criteria`) decide whether a candidate unit has the
cause-effect power to count as a unit at all, returning verdicts with
witnesses. The bounded grain search (`pyphi.macro.search`) enumerates
candidate mappings within explicit `SearchBounds` and answers the top-level
question — which systems, at which grains, are complexes? — with
`pyphi.macro.complexes()`, returning winners, ties, and the full evaluation
record. The search is also reachable from the main entry point:
`pyphi.analyze(substrate, state, grains=True)`. Under the default
formalism, the search skips partition sweeps whose outcome is already
certified by
[the intrinsic-information requirement](theory/intrinsic-information.md),
with identical results.

See the theory page [Macro units](theory/macro-units.md), the how-to
[Search across grains](howto/grain-search.md), and the
[macro tutorial](tutorials/macro.md).

## Multi-valued units

Units are no longer required to be binary. The whole pipeline —
repertoires, partitions, the system irreducibility analysis, cause-effect
structures, relations, display, and serialization — supports units with any
number of states, including substrates that mix alphabets. Pass
`state_space=` (a uniform alphabet size or per-node state labels) or
`alphabet=k` when constructing a `Substrate`, and states can be given as
labels rather than integers. The TPM is stored in per-node-factored form
(`FactoredTPM`), which is what makes heterogeneous alphabets natural to
represent; the joint conditional is available on demand as a read-only view.

Multi-valued support extends to the measures and to actual causation: the
EMD repertoire measure counts differing states over the actual state space,
so it remains usable as the IIT 3.0 mechanism measure on non-binary
substrates, and the actual-causation pipeline inherits k-ary support through
the shared `System` machinery. Multi-valued golden fixtures pin end-to-end
correctness, including the three-candidate voting example of Albantakis et
al. (2019).

## Query the relational structure without enumerating it

Relations dominate a cause-effect structure at any interesting scale. The
IIT 4.0 paper's specialized-lattice example (Fig. 6D) has 27 distinctions
and 1,537,080 relations; enumerating them takes about a minute and more
than a gigabyte of memory, and past roughly 35 distinctions enumeration
stops fitting in memory at all. But the relation set is fully determined
by the distinctions, so most questions about it can be answered exactly
without ever building it.

PyPhi 2.0 turns that fact into an interface — and makes it the default.
Computed structures now carry analytical relations: a closed-form summary
that answers structural questions in milliseconds, at any scale:

```python
relations = ces.relations

relations.sum_phi()           # exact Σφ_r
relations.num_relations()     # exact count
relations.degree_spectrum()   # exact count and Σφ_r at every degree
relations.phi_histogram()     # exact histogram of φ_r values
relations.phi_mean_std()      # exact mean and std of φ_r
relations.max_phi()           # the strongest relation's φ_r
relations.binding_matrix()    # how strongly each pair of unit-states is
                              # bound by relations (as a DataFrame)
relations.maximal_relations() # the inclusion-maximal relations — the
                              # facets of the relation complex
relations.maximal_faces()     # the inclusion-maximal relation faces
```

Three more tools cover what closed forms can't:

- `relations.strongest(k)` yields relations one at a time in exact
  descending φ_r order, so the top ten of a million-relation structure
  cost about a millisecond — useful for plots and reports that only ever
  show the strongest few.
- `relations.sample(n, seed=...)` draws an unbiased sample of relations
  and returns estimates with standard errors for any per-relation
  quantity you define — including questions with no closed form.
- `relations.materialize(max_degree=..., min_phi=...)` builds concrete
  `Relation` objects again when you need them, with bounds so you
  can't accidentally ask for all 2^27 of them.

The same queries work on Φ-folds (restricted to the relations touching a
set of distinctions), and
`ces.distinction_importance()` ranks each distinction by its additive
contribution to Φ — the contributions sum to Φ exactly, so the ranking is
a true decomposition of the structure's integrated information. Fold
contributions are share-weighted, so the same decomposition generalizes:
the Φ-folds of *any* partition of the distinctions — not just singletons —
tile Φ exactly.

The structures compose as objects in their own right:
`ces.distinctions.filter(...)` selects distinctions by predicate,
`ces.induce(...)` and `ces.meet(other)` produce relation-closed
substructures that the same queries and plots accept, `ces.relabel(...)`
carries a structure through a node relabeling, and
`pyphi.automorphism.are_structures_isomorphic()` decides whether two
structures are identical up to a relabeling of the units.

Enumerated relation sets remain available: set
`pyphi.config.relation_computation = "CONCRETE"` to compute them for every
structure, or call `materialize()` on one. See the how-to guide
[Query relational structure](howto/query-relations.md).

## Finding complexes

Complex-finding follows the exclusion postulate exactly. `Substrate.complexes()`
returns `Complex` objects — the non-overlapping local maxima of integrated
information — resolving overlaps by the recursive exclusion cascade
(Marshall et al. 2023, Algorithm A1), including the chain topologies and
tied cliques where a greedy pass selects the wrong complexes. Each
`Complex` records the
overlapping candidates excluded in its favor and its `exclusion_margin`,
the φ_s gap to the best rival it beat. The
[recursive exclusion tutorial](tutorials/recursive-exclusion.md) walks
through how complexes carve up a substrate.

## Intrinsic meaning and matching

`pyphi.matching` implements the perception and matching framework of
Mayner, Juel & Tononi (2024,
[arXiv:2412.21111](https://arxiv.org/abs/2412.21111)). A
`PerceptualSystem` embeds a complex in an environment through a sensory
interface; each stimulus triggers a state of the complex, and the portion
of the Φ-structure that stimulus evokes is the *perception* — its intrinsic
meaning for the system. `Perception` decomposes perceptual richness by
distinction, relation, and Φ-fold; `Differentiation` measures how much the
triggered structures differ across a set of stimuli (with a closed-form
variant that needs no relation enumeration); and
`MatchingAnalysis.matching()` estimates *matching* — how much more
perceptual differentiation the environment evokes than random noise — with
seeded paired sampling and the per-trial raw values kept on the result.
Composable environment generators (`segment`, `point`, `noise`,
`superpose`, `mixture`) build world distributions over the sensory
interface, and the paper's mechanism library is ported into
`pyphi.substrate_generator`, so perceptual substrates can be built natively.

## IIT 4.0 (2026) is the default formalism

PyPhi 2.0 computes the 2026 refinement of IIT 4.0 by default: the system's
intrinsic information enters the minimum that defines system integrated
information (Mayner, Marshall, Tononi 2026), so a system must both furnish
itself a repertoire of alternatives and specify one of them. One consequence
to know before comparing against published numbers: **deterministic systems
compute φ_s = 0** under the default. The 2023 formulation and IIT 3.0 remain
fully supported — `pyphi.analyze(..., formalism="IIT_4_0_2023")` or the
presets in `pyphi.conf.presets` reproduce published values exactly. See the
theory page
[The intrinsic-information requirement](theory/intrinsic-information.md).

## Every formalism, restored and validated

The formalisms are first-class objects, selected by name: `"IIT_3_0"`,
`"IIT_4_0_2023"`, `"IIT_4_0_2026"`, and `"AC_2019"` for actual causation.
Each formalism owns its algorithms, partition schemes, and compatible
measures, and configurations that mix formalisms incoherently — a
distribution distance with an IIT 4.0 version, say — are rejected when you
set them rather than computing a quantity the papers never defined.
Configuration is layered to match — options that affect results
(`formalism`), options that only affect execution (`infrastructure`), and
numerical settings (`numerics`) — and presets switch formalism wholesale in
one call: `with pyphi.config.override(**pyphi.iit3): ...`.

IIT 3.0 is restored paper-faithfully, with tie resolution matching the 2014
paper, the PyPhi 1.x background convention on subset systems (so published
1.x results reproduce), and canonical reference values pinned in the test
suite. Actual causation is restored per Albantakis et al. (2019), with its
own configuration namespace, paper-faithful defaults, tie cascades, and
enforcement of the realization principle: transitions that cannot occur
under the substrate dynamics are rejected up front. See
[Formalism versions](theory/formalism-versions.md),
[IIT 3.0](theory/iit-3.0.md), and the
[actual-causation tutorial](tutorials/actual-causation.md).

## One call from substrate to answer

The common workflows are one call each:

```python
analysis = pyphi.analyze(substrate, state)    # SIA, CES, and Φ in one bundle
analysis = pyphi.analyze(substrate, state, formalism="IIT_3_0")

result = pyphi.sweep(                         # batch across axes
    {"grid": grid, "ring": ring},
    states="all",
    formalisms=["IIT_4_0_2023", "IIT_4_0_2026"],
    compute="sia",
)
result.df                                     # tidy long-format DataFrame
```

`sweep()` enumerates the cartesian product of substrates, states, candidate
subsystems, and formalisms, runs cells in parallel, records
dynamically-unreachable states instead of aborting, and returns both a tidy
DataFrame and the aligned raw result objects. See
[Sweep states and subsystems](howto/sweep.md).

## Ties are resolved by the postulates

Whenever candidates tie — specified states, mechanism partitions, purviews,
system partitions, or overlapping candidate systems — PyPhi 2.0 resolves
the tie the way the theory says to: by escalating through the postulates
(the IIT 4.0 S1 tie supplement), at every selection point, under every
formalism. A tie that survives the cascade is reported as a tie: the tied
set is carried on the result (`sia.ties`, partition and state ties on
repertoire analyses, purview ties on MICE), survives serialization, and a
tie the postulates cannot adjudicate at all yields a null result with an
explicit reason. Outcomes are deterministic — across runs, across parallel
backends, and regardless of worker scheduling. See
[Break ties deliberately](howto/tie-breaking.md).

## Results explain themselves

Every result can account for itself:

- `result.explain()` reports why a Φ/φ/α value came out as it did: which
  short-circuit conditions applied, the winning and runner-up partitions
  and the φ gap between them, and the binding direction.
- `result.diff(other)` compares two results: Δφ, whether a MIP change is
  real or a reshuffle among tied partitions, gained and lost distinctions
  and relations, and which configuration differences could account for the
  change.
- Selection margins quantify how decisively each choice was made — the gap
  to the runner-up partition, purview, and specified state — at the system,
  mechanism, and complex levels, with an `effectively_tied` flag when a
  margin is within numerical precision of zero.
- Every top-level result carries the exact configuration that produced it
  (`result.config`) and a provenance record (pyphi version, git revision,
  timestamp, dependency versions), so a saved result is self-describing and
  `pyphi.config.override(**result.config.as_overrides())` reruns it
  exactly.

## Every result displays itself

Every result type renders as a structured card — grouped sections, readable
numbers, collections as tables — in the terminal, and as styled HTML in
notebooks, with node labels used throughout and multi-valued states written
as subscripts (`A₂`). Verbosity levels control how much is computed and
shown, large tables truncate rather than flood the terminal, and
`to_pandas()` exports labeled, analysis-friendly data from every
displayable type. The visualization module gains five views of a
cause-effect structure on one projection layer — the 3-D hypergraph, an
inclusion lattice, a composition scatter, a shared-φ matrix, and a
per-degree spectrum — all of which render analytically-computed structures
via `max_relations=`.

The value types also bridge to the graph ecosystem: `Substrate.to_networkx()`
and `from_networkx()` convert to and from labeled directed graphs,
`to_graphml()` and `to_adjacency()` export for graph tools, and
`Substrate.to_dbn()` unrolls the substrate into a two-timeslice dynamic
Bayesian network — all built on the TPM-inferred causal connectivity by
default, so the exported graph reflects the actual dynamics rather than the
declared connectivity matrix.

See [Visualize results](howto/visualize.md) and
[Export results](howto/export.md).

## Save anything, load it anywhere

`pyphi.save(obj, path)` and `pyphi.load(path)` (and `.save()`/`.load()` on
result objects) serialize every result type through typed schemas, as JSON
or compact binary msgpack, with transparent gzip for `.gz` paths. Files are
an order of magnitude smaller — a phi-structure that was 1.3 MB is now
56 KB — numpy arrays are stored exactly, and every computed field
round-trips, including tie sets, margins, the config snapshot, and
provenance. An opt-in disk cache (`disk_cache_results`) makes repeated
top-level computations load instead of recompute, keyed on the system's
mathematical content, the result-affecting configuration, and the pyphi
version. See [Save and load results](howto/save-load.md) and
[Caching](howto/cache.md).

## Estimate substrates from data

When the TPM is measured rather than known, `pyphi.estimate` keeps the
uncertainty attached. `estimate_substrate(data, regime=...)` builds a
`SubstratePosterior` from perturbational transition pairs or an
observational trajectory, with a `CoverageReport` recording which states
the data actually constrained. Any existing computation applies to
posterior samples unchanged, and `phi_posterior()` propagates the posterior
through the analysis by Monte Carlo, reporting the full mixture — the
probability the system is integrated at all, conditional and unconditional
quantiles, raw Φ samples, and which unit set is maximal per sample.
Coercing the result to a bare float raises, pointing at the summaries that
respect the uncertainty.

## Know the bounds, and the cost, before you run

`pyphi.formalism.iit4.bounds` implements the certified upper bounds of
Zaeemzadeh & Tononi (2024) on distinction, relation, system, and structure
quantities, each returned with its certificate and assumptions; measured
bounds evaluated on a distinction set's per-atom profile are typically
orders of magnitude tighter and still require no relation enumeration. A
debug check (`validate_phi_bounds`) compares every in-domain result against
the theorem-certified ceilings — since the bounds are proven, a violation
demonstrates a bug rather than noise.

Costs are countable before anything runs: `pyphi.estimate_analysis()`
counts the workload of a single-system analysis — system partitions,
mechanisms, purview evaluations, mechanism-partition sweeps — without
computing any φ, and `SearchBounds.estimate()` does the same for the grain
search. These pre-flights power admission checks in the MCP server and the
cluster-campaign planner, so an intractable run is refused with numbers
instead of discovered by timeout.

## Explore the substrate landscape

IIT quantities are functions of the substrate's parameters, and PyPhi 2.0
treats them that way. `pyphi.landscape` evaluates the analysis along
continuous parameter axes: `landscape_section()` tracks φ, the identity of
every discrete selection (MIP, specified states), the selection margins,
and the boundaries where a selection switches; `perturb()` estimates local
derivatives and the parameter distance to the nearest selection switch.
`pyphi.optimize()` searches over connection weights for maximizers of
signed normalized φ_s (or any other objective), seeded and with the full
evaluation trajectory saved. See
[Explore substrate parameter landscapes](howto/landscape.md).

## Faster across the board

Several changes compound into orders-of-magnitude speedups:

- Every configuration change used to serialize the entire config to disk —
  including the scoped overrides the compute pipeline makes internally.
  Removing that overhead made hot paths ~60–300× faster; the golden test
  suite dropped from about 13 minutes to about 13 seconds.
- The repertoire computations were rewritten as a stateless kernel;
  evaluating a system partition is roughly 18–20× faster than in the
  pre-2.0 implementation.
- The cause-side Bayesian inversion evaluates as a sum-product contraction
  over the factored TPM's dependence structure instead of materializing the
  joint likelihood over all substrate units, so a small system embedded in
  a large, sparsely connected substrate is now tractable on the cause side.
- Caches are keyed on the mathematics rather than object identity:
  reconstructed systems, relabelings, and same-topology parameter sweeps
  reuse each other's results.
- The specified-state computation no longer materializes the full state
  space — memory drops from 2ⁿ repertoires to one — and parallel dispatch
  decisions were retuned against measured per-item costs.

These are per-partition and per-structure gains; end-to-end wall time also
depends on how many partitions the configured scheme sweeps, and the
default system partition scheme changed in 2.0 to the paper-faithful
`DIRECTED_SET_PARTITION`, which evaluates a larger partition family than
the 1.x default. So a run under default settings is not directly comparable
to a 1.x timing, and can even take longer despite the faster kernel.

The [computational complexity](theory/computational-complexity.md) page
derives where the time goes for every formalism and measures which
configuration choices extend the tractable system size.

## Parallelism, overhauled

Parallel execution runs on a single scheduler abstraction with process,
thread, and Dask backends; on free-threaded Python builds the thread
backend is selected automatically, and a free-threaded CI lane runs the
full test suite with the GIL disabled. Work is packed into cost-balanced
chunks using cheap per-item cost estimates, dispatch thresholds are tuned
to measured per-item costs, and workers install the caller's exact
configuration — so `config.override(...)` scopes apply on every backend,
and parallel results are identical to sequential ones, including tie
resolution. See [Parallelize computations](howto/parallel.md).

## From laptop to cluster

`pyphi.campaign` turns a computation into a directory of self-contained
batch jobs for an HTCondor pool: `prepare()` packs the work into
cost-balanced tasks and writes a submit file, `status()` and `collect()`
work purely from output files, and resubmitting failures is just running
`condor_submit` again. For a single system too large to analyze whole,
`prepare_ces()` distributes the cause-effect structure computation itself:
a `CESScope` declares which mechanisms and purviews are feasible, shards
are planned to a per-job budget, and `collect()` reassembles the exact
structure — tie sets preserved — with a certified scope report bounding
what the scope excluded. Within the scope every value is exact: a scope
narrows the computation, it never approximates it. A Dask backend covers
the interactive path: connect a `distributed.Client`, set
`config.parallel_backend = "dask"`, and PyPhi's parallel levels spread over
the cluster. See [Run campaigns](howto/campaigns.md) and
[PyPhi on CHTC](howto/chtc.md).

## An interface for AI assistants

PyPhi ships a Model Context Protocol server
(`pip install pyphi[mcp]`, then `pyphi-mcp`) that lets AI assistants drive
the library: tools to build substrates, run and inspect analyses, render
the built-in visualizations, estimate costs, and prepare and collect
cluster campaigns, plus a bundled, citation-checked IIT reference and
guided prompts — explaining a result in plain language, porting pre-2.0
code, turning a natural-language description into a valid substrate, and
planning a cluster campaign step by step. See
[Use the MCP server](howto/mcp-server.md).

## Published results reproduce

Every published worked example is reproduced at its published precision and
pinned in a permanent acceptance suite: the IIT 4.0 paper's Figs. 1, 2, and
4, all five Fig. 6 architectures, and the three Fig. 7 panels (Albantakis
et al. 2023) — with the authors' exact weight matrices, previously
available only as figure graphics, now shipping in `pyphi.examples`; both
worked examples of the macro-unit paper (Marshall et al. 2024); the Fig. 12
constellation and Φ of the IIT 3.0 paper (Oizumi et al. 2014); the
canonical OR-AND account and the three-candidate voting example of the
actual-causation paper (Albantakis et al. 2019); the multi-valued p53-Mdm2
network (Gómez et al. 2020); and the matching manuscript's environment
distributions (Mayner et al. 2024). Where a paper publishes a number, PyPhi
2.0 computes it — with two documented exceptions. The structure integrated
information of the IIT 4.0 paper's Fig. 6D, and the relation count and Φ of
its Fig. 7B, embed an ordering-dependent resolution of tied distinction
states; resolving those ties by the rule of the paper's own S1 supplement
(the state that maximizes Φ) yields strictly larger structures, and PyPhi
2.0 computes those values instead (Φ = 12395 for Fig. 6D; 13498 relations
and Φ = 19.32 for Fig. 7B). φ_s and the distinction counts match the
figures exactly in both cases.

## Correctness and development

Beyond the features, 2.0 closes a long list of correctness gaps; the
[changelog](https://github.com/wmayner/pyphi/blob/main/CHANGELOG.md)
has the full accounting. The themes: formalism equations validated against
the papers (including two fixes to the Eq. 23 intrinsic-information
requirement found by cross-formalism property tests); parallel results made
deterministic and identical to sequential ones; serialization made
complete; construction-time validation added where malformed input
previously produced wrong numbers without an error; and multi-valued
repertoires verified to machine precision against an independent reference.

The release is backed by new infrastructure for keeping it correct: a
golden regression harness pinning raw numerical outputs across more than
two dozen fixtures spanning all three formalisms, Hypothesis property tests
for invariants from the IIT 4.0 paper, the paper-reproduction suite run in
CI, deterministic call-count gates that block performance regressions, and
a nightly ASV benchmark suite. Packaging is modernized: a single
`pyproject.toml`, `uv`-based development, wheels built from a clean
`hatchling` backend, and Python 3.13+.
