<!--
Staged release notes for 2.0.0. This curated section was moved out of
CHANGELOG.md so `towncrier build` can insert the fragment-generated section
there at release time (towncrier refuses to run when the target version is
already present in the changelog). Newer changes live as fragments in
changelog.d/ — render them with `just changelog-draft`. At release: run
`towncrier build --version 2.0.0`, then merge this narrative and the
generated entries into one section.
-->

2.0.0
-----
_(unreleased)_

PyPhi 2.0 is a comprehensive rework of the library around IIT 4.0 (Albantakis et
al. 2023; Mayner et al. 2026): new core value types, first-class formalism
objects covering IIT 3.0, both IIT 4.0 variants, and actual causation,
multi-valued units, closed-form relations, distributed computation, and
rebuilt configuration, serialization, and display. Changes are described
relative to PyPhi 1.2.0 and the `feature/iit-4.0` development branch.

For a narrative tour see [What's new in 2.0](docs/whats-new-in-2.0.md); for
porting existing code, see the migration guide (the `migration` reference topic
and `migrate_code` prompt on the MCP server).

### Highlights

- **The intrinsic-units macro framework** of Marshall, Findlay, Albantakis &
  Tononi (2024) is implemented in `pyphi.macro`: macro units defined by
  coarse-graining or blackboxing micro or meso constituents, the four-step
  macro TPM construction, the intrinsic-unit criteria, and the bounded
  cross-grain search — and a `MacroSystem` goes through the standard IIT
  4.0 pipeline exactly like a micro `System`. Both paper examples reproduce
  at the published precision.
- **Multi-valued (k-ary) units** are supported throughout the SIA/CES
  pipeline, with per-node alphabets and a per-node-factored TPM
  (`FactoredTPM`) as the canonical representation.
- **Relations without enumeration.** Analytical relations are the default,
  and many new properties of the relation set are computed in closed form,
  with no enumeration of the exponentially large set: Σφ_r, relation and
  face counts, the full degree spectrum, φ_r moments and the exact φ_r
  histogram, the maximum φ_r, the atom-pair binding matrix, maximal
  relations and faces (the facets of the relation complex), per-fold and
  per-distinction Φ contributions, and each distinction's importance. The
  strongest k relations enumerate lazily in exact descending-φ_r order, and
  seeded sampling gives unbiased coverage-weighted estimates with standard
  errors.
- **Intrinsic meaning and matching.** `pyphi.matching` implements the
  perception and matching framework of Mayner, Juel & Tononi (2024,
  [arXiv:2412.21111](https://arxiv.org/abs/2412.21111)): perceptual systems
  embedded in an environment through a sensory interface, triggered
  Φ-structures and per-stimulus perception, differentiation across stimuli,
  and matching — how much more perceptual differentiation the environment
  evokes than random noise — with composable environment generators and
  seeded paired sampling.
- **The default formalism is IIT 4.0 (2026)**, which applies the
  intrinsic-information requirement φₛ = min{φ_c, φ_e, ii(s)} (Mayner et al.
  2026, Eq. 23). Select `"IIT_4_0_2023"` for system φ without the
  requirement, or `"IIT_3_0"` for IIT 3.0. Under the default, deterministic
  systems compute φₛ = 0.
- **Large performance gains**, including removal of a configuration-write
  overhead that slowed hot paths by ~60–300×, a reduced-dimension cause-side
  inversion that makes small systems inside large sparse substrates
  tractable, and caches keyed on mathematical content so equivalent systems
  share work.
- **New research tools**: substrate estimation from data with uncertainty
  propagation (`pyphi.estimate`), continuous parameter landscapes and
  optimization (`pyphi.landscape`, `pyphi.optimize`), and certified upper
  bounds on IIT quantities (`pyphi.formalism.iit4.bounds`).
- **Formalisms are first-class objects.** The active formalism is selected by
  name — `"IIT_3_0"`, `"IIT_4_0_2023"`, `"IIT_4_0_2026"` — and owns its
  algorithms, partition schemes, and compatible measures; incompatible
  combinations are rejected at configuration time. Presets (`pyphi.iit3`,
  `pyphi.iit4_2023`, `pyphi.iit4_2026`) switch formalism wholesale in one
  call.
- **Ties are resolved by postulate cascades everywhere.** Tie resolution
  follows the IIT 4.0 S1 tie supplement at every selection point, tied sets
  are carried on results, and outcomes are deterministic across runs,
  backends, and worker scheduling.
- **Actual causation is restored** per Albantakis et al. (2019), with its own
  registered formalism, config namespace, tie cascades, and enforcement of
  the realization principle.
- **IIT 3.0 is restored paper-faithfully**, with canonical reference values,
  its own complete preset, and tie resolution matching the 2014 paper.
- **Paper-aligned naming.** `Network` is now `Substrate`, `Subsystem` is now
  `System`, `Concept` is now `Distinction`, and the partition/cut vocabulary
  follows the IIT 4.0 paper throughout (see the rename tables under API
  changes).
- **New entry points.** `pyphi.analyze()` runs a single analysis and returns
  a uniform bundle; `pyphi.sweep()` runs batches across substrates, states,
  subsets, and formalisms; `pyphi.estimate_analysis()` prices a computation
  before you run it. Results carry `explain()`, `diff()`, selection margins,
  a config snapshot, and provenance.
- **Rich display.** Every result type renders as a structured card in the
  terminal and as styled HTML in notebooks, and exports labeled data via
  `to_pandas()`.
- **Serialization is rebuilt** on typed schemas — JSON and binary msgpack,
  transparent gzip, `pyphi.save`/`pyphi.load` — with files an order of
  magnitude smaller and every computed field round-tripping.
- **Distributed computation.** A scheduler abstraction with process, thread,
  and Dask backends; free-threaded Python support; and `pyphi.campaign` for
  HTCondor batch campaigns, including scoped cause-effect-structure sharding
  with exact tie-preserving reassembly and certified scope reports.
- **An MCP server** (`pyphi-mcp`) exposes PyPhi to AI assistants: building
  substrates, running and inspecting analyses, plotting, cost estimation,
  campaign preparation, and a citation-checked IIT reference.

### API additions

**Analysis entry points**

- `pyphi.analyze(substrate, state, *, subset=…, formalism=…, compute=…)`:
  analyze one candidate system and get an `Analysis` bundle exposing `.sia`,
  `.ces`, and `.phi` uniformly across formalisms. `formalism=` switches
  formalism per call; `compute="sia"`/`"ces"` returns the raw result;
  `grains=True` runs the bounded intrinsic-unit grain search instead.
- `pyphi.sweep(substrates, states=…, subsets=…, formalisms=…, compute=…)`:
  run the same computation over the cartesian product of axes and get a
  `SweepResult` — a tidy long-format DataFrame plus the aligned raw results.
  Accepts a single substrate, a sequence, or a `{label: substrate}` mapping;
  `"all"` enumerates states or subsets, recording dynamically-unreachable
  states in `.skipped`; SIA rows carry the selection-margin columns; cells
  run in parallel when parallelism is enabled.
- `pyphi.estimate_analysis()`: an analytic pre-flight that counts the
  workload of a single-system analysis — system partitions, candidate
  mechanisms, purview evaluations, mechanism-partition sweeps — without
  computing any φ, optionally restricted to a scope.
  `SearchBounds.estimate()` is the same pre-flight for the macro grain
  search.
- A `Complex` result type: `Substrate.complexes()` and `maximal_complex()`
  return `Complex` objects exposing `is_maximal`, the excluded overlapping
  candidates, and `exclusion_margin` (the φₛ gap to the best overlapping
  rival).

**Intrinsic units and macro analysis**

- `pyphi.macro`: the intrinsic-units macro framework of Marshall, Findlay,
  Albantakis & Tononi (2024). `MacroUnit`, `coarse_grain()`, and
  `blackbox()` define macro units over micro or meso constituents;
  `macro_tpms()` implements the four-step macro TPM construction (Eqs.
  26–40); and a `MacroSystem` goes through the standard IIT 4.0 pipeline
  exactly like a micro `System`. Both paper examples are reproduced at the
  published precision.
- The intrinsic-unit criteria and bounded grain search (Marshall et al.
  2024, Sec. 2.2.2): `pyphi.macro.criteria` (Eqs. 15–16 verdicts with
  witnesses) and `pyphi.macro.search` (`SearchBounds`, the intrinsic-unit
  recursion, the valid-system set, and the one-call Eq. 19 `complexes()`
  driver returning winners, ties, and the full evaluation record). Search
  drivers can parallelize their independent φₛ evaluations, and under the
  default formalism the search skips partition sweeps whose outcome is
  certified by the intrinsic-information requirement (`prune=`), with
  identical results.

**Multi-valued units and TPMs**

- Multi-valued (k > 2) units are supported throughout the SIA/CES pipeline,
  including heterogeneous per-node alphabets, with golden fixtures verifying
  end-to-end correctness. Substrates are constructed via
  `Substrate(state_space=…)` (a uniform alphabet size or per-node state
  labels) or `alphabet=k`; states may be given as labels. The EMD repertoire
  measure generalizes to k-ary state spaces, so it remains usable as the IIT
  3.0 mechanism measure on non-binary substrates. Actual causation inherits
  k-ary support through the shared `System` machinery.
- `FactoredTPM` is the canonical TPM representation: per-node conditional
  factors, constructed directly via `marginals=` or
  `Substrate.from_factored()`, with joint-array input auto-converted.
  Convenience methods include `is_deterministic()`, `permute_nodes()`,
  `subtpm()`, `infer_cm()`/`infer_edge()` (TPM-implied connectivity), and
  `to_xarray()`. `JointTPM` is a read-only view of the joint conditional
  (`Substrate.joint_tpm()`), uniform for binary and k-ary substrates.

**Relations and structure queries**

- Relation sets answer structural queries without enumeration: φ_r moments,
  per-degree counts and sums, the exact φ_r histogram, the atom-pair binding
  matrix, maximal relations and faces (the facets of the relation complex),
  `strongest(k)` (lazy, in exact descending-φ_r order), and seeded unbiased
  `sample(n)` with standard errors. Every query is closed-form on
  `AnalyticalRelations` and answered by iteration on `ConcreteRelations`;
  `materialize()` explicitly enumerates a bounded concrete set when one is
  needed.
- `PhiFold`: the slice of a cause-effect structure induced by one or more
  distinctions with their incident relations. `big_phi_contribution` gives a
  fold's additive share of Φ, and fold contributions tile: summing over any
  partition of the distinctions recovers `big_phi` exactly.
  `CauseEffectStructure.distinction_importance()` ranks distinctions by that
  contribution.
- Structure algebra: `CauseEffectStructure.induce(distinctions)` (an induced
  substructure view), `.meet(other)` (the induced substructure on the shared
  distinctions), and `.relabel(mapping)` (rewrite a structure through a
  node-index bijection, preserving φ exactly).
  `pyphi.automorphism.structure_signature` and `are_structures_isomorphic`
  compare structures up to unit relabeling, and `substrate_automorphisms` /
  `substrate_canonical_form` canonicalize substrates by exact
  node-permutation enumeration.

**Matching and perception**

- `pyphi.matching`: the cross-stimulus matching layer of Mayner, Juel &
  Tononi (2024, [arXiv:2412.21111](https://arxiv.org/abs/2412.21111)).
  `PerceptualSystem`
  embeds a system in an environment via a sensory interface; `TriggeredTPM`
  captures the fixed-lag response to each stimulus; `Perception` exposes
  per-distinction, per-relation, and per-fold perceptual richness for a
  stimulus; `Differentiation` computes the differentiation of the triggered
  structures (with a closed-form `analytical_differentiation` that needs no
  relation enumeration); and `MatchingAnalysis.matching()` estimates
  matching as the expected world-minus-noise perceptual-differentiation gap,
  with seeded paired sampling and per-trial raw values on the result.
  Environment generators (`segment`, `point`, `noise`, `superpose`,
  `mixture`) build world distributions compositionally. Bjørn Juel's
  `substrate_modeler` mechanism library is ported into
  `pyphi.substrate_generator` (16 unit mechanisms, six composite-combination
  strategies, and a per-node `create_substrate()` factory), so perceptual
  substrates can be built natively.

**Estimation and uncertainty**

- `pyphi.estimate`: `estimate_substrate(data, *, regime, prior)` builds a
  `SubstratePosterior` (independent Beta posteriors over TPM cells) from
  perturbational transition pairs or an observational trajectory, with a
  `CoverageReport` recording which states the data constrained.
  `phi_posterior()` propagates the posterior through the SIA by Monte Carlo
  and keeps the whole mixture: `p_positive` (the probability the system is
  integrated at all), unconditional and conditional quantiles, the raw Φ
  samples, and the complex-identity categorical — it cannot be coerced to a
  bare float. `SubstratePosterior.edge_probability` gives a graded
  connectivity estimate for estimated substrates, and margin-gated screening
  (`screen_margin`) reuses the posterior-mean complex identity per draw when
  that identity is decisive. Estimation results serialize, and `Provenance`
  records how an estimated substrate was produced.

**Landscapes and optimization**

- `pyphi.landscape`: continuous-parameter analysis of IIT quantities over
  substrate space. `landscape_section()` evaluates the SIA along a parameter
  axis into a tidy DataFrame tracking φ, every discrete selection identity,
  selection margins, and regime boundaries; `perturb()` estimates local
  derivatives and the linearized distance to each kind of selection switch;
  `weight_axis()` builds axes over connection weights. `pyphi.optimize()` is
  a seeded population-based black-box optimizer over substrate weights.
  `pyphi.substrate_generator.random_substrate(n, seed=…)` gives exactly
  reproducible random substrates.

**Bounds**

- `pyphi.formalism.iit4.bounds`: the certified upper bounds of Zaeemzadeh &
  Tononi (2024) on distinction, partition, relation, system, and structure
  quantities, each returning an `UpperBound` carrying its certificate;
  `sum_phi_relations_measured_bound()` and `big_phi_measured_bound()`
  evaluate much tighter bounds on the measured per-atom profile of a
  distinction set in O(|D|·n) with no relation enumeration. The
  `validate_phi_bounds` debug check compares every in-domain IIT 4.0 result
  against the theorem-certified ceilings and raises on violation; because
  the bounds are proven, an in-domain violation can only come from a
  formalism bug.

**Result transparency**

- `result.explain()`: a typed, displayable account of why a Φ/φ/α value came
  out as it did — which short-circuit conditions applied, the winning and
  runner-up partitions and the φ gap between them, the binding direction —
  across IIT 4.0, IIT 3.0, and actual causation. The formalisms'
  short-circuit enums are unified into a single `NullResultReason`.
- `result.diff(other)`: a typed, displayable comparison — Δφ; whether a MIP
  change is real or a reshuffle among tied partitions; gained, lost, and
  changed distinctions, relations, and links; and attribution of differences
  to configuration changes via `ConfigSnapshot.diff`.
- Selection margins at every selection point: the SIA's `partition_margin`
  and per-direction specified-state margins, the mechanism-level
  `partition_margin`, `purview_margin`, and `state_margin`, complex-level
  `exclusion_margin`, and `effectively_tied` flags — reported in
  `explain()`, the display cards, `to_pandas()`, and sweep DataFrames.
- `SystemIrreducibilityAnalysis.intrinsic_information` (ii(s), Mayner et al.
  2026, Eq. 23) and `integrated_fraction` (φₛ/ii(s)).
- Top-level results carry a `provenance` record (pyphi version, git
  revision, timestamp, wall time, dependency versions) alongside the config
  snapshot, so a saved result is self-describing;
  `result.with_provenance(note=…, seed=…)` records your own context.
- An opt-in disk cache for top-level results (`disk_cache_results`): repeated
  computations load instead of recompute, keyed on the system's mathematical
  identity, the result-affecting configuration, and the pyphi version.

**Tie resolution**

- `pyphi.resolve_ties`: a cascade primitive that walks the postulate
  hierarchy (Existence → Intrinsicality → Information → Integration →
  Exclusion → Composition, plus a determinism canonicalization level) and
  resolves ties at the lowest sufficient level, with escalation budgets and
  memoization. Tied sets are carried on results — `SIA.ties`, partition and
  state ties on repertoire analyses, purview ties on MICE, α ties on actual
  causation results — and round-trip through serialization.

**Dynamics and graphs**

- `pyphi.dynamics.simulate()` samples state trajectories from a substrate
  TPM; `settle()` iterates the most-probable-transition map to a fixed
  point; both support clamped units. `pyphi.tpm.is_deterministic()` is
  exposed.
- A networkx bridge: `Substrate.to_networkx()`/`from_networkx()`,
  `System.to_networkx()`, `to_graphml()`/`to_adjacency()` export, and
  topology helpers. Exported graphs default to the TPM-inferred causal
  connectivity, so declared edges the dynamics never realize are dropped.
  `Substrate.to_dbn()` exports a 2-timeslice dynamic Bayesian network.

**Display and visualization**

- Every user-facing result type renders as a structured card — grouped
  sections, readable numbers, collections as tables — in the terminal, and
  as a styled HTML card in notebooks, with verbosity levels controlling how
  much is computed and shown. Node labels are used wherever available, with
  k-ary states rendered as Unicode subscripts (`A₂`). `to_pandas()` exports
  labeled, analysis-friendly data (a `Series` for scalar records, a
  `DataFrame` for collections) from every displayable type, including TPMs,
  partitions, and state specifications.
- `plot_ces` now offers five views sharing one projection layer: the 3-D
  hypergraph (with star-expanded relation faces of every degree, richer
  hover detail, a compositional embedding layout, and a per-degree spectrum
  panel), an inclusion-lattice (Hasse) view, a PCA scatter, a
  distinctions-by-distinctions matrix, and the spectrum summary.
  Analytically-computed structures plot via `max_relations=N` (the N
  strongest relations drawn; node sizes and the spectrum stay exact).
  `highlight_phi_fold()` renders a fold against its parent structure.
  Auxiliary plots (`plot_tpm`, `plot_system`, dynamics, repertoires) return
  figures and axes for composition.

**Serialization and persistence**

- Path-based saving everywhere: `pyphi.save(obj, path)` / `pyphi.load(path)`
  and `.save()`/`.load()` on result types, with the wire format inferred
  from the extension and transparent gzip for `.gz` paths. `SweepResult` and
  `OptimizationResult` serialize with their DataFrames embedded as parquet.
  Script-facing provenance writers (`pyphi.provenance.save_json`,
  `save_npz`, `save_dataframe`) encode parameters in filenames, never
  overwrite, and embed a full provenance record readable via
  `read_metadata()`.

**Examples**

- `pyphi.examples` gains the IIT 4.0 paper's worked networks: the Fig. 1A
  introductory network, all five Fig. 6 architectures, and the Fig. 7
  state-dependence example (including the inactivated variant), built from
  the authors' weight matrices and reproduced to the paper's printed
  precision in the test suite.

**Cluster campaigns**

- `pyphi.campaign`: distribute computations across an HTCondor pool as
  self-contained batch campaigns. `prepare()` writes a campaign directory
  with cost-balanced task packing, per-shard memory requests, and a
  generated submit file; `python -m pyphi.campaign run` executes one task;
  `status()` and `collect()` work purely from output files, so resubmission
  is just `condor_submit` again, and `collect()` reassembles the exact
  result a local run would produce.
- Scoped cause-effect-structure sharding: declare which mechanisms and
  purviews are feasible with `CESScope`/`AxisScope` (explicit lists, order
  bounds, unit containment, a per-order purview-order table), let
  `prepare_ces()` plan mechanism, purview-range, and partition-stride shards
  to a per-job budget, and `collect()` reassembles the exact
  `CauseEffectStructure` — tie sets preserved, congruence and relations
  through the standard path — with a certified scope report: within the
  scope every value is exact, and the excluded remainder is covered by a
  Σφ_r lower bound plus measured upper bounds. The system irreducibility
  analysis can be sharded in the campaign, supplied precomputed, or skipped.
  Multi-cell campaigns sweep substrates × states × subsets × formalisms
  under one shared scope, with per-cell congruence-resolution states.
- A Dask parallel backend: with the `cluster` extra installed and a
  `distributed.Client` connected, `config.parallel_backend = "dask"`
  distributes PyPhi's parallel levels across a cluster.

**MCP server**

- A Model Context Protocol server (`pip install pyphi[mcp]`, `pyphi-mcp`)
  exposes PyPhi to AI assistants: tools to build substrates, run analyses,
  inspect and plot results, estimate cost, control parallelism, and
  prepare/monitor/collect cluster campaigns; a bundled citation-checked IIT
  reference (theory topics, gotchas, migration, parallelization,
  visualization, campaigns); and guided prompts for explaining results,
  porting pre-2.0 code, building a substrate from a description, and
  planning a cluster campaign step by step.

### API changes

**Renames to match the IIT 4.0 paper**

- `pyphi.Network` → `pyphi.Substrate`; `pyphi.Subsystem` → `pyphi.System`;
  `pyphi.network_generator` → `pyphi.substrate_generator`.
- `pyphi.models.Concept` → `pyphi.models.Distinction` (with `Concept` kept
  as an alias for the IIT 3.0 idiom); the canonical query is
  `pyphi.formalism.distinction`; `System` gains `ces()` and
  `phi_structure()` mirroring `sia()`.
- The cause-effect-structure hierarchy now matches the paper's terms: the
  old distinctions-only `CauseEffectStructure` is now
  `pyphi.models.Distinctions`, and the old `PhiStructure` (distinctions +
  relations + SIA) is now `pyphi.models.CauseEffectStructure`. "Φ-structure"
  remains the conceptual term for the cause-effect structure of a complex.
- `pyphi.metrics` → `pyphi.measures` (most registry entries are divergences,
  and the paper says "measure").
- The IIT 4.0 implementation moved from `pyphi.new_big_phi` to
  `pyphi.formalism.iit4`, with IIT 3.0 at `pyphi.formalism.iit3` and the
  actual-causation algorithms at `pyphi.formalism.actual_causation`.
- "Partition" now always means a vertex partition and "cut" an edge cut.
  `pyphi.models.cuts` is now `pyphi.models.partitions`, and the classes are
  renamed:

  | Old name | New name |
  |---|---|
  | `Cut` / `SystemPartition` | `DirectedBipartition` |
  | `KPartition` | `JointPartition` |
  | `Bipartition` | `JointBipartition` |
  | `Tripartition` | `JointTripartition` |
  | `CompletePartition` | `CompleteJointPartition` |
  | `AtomicPartition` | `AtomicJointPartition` |
  | `KCut` / `ActualCut` | `DirectedJointPartition` |
  | `GeneralKCut` | `EdgeCut` |
  | `CompleteSystemPartition` / `CompleteGeneralKCut` | `CompleteEdgeCut` |
  | `GeneralSetPartition` | `DirectedSetPartition` |

  Attributes follow (`System.cut` → `.partition`, `.is_cut` →
  `.is_partitioned`, `SIA.cut_system` → `SIA.partitioned_system`, and
  siblings). The `Cut` replacement takes an explicit `Direction` as its
  first argument; IIT 3.0 callers should pass `Direction.EFFECT` (the IIT
  3.0 φ computation does not read the direction, so values are unchanged).
- The partition-scheme registries are renamed to describe what they yield.
  Mechanism-level: `BI` → `JOINT_BIPARTITION`, `TRI` → `WEDGE_TRIPARTITION`,
  `ALL` → `JOINT_PARTITION_ALL`. System-level: `DIRECTED_BI` →
  `DIRECTED_BIPARTITION` (and its `_CUT_ONE`, `_SEQUENTIAL`, and
  `TEMPORAL_*` variants), `GENERAL` → `EDGE_CUT_ALL`,
  `GENERAL_BIDIRECTIONAL` → `EDGE_CUT_BIDIRECTIONAL`, `SET_UNI`/`SET_BI` →
  `DIRECTED_SET_PARTITION`. The generator functions in `pyphi.partition`
  follow suit.
- `System.cause_tpm` / `effect_tpm` (and the `proper_*` variants) are now
  `cause_marginal` / `effect_marginal`. The new names describe what they
  compute — the causal marginals of IIT 4.0 Eqs. 3–4, with the cause side a
  posterior over past states. Both `proper_*` marginals now return a
  `FactoredTPM`, giving multi-valued substrates a meaningful answer.

**Configuration**

- Configuration is layered into three frozen dataclasses:
  `config.formalism` (with nested `iit` and `actual_causation`
  sub-namespaces), `config.infrastructure`, and `config.numerics`. Reads use
  the layered path (`config.formalism.iit.version`) or flat lowercase
  shortcuts (`config.precision`); scoped changes use
  `config.override(...)` (reentrancy-safe, dotted-path keys accepted);
  `pyphi_config.yml` uses the nested format and is validated on load. Every
  top-level result carries a `ConfigSnapshot`, and
  `config.override(**result.config.as_overrides())` reruns the exact
  recorded computation. The config facade implements the Mapping protocol
  (iterate all leaf settings as dotted paths). The 1.x flat uppercase
  format raises a `ConfigurationError` pointing at the rename map:

  | 1.x flat option | 2.0 layered option |
  |---|---|
  | `IIT_VERSION` | `formalism.iit.version` (`"IIT_3_0"` / `"IIT_4_0_2023"` / `"IIT_4_0_2026"`) |
  | `REPERTOIRE_DISTANCE` | `formalism.iit.mechanism_phi_measure` (with `system_phi_measure` and `specification_measure` siblings) |
  | `CES_DISTANCE` | `formalism.iit.ces_measure` |
  | `PARTITION_TYPE` | `formalism.iit.mechanism_partition_scheme` |
  | `SYSTEM_PARTITION_TYPE` | `formalism.iit.system_partition_scheme` |
  | `RELATION_COMPUTATION` | `formalism.iit.relation_computation` |
  | `ASSUME_CUTS_CANNOT_CREATE_NEW_CONCEPTS` | `formalism.iit.assume_partitions_cannot_create_new_concepts` |
  | `ACTUAL_CAUSATION_MEASURE` | `formalism.actual_causation.alpha_measure` |
  | `PARALLEL_CUT_EVALUATION` | `infrastructure.parallel_partition_evaluation` |
  | `PARALLEL_CONCEPT_EVALUATION` | `infrastructure.parallel_distinction_evaluation` |
  | `CLEAR_SUBSYSTEM_CACHES_AFTER_COMPUTING_SIA` | `infrastructure.clear_system_caches_after_computing_sia` |
  | `PRECISION` | `numerics.precision` |
  | `LOG_FILE` / `LOG_FILE_LEVEL` / `LOG_STDOUT_LEVEL` | removed — use `pyphi.enable_logging()` |
  | other options | same name, lowercase, under `infrastructure` |

- The default formalism is IIT 4.0 (2026): `formalism.iit.version =
  "IIT_4_0_2026"` with `system_phi_measure = "INTRINSIC_INFORMATION"`, which
  applies the intrinsic-information requirement (Eq. 23). System φ values
  may be lower than under IIT 4.0 (2023) where the requirement binds — in
  particular, deterministic systems compute φₛ = 0.
- Each formalism declares its compatible measures and partition schemes, and
  incompatible configurations (for example IIT 4.0 with `EMD`, or IIT 3.0
  with `JOINT_PARTITION_ALL` or `INTRINSIC_INFORMATION`) are rejected at
  configuration time instead of silently computing a different quantity. If
  you previously combined a distribution measure such as `EMD` with an IIT
  4.0 version, switch to `"IIT_3_0"` to keep the same numerical behavior.
- PyPhi produces no log output by default: it attaches a `NullHandler` and
  no longer configures the root logger or writes `pyphi.log`. Enable
  logging explicitly with `pyphi.enable_logging(level=…, file=…)`.

**Relations**

- `relation_computation` defaults to `"ANALYTICAL"`: `ces.relations` is a
  closed-form summary that answers aggregate queries without enumerating
  relations and agrees numerically with the concrete backend. Iterating or
  indexing it raises a guided `TypeError` pointing at `.strongest(k)`,
  `.materialize()`, and the `CONCRETE` setting; use `.num_relations()` for
  the exact count (`len()` is not defined, since the count can exceed
  `len()`'s range). Plotting renders the strongest 1000 relations by
  default when the set is not enumerable.

**Serialization**

- The custom `pyphi.jsonify` layer is replaced by `pyphi.serialize`, built
  on [msgspec](https://jcristharif.com/msgspec/): every result type
  serializes through a typed, tag-discriminated schema to JSON or compact
  binary msgpack, numpy arrays are stored as their exact `.npy` bytes, and
  the cause-effect structure is normalized so distinctions are stored once
  and relations reference them by index (a phi-structure example drops from
  1.3 MB to 56 KB). The per-class `to_json`/`from_json` methods are gone.
  This is a format break: results saved with the old format must be
  recomputed.

**Values and semantics**

- φ, Φ, and α are plain floats with exact comparison semantics. Tolerant
  comparison (up to `config.numerics.precision`) is applied at the decision
  sites: the predicates in `pyphi.numerics` and the tie cascades in
  `pyphi.resolve_ties`, which cluster float keys tolerantly so candidates
  tied up to precision are co-selected regardless of iteration order.
  Structural equality on result objects is precision-aware up to
  `EQUALITY_TOLERANCE = 1e-13`, with hashes structural-only to keep the
  equality/hash contract.
- System-level φ is the paper-faithful non-negative value (the |·|⁺ operator
  of Eqs. 19–20), with the raw signed value preserved as `signed_phi` (and
  `signed_normalized_phi`) for preventative-cause visibility. The
  system-level MIP minimizes |φ|.
- Specified-state, partition, purview, and exclusion ties all follow the IIT
  4.0 S1 tie supplement: state ties escalate through φₛ to structure Φ, a Φ
  tie among relabeling-isomorphic readings reports a canonical
  representative, and a Φ tie among non-isomorphic structures yields a null
  SIA with reason `NONUNIQUE_SYSTEM_STATE`; distinction state ties resolve
  per direction to the congruent MICE and then the largest congruent
  purview; and substrate exclusion applies the recursive cascade of
  Marshall et al. (2023, Algorithm A1), so φₛ-tied candidates that overlap
  only excluded rivals are handled correctly and disjoint tied candidates
  are all accepted. Distinction bags are typed by resolution status
  (`UnresolvedDistinctions` / `ResolvedDistinctions`), so an unresolved
  tied-state pick cannot flow into relation computation unnoticed.
- A direction whose intrinsic information is zero up to the configured
  precision short-circuits as having no cause or effect instead of
  computing through to a noise-level φₛ.
- Actual causation enforces the realization principle (Albantakis et al.
  2019): `Transition` construction raises for occurrence pairs with zero
  probability, and the analysis entry points reject observed state pairs
  impossible under the substrate dynamics. AC is configured by its own
  namespace (`formalism.actual_causation`: `alpha_measure`,
  `mechanism_partition_scheme`, `partitioned_repertoire_scheme`,
  `background_scheme`, `alpha_aggregation`) with paper-faithful defaults,
  independent of the IIT settings.
- IIT 3.0 is restored paper-faithfully: `iit3.ces()` returns a
  `CauseEffectStructure` wrapping the SIA and distinctions (with an empty
  `NullRelations`); the IIT 3.0 SIA class is
  `IIT3SystemIrreducibilityAnalysis`; the `IIT_3_0` preset is a complete
  formalism specification (including
  `background_conditioning="CONDITION_CURRENT_STATE"`, reproducing
  published PyPhi 1.x results on subset systems, and raw-φ mechanism MIP
  selection per the 2014 paper); and canonical reference values are pinned
  in the test suite. The EMD backend is now POT (`pyemd` is deprecated
  upstream); the two agree to machine epsilon, and the IIT 3.0 CES distance
  is reformulated as a proper non-negative optimal-transport problem that
  reproduces the published golden values exactly.

**Removals and smaller changes**

- Removed: `FlatCauseEffectStructure` and `flatten()`/`unflatten()`; the
  concept-style cuts machinery; the legacy `pyphi.macro` module
  (`CoarseGrain`/`Blackbox`/`MacroSubsystem`, replaced by the 2024
  framework); `DistanceResult.__array__` (use the explicit
  `DistanceResult.values_array()`); parent-object back-references on SIA
  result types (the metadata is stored directly, so equivalent results
  compare equal); and the unused Redis cache.
- Dependencies: `graphillion` (concrete relations enumeration is now pure
  Python and free-threading safe), `pyemd` (→ POT), `toolz`, `ordered-set`,
  and `igraph` are dropped; `pyarrow` is now a core dependency.
- `import pyphi` imports submodules lazily: imports are faster, optional
  dependencies are only loaded when used, and `from pyphi import *` works
  on a base install.
- `Substrate.complexes()` follows the paper's meaning of "complex" (a
  non-overlapping local maximum under exclusion); the previous
  every-irreducible-system semantics is `substrate.irreducible_sias`.

### Config

- `formalism.iit.background_conditioning`: how background units enter cause
  repertoires — `"CAUSAL_MARGINALIZATION"` (IIT 4.0 Eq. 4, the default) or
  `"CONDITION_CURRENT_STATE"` (background fixed at its observed state, the
  PyPhi 1.x convention). Actual causation is unaffected by this option; its
  background rule is `formalism.actual_causation.background_scheme`.
- `formalism.iit.shortcircuit_distinctions` (default on): skip the remaining
  MICE search when a distinction is already known reducible; set to `False`
  for exact selection margins and complete ties. Likewise
  `shortcircuit_sia=False` now also disables the sweep-level short-circuit,
  so margins are exact everywhere.
- `infrastructure.validate_config` (default on): eager cross-field
  validation of configuration combinations, applied on `override()`, YAML
  load, and at import time for `pyphi_config.yml`.
- `infrastructure.repr_max_table_rows` (default 50): collection tables in
  result displays truncate with a `… N more` indicator.
- Parallel dispatch thresholds are retuned to measured per-item costs:
  `parallel_partition_evaluation` 1024 → 64,
  `parallel_mechanism_partition_evaluation` and
  `parallel_relation_evaluation` 1024 → 8192.

### Optimizations

- Removed a configuration-write overhead that serialized the entire config
  to disk on every `config.override` and config assignment. Hot paths that
  mutate config — essentially the whole compute pipeline — are ~60–300×
  faster; the full golden suite dropped from ~13 minutes to ~13 seconds.
- The cause-side Bayesian inversion (IIT 4.0 Eq. 4) evaluates as a greedy
  sum-product contraction over the factored TPM's dependence structure
  instead of materializing the joint likelihood over all substrate units,
  making small systems embedded in large sparse substrates tractable on the
  cause side. Densely coupled substrates whose contraction would exceed the
  intermediate-size limit raise `IntractableCauseInversionError` instead of
  exhausting memory.
- The repertoire kernel cache and potential-purview cache are keyed on a
  label-free content fingerprint of the system's mathematics rather than
  object identity: mathematically equivalent systems (reconstructed copies,
  relabelings, same-topology sweeps) reuse each other's results, with
  entries still released when their systems are garbage-collected.
- The specified-state computation no longer materializes the full state
  space: vectorized winner/tie selection and a running-mean unconstrained
  repertoire drop memory from 2ⁿ full repertoires to one, and infeasible
  requests fail immediately with the estimated cost.
- `Substrate.potential_purviews` accepts `max_order`, bounding the purview
  enumeration itself; scoped campaign planning derives the bound from the
  scope, removing the dominant planning cost on large substrates.
- Parallel dispatch now engages whenever the chunker would produce more than
  one chunk, chunk counts are floored at the worker count, and
  heterogeneous sites pack chunks by estimated cost — measured 2.5–4×
  faster partition and purview evaluation in the affected regimes. The
  per-item config-snapshot hash is computed once at dispatch
  (relation-candidate evaluation ~250× faster sequentially).
- Macro TPM construction caches its mapping-independent intermediates per
  substrate (`cache_macro_construction`), so grain-search candidates that
  differ only in their mapping reuse the expensive construction prefix; the
  per-unit loop also hoists its invariants.
- Smaller wins: system-level partition evaluation builds the induced cut
  system once per partition (not per direction) and computes
  `intrinsic_differentiation` once per direction (not per partition);
  measure-shape classification is memoized per measure; `purview_units` is
  cached per analysis; AC `PRODUCT` repertoires share the kernel cache with
  the unpartitioned path; `dynamics.simulate` samples by inverse CDF (~120×
  faster per step); and serialized binary substrates store only the
  on-probability slice of each factor (about half the file size).

### Fixes

- Fixed the IIT 4.0 (2026) intrinsic-information requirement (Eq. 23) being
  applied per partition inside the system-MIP search, which could change the
  selected MIP and make the reported 2026 system φ exceed the 2023 value.
  The MIP is now selected on the integration value exactly as in IIT 4.0
  (2023) and the requirement is applied once to the chosen MIP, so 2026 φₛ
  ≤ 2023 φₛ always holds (a cross-formalism property test guards this).
- Fixed the Eq. 23 differentiation term: `i_diff` is evaluated at the
  specified state (Mayner et al. 2026, Eqs. 4, 6, 12) rather than the
  current state, with the Eq. 11 Bayes normalization applied on the cause
  side. Previously 2026 φₛ was wrong whenever the specified state differed
  from the current state or the dynamics were not doubly stochastic. The
  requirement is also applied consistently across tie candidates, so tie
  sets compare like with like.
- Fixed cause and effect repertoires for multi-valued units: node TPM
  construction used a binary-only rule to choose marginalized dimensions,
  so on k-ary substrates system partitions could fail to sever the
  dependency (under-reporting integrated information) and sparse
  heterogeneous substrates crashed. Verified to machine precision against
  an independent reference and the voting example of Albantakis et al.
  (2019, Fig. 11).
- Fixed the IIT 3.0 SIA/CES parallel dispatch dropping concepts from the
  unpartitioned structure (a truthy options dict was passed as the
  `parallel` boolean), which under-reported φ — e.g. the basic substrate at
  (1,0,0) reported φ = 0.5 instead of the canonical 2.3125.
- Fixed the `EDGE_CUT_ALL` disconnecting-partition filter never being
  installed (a wrong-variable comparison left it unreachable), so edge cuts
  that do not disconnect the system were included in the MIP search,
  violating Eq. 14; and fixed `EDGE_CUT_BIDIRECTIONAL` omitting half of the
  valid bidirectional cuts for systems of 4 or more units.
- Fixed two uninitialized-memory reads: `pointwise_intrinsic_differentiation`
  called `np.log2(p, where=…)` without an output buffer, so leftover memory
  contents in zero-probability slots could corrupt the reported minimum; and
  `forward_cause_repertoire` with an explicit `purview_state` left its
  uncomputed entries as whatever the buffer held (now NaN). Related
  precision fixes: `intrinsic_differentiation` excludes surprisals that are
  zero up to the configured precision (a probability of 1 up to float noise
  previously produced a spurious ~3e-16 minimum), and specified-state tie
  membership is clustered within the precision rather than compared
  exactly.
- Fixed permutation-symmetry breaking from arbitrary tie-breaking among
  specified states: tied states are evaluated and the minimum taken, with
  the resolved state back-propagated to the SIA. More broadly, `sia()`
  results are deterministic across runs, parallel backends, and worker
  scheduling: parallel evaluation restores canonical enumeration order
  before tie resolution, short-circuited sweeps collect in submission
  order, and worker exceptions cancel the remaining chunks.
- Parallel workers now install the parent's config snapshot, so
  computations under `config.override(...)` (for example an IIT 3.0 pin)
  no longer run workers under the default configuration.
- Fixed the exclusion cascade on chain topologies: candidates beaten only by
  rivals that themselves lost to a stronger complex were missing from
  `complexes()` (both the substrate-level and macro drivers now apply the
  recursive cascade), and φₛ-tied candidates overlapping only excluded
  rivals are no longer dropped without record.
- In actual causation, cause-direction background units are conditioned on
  the after-state rather than the before-state, keeping both halves of the
  inversion anchored to the same time; this changes results only for
  partial analyses (whole-network analyses, including the paper's worked
  examples, are unaffected). AC partition enumeration reads its own
  configured scheme instead of inheriting the IIT setting.
- Serialization round-trips every computed field: signed φ values,
  selectivity, short-circuit reasons, tie sets, runner-up partitions,
  config snapshots and provenance, and noised transitions all survive
  save/load, and loading a file written by a newer format version raises
  instead of dropping the unknown fields.
- Caching safety: cached repertoire arrays are read-only (caller mutation
  raises instead of corrupting later computations); `FactoredTPM` copies
  and freezes its factors; the disk result cache keys on every
  result-affecting config field (previously eight hand-picked ones), is
  best-effort on write failures, and decodes hits with the requesting
  system's node labels; and cache eviction is thread-safe under
  free-threaded Python.
- Construction-time validation now catches inputs that previously produced
  wrong results without an error: TPM probabilities are range-checked;
  state-by-state conversions reject non-power-of-two state counts instead
  of truncating; reduced-dimension factors are rejected with a clear error;
  mismatched mechanism/state lengths raise; unreachable system states are
  rejected at construction (restoring the pre-2.0 behavior for candidate
  subsystems); and macro, matching, and estimation entry points validate
  their preconditions.
- Fixed `convert.be2le_state_by_state()` / `le2be_state_by_state()` (columns
  were not permuted) and an operator-precedence bug in
  `propagation_delay_substrate` (128 of 512 rows of unit D's XOR were
  wrong).
- Closed-form subset counts share one saturating overflow policy: counts
  stay exact through int64's range and saturate to `inf` beyond float64's,
  so `AnalyticalRelations.sum_phi()` and the measured bounds are correct on
  structures where many distinctions share an atom (previously int64
  wrap-around could corrupt Σφ_r, and the bounds raised `OverflowError`
  past 1023 values).
- Many smaller fixes to features new in 2.0 — campaign transfer contracts,
  fold sums, diff/explain edge cases, plotting and MCP crashes on valid
  input, star-import and `__all__` completeness — are recorded in the git
  history.

### Documentation

- The documentation toolchain is rebuilt: pydata-sphinx-theme, MyST Markdown
  with build-time-executed code cells, notebook pairing with Colab links, an
  API reference generated from the current module layout on every build, and
  a docs CI job that fails on any warning or failed cell.
- The documentation is overhauled for the IIT 4.0 (2026) default: getting
  started and the theory pipeline run on the paper's Fig. 1A network, a
  worked tutorial follows the paper's Figs. 1→2→4, and a theory page
  explains the intrinsic-information requirement and why deterministic
  systems compute φₛ = 0.
- New theory pages and tutorials: computational complexity (per-stage cost
  derivations for every formalism, confirmed empirically, and which
  configuration settings extend the tractable system size), macro units and
  exclusion across grains, recursive exclusion, and the intrinsic-units
  tutorial.
- New how-to guides: sweeps, selection margins and tie-breaking, substrate
  parameter landscapes, the relations query interface, visualization, the
  grain search, cluster campaigns, and running PyPhi on an HTCondor cluster
  (CHTC).
- A migration guide for pre-2.0 code ships as an MCP reference topic with a
  matching `migrate_code` prompt.

### Refactor

- The computational core is layered: stateless repertoire algebra and TPM
  kernels in `pyphi.core`, formalism strategy objects in `pyphi.formalism`
  (IIT 3.0 / IIT 4.0 / actual causation), and frozen `Substrate` / `System`
  value types — with runtime-checkable Protocols and registration-time
  validation at the dispatch points, and an architectural test pinning the
  models tier as pure data.
- The models tier is one concept per file, and heavy parent back-references
  are gone from result types: results store the metadata they need, so
  mathematically equivalent results compare equal and serialize compactly.
- Caching has a single observability interface: `pyphi.cache.info()` reports
  per-cache statistics, `clear_all()`/`clear(name)` reset them, and the
  kernel cache respects the configured memory limit.
- Parallel execution is unified on a typed `Scheduler` Protocol with one
  `map_reduce()` path (process, thread, and Dask backends;
  `parallel_backend="auto"` selects threads on free-threaded runtimes), and
  a free-threaded Python CI lane runs the full suite with the GIL disabled.
- Development infrastructure: a golden regression harness (raw numerical
  outputs across 25 fixtures spanning all three formalisms, byte-stable),
  Hypothesis property-based invariants from the IIT 4.0 paper,
  paper-reproduction suites (IIT 4.0 Figs. 6–7, the 2019 actual-causation
  examples, the matching manuscript environments), deterministic call-count
  regression tests, an ASV benchmark suite with nightly CI, and a test
  suite reorganized to mirror the package layout.
