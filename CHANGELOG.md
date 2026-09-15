Changelog
=========

<!-- Towncrier will insert release notes here. See changelog.d/ for fragments. -->

2.0.0
-----
_2026-09-15_

PyPhi 2.0 is a comprehensive rework of the library around IIT 4.0 (Albantakis et
al. 2023; Mayner et al. 2026): new core value types, first-class formalism
objects covering IIT 3.0, both IIT 4.0 variants, and actual causation,
multi-valued units, closed-form relations, distributed computation, and
rebuilt configuration, serialization, and display. Changes are described
relative to PyPhi 1.2.0.

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
- **Published worked examples reproduce.** The examples reproduce figures from Albantakis et al. (2023) and (2019), Mayner et al.
  (2026), Marshall et al. (2023), Barbosa et al. (2020), Oizumi et al. (2014),
  and Gómez et al. (2020) to their published values, with every deviation from a quoted value documented alongside it.

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
  including heterogeneous per-node alphabets. Substrates are constructed via
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

- PyPhi now prints a short note to stderr when it is imported under an AI coding
  agent, naming the two mistakes that most often go wrong unaided — reporting φₛ
  as Φ, and the little-endian state convention — and pointing at the bundled
  reference. It is written to stderr, never stdout, so it cannot corrupt the MCP
  server's JSON-RPC stream. Suppress it with `PYPHI_AGENT_NOTE_OFF=1` or the new
  `agent_note_off` infrastructure option; `welcome_off` controls only the welcome
  message. Any harness can opt in by setting `PYPHI_AGENT`.
- `Analysis` — the type `pyphi.analyze()` returns — is now serializable. It
  round-trips through `pyphi.serialize.dumps`/`loads` under every formalism and
  gains `.save()` / `.load()`, so a whole analysis can be written to disk and read
  back with its system, SIA, and cause-effect structure intact. Previously only
  the `.ces` component could be saved.
- Campaign task outputs now record what the task cost to run.
  `CampaignTaskOutput.metrics` carries wall and CPU seconds, cache
  hit/miss/eviction counts, and the shard's planned units, payload kind, and
  memory request — enough to recalibrate `pyphi.cost` against a campaign's own
  observed runtimes.
- `pyphi.analyze(..., compute="distinctions")` computes a system's distinctions
  without the system-partition search, and the MCP server's `analyze` tool takes
  the same option. Under IIT 4.0 unfolding a cause-effect structure computes a
  system irreducibility analysis first, and over a sparse substrate that search
  is most of the running time; on the nine-unit `propagation_delay` example the
  search is nearly all of it.

  The distinctions come back filtered for congruence with the system's specified
  state, exactly as a cause-effect structure filters them, whenever that state is
  untied — the specified state a system irreducibility analysis would start from
  is available without evaluating a single partition. When the state ties, the
  tie is broken by the φₛ cascade over the tied cause/effect pairs, which does
  need the partition search; the unfiltered distinctions are returned instead, as
  an `UnresolvedDistinctions` rather than a `ResolvedDistinctions`. The two are
  worth telling apart: congruence filtering can remove any number of
  distinctions, including all of them, so an unfiltered count and Σφ_d are upper
  bounds on the structure's rather than estimates of it. The MCP tool reports
  which of the two it has under a `congruence` key, and renames the unfiltered
  counts to `num_distinctions_upper_bound` and `sum_phi_distinctions_upper_bound`
  so they cannot be read as the structure's.

  `System.distinctions()` gains a `congruent` argument for the same thing, and
  `pyphi.cost.estimate_analysis` accepts `compute="distinctions"`.
- `pyphi.dynamics.simulate()` and `mean_dynamics()` accept a `seed` argument for reproducible trajectories.
- Added `pyphi-mcp install`, which sets the MCP server up in a project: it
  registers the server in the client's configuration and writes a short block of
  PyPhi facts — φₛ versus Φ, little-endian states, the cost of an analysis — into
  the project's `AGENTS.md`, with an `@AGENTS.md` import added to `CLAUDE.md`
  because Claude Code reads that name instead. A server's `instructions` reach
  only an assistant that connects to it, while a project's instruction file is
  read before anything else happens. The block sits between markers so a later
  install refreshes it without touching surrounding content, and
  `pyphi-mcp uninstall` removes both halves. `--from`, `--scope`, `--client`,
  `--print` and `--force` cover the variants; with no subcommand `pyphi-mcp`
  still runs the server.
- `pyphi-mcp install` offers to install two agent skills, `iit` and `pyphi`, into
  Claude Code, Codex and Cursor. `--skills` and `--no-skills` answer the prompt
  without a terminal; `--agent` and `--agent-path` reach agents that were not
  detected. `pyphi-mcp uninstall` removes them.
- Added `pyphi.numerics.lt()` and `le()`, tolerant order predicates consistent with `eq()`: `lt` requires a difference beyond `config.numerics.precision`, and `le` is `lt` or tolerant equality. The binding-direction selection in `Distinction.explain()` now routes through `le` instead of a hand-rolled composition.
- `pyphi.campaign.prepare_ces()` accepts a `workloads` mapping, planning
  shards against caller-supplied per-mechanism costs instead of the analytic
  counting walk. Useful when measured runtimes describe a workload better than
  the model does.
- `MacroSystem` and `ComplexesResult` (the return type of `macro.complexes()` and
  `analyze(grains=...)`) can now be saved and loaded with `pyphi.serialize`.
  Previously `save()` raised `TypeError: No serializer registered`. The stored
  `MacroSystem` carries the macro construction — units, micro substrate and
  history, and the construction's cause TPM — so a reloaded system reproduces the
  original's repertoires without recomputation.
- `pyphi.cost.estimate_analysis` now counts the specified-state search as its own
  work axis, `specified_state_evaluations`. The search maximizes intrinsic
  information over the whole system as both mechanism and purview, so it performs
  two forward repertoire evaluations per system state — a cost that grows with
  the size of the system rather than of any mechanism, and one that none of the
  partition or purview axes bound. The MCP server's `analyze` guard checks the
  new axis, so an analysis whose cost is dominated by the search is refused with
  that reason named instead of being admitted on modest partition counts.
  The `performance` reference topic now calls the axis out, since it is the one
  that dominates a large *sparse* system: thinning connectivity shrinks every
  other axis through purview pruning but leaves this one untouched.
- Added `intrinsic_specification` on `StateSpecification` and, per direction, on
  the IIT 4.0 `SystemIrreducibilityAnalysis` — the name Mayner et al. (2026,
  Eqs. 7 and 9) give the quantity Albantakis et al. (2023) call intrinsic
  information — alongside the existing `intrinsic_differentiation`, so both
  terms of the intrinsic-information requirement (2026, Eq. 13) are readable by
  their paper names. `to_pandas()` on the SIA gains the four per-direction
  columns.
- `explain()` on an IIT 4.0 system analysis now reports when the
  intrinsic-information requirement (Mayner et al. 2026, Eq. 23) set φₛ, naming
  the direction and the term — intrinsic differentiation or intrinsic
  specification — whose value is the minimum. The finding never fires under
  formalisms without the requirement.
- Added `Substrate.inactivate(fixed)`: returns a copy with the given units (by
  index or label) frozen in a state and conditioned into every other unit's
  dynamics — the lesion Albantakis et al. (2023, Fig 7C) call inactivation,
  distinct from an inactive unit and from a background condition. The Fig 7C
  example substrate is built with it.
- The paper-reproduction acceptance suite now pins Mayner, Marshall & Tononi
  (2026): the monad's φₛ peak (Fig 2), the complex-size and differentiation
  crossovers of the Fig 6D lattice under a determinism sweep (Fig 3), and the
  macro/micro crossover of the intrinsic-units example (Fig 4). New example
  `mayner_2026_monad_substrate`; `iit4_2023_fig6d_substrate` takes the logistic
  slope `k`.
- The paper-reproduction acceptance suite now pins Marshall et al. (2023),
  System Integrated Information: determinism and degeneracy (Fig 1), fault lines
  and integrated fractions (Fig 2), and the eight-unit universe condensing into
  three complexes (Fig 3), with new `marshall_2023_fig1_substrate`,
  `marshall_2023_fig2_substrate`, and `marshall_2023_fig3_substrate` examples.

- The paper-reproduction acceptance suite now pins the causal accounts of
  Albantakis et al. (2019), "What caused what?", Figs 7–16 — including the
  three-candidate election of Fig 11, the suite's first multi-valued
  actual-causation reproduction (new example
  `ac_2019_three_candidate_election_substrate`).
- The intrinsic-difference measure is now pinned against the channel and neuron
  examples of Barbosa et al. (2020), "A measure for intrinsic information", Figs
  2–4.
- `Analysis.formalism` names the formalism that produced a result
  (`"IIT_4_0_2026"`, `"IIT_4_0_2023"`, or `"IIT_3_0"`), and the analysis card
  shows it.
- The MCP `analyze` tool takes a `subset` (node indices or labels) to analyze a
  candidate system inside a larger substrate, and its summary and card now carry
  the minimum information partition, the intrinsic information ii(s), and which
  term of the intrinsic-information requirement set φₛ when it did.
- The package ships a `py.typed` marker, so type checkers use PyPhi's type hints
  in downstream projects.
- The MCP server has a `documentation` reference topic describing how to read
  the documentation site programmatically (llms.txt, per-page sources, the API
  inventory), and the primer, migration, and visualization topics link to the
  pages they condense.
- MCP substrate summaries (`load_example`, `describe_substrate`) now include the
  transition probability matrix as state-by-node rows for substrates of up to
  256 states. `describe_substrate` also reports the installed PyPhi version, so
  an assistant can tell which release it is talking to.


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
  | `CompleteSystemPartition` / `CompleteGeneralKCut` | `TotalCut` |
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

- In-memory caches now evict under memory pressure instead of freezing. Once
  resident memory reaches `memory_ceiling_bytes` (or
  `memory_ceiling_percentage`), a cache used to refuse every new entry for
  the rest of the process, so which results stayed cached was decided by whichever
  happened to be computed first. It now holds its occupancy at the level it had
  reached and admits new entries by discarding the least recently used ones. On a
  scoped cause-effect structure sweep with the ceiling binding over the last 58%
  of the work, that cut the cost of the ceiling from 1.46× the unbounded run to
  1.04×, and raised the hit rate from 72.6% to 95.5% against an unbounded 95.6% —
  while holding fewer entries and less resident memory than freezing did.

  Occupancy is measured in bytes. An entry too large to fit the whole budget is
  skipped rather than allowed to displace everything else, and a ceiling reached
  during a transient spike is re-checked and lifted if memory frees up again.

  Eviction does not lower resident memory — freeing Python objects returns their
  memory to the process allocator, rarely to the operating system. What it changes
  is which entries a fixed allocation is spent on.

  `pyphi.cache.info()` now reports `nbytes` and `evictions` alongside hits, misses,
  and entry count.
- The MCP server's always-loaded instructions now carry the gotchas reference
  alongside the primer, so the mistakes that produce wrong results — reporting φₛ
  as Φ, reading a state as big-endian, treating Φ = 0 as "no structure" — are in
  front of the assistant before its first tool call instead of waiting behind
  `get_iit_reference("gotchas")`. The primer's abbreviated version of the same
  material is removed. The other topics remain on demand.
- `pyphi-mcp install` now registers the Python interpreter it was run with
  (`python -m pyphi.mcp`) instead of a `uvx` command resolving `pyphi[mcp]`. The
  client starts the server from the environment PyPhi was installed into, with no
  `PATH` lookup and no package resolution at startup. Pass `--from
  <specification>` for the `uvx` form. Running `install` from the throwaway
  environment that `uv run --with` or `uvx` builds is refused, since a client
  could not launch it again.
- Removed the `TEMPORAL_DIRECTED_BIPARTITION` and `TEMPORAL_DIRECTED_BIPARTITION_CUT_ONE` system partition schemes. They enumerated every split in both causal directions, but no evaluation path reads a system partition's temporal direction, so the direction pairs computed identical results and the schemes were unused throughout the library.
- Campaign work units now weight their two axes by measured cost. A purview
  evaluation is charged `pyphi.cost.PURVIEW_EVALUATION_UNITS` (12) rather than
  1, which is what it costs relative to one mechanism partition, so a unit
  means the same amount of work whichever rung of the shard-planning ladder
  produced the shard carrying it. `pyphi.cost.SECONDS_PER_UNIT`,
  `units_for_runtime()`, and `runtime_seconds()` convert between units and CPU
  seconds, so `units_per_job` can be set from a per-shard runtime target.
- Renamed the measure and formalism Protocol attribute `applies_ii_cap` (and the
  formalism's `requires_ii_cap`) to `applies_intrinsic_information_requirement`,
  matching the project's terminology for Mayner et al. (2026) Eq. 23.

- `pyphi.cost.estimate_analysis` now takes `subset`, `compute`, `limit`, and
  `scope` by keyword only, so a state passed by mistake raises instead of being
  read as the candidate subset.
- Under IIT 3.0, `System.ces()` and `analyze().ces` return a
  `ResolvedDistinctions` (its concepts under `.concepts`) rather than the
  internal `UnresolvedDistinctions`.
- Result cards align every key/value section to one label-column width, so
  values line up down the whole card in both the text and the HTML rendering.

- Numeric columns in display tables, such as TPMs in HTML cards and text output,
  are aligned on the decimal point.
- Result objects print their one-line compact form when nested in a list, tuple,
  or dict under IPython and Jupyter, instead of stacking full cards.
- The Analysis card's System section now names the units, the current state, and
  the specified cause and effect states ahead of the MIP.
- Result-card summary headers render as real two-column tables, so converting a
  card to plain text keeps each label paired with its value.

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
- Parallel dispatch thresholds are returned to measured per-item costs:
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

- The cache memory check no longer builds a new `psutil.Process` on every call.
  It runs on every cache miss, and constructing the handle cost about ten times as
  much as reading resident memory from an existing one: 14.5 µs per call against
  1.4 µs. On a scoped cause-effect structure sweep making 1.3 million misses, that
  was 19 seconds of a 208-second run, and the share grows the more the cache
  misses. The handle is now reused, and rebuilt after a fork.
- `FrozenMap` now compares itself to another `FrozenMap` by comparing the two
  underlying dicts directly. The equality it inherits from `Mapping` first
  rebuilds a dict from each operand one key at a time, which every cache lookup
  pays for on a hit — 18.4 million times while unfolding the distinctions of a
  6-unit system. Comparison against any other kind of mapping still uses the
  inherited equality, so comparing a `FrozenMap` to a plain dict behaves as
  before. Unfolding the IIT 4.0 Fig 6D distinctions is about 9% faster.
- Non-binary EMD no longer reloads its Hamming ground-distance matrix from the joblib filesystem cache on every call: matrices for small state spaces are memoized in memory, mirroring the precomputed binary path.
- The mechanism-MIP search no longer builds every candidate partition before
  evaluating any of them. The search stops at the first reducible partition, but
  the full set was constructed up front regardless of where it stopped: for a
  fully connected 6-unit system, 31.9 million partitions built to evaluate 4.4
  million, with the full-system pair alone holding a list of 2.2 million
  partition objects. Partitions are now built one at a time as the search
  consumes them. The total, which the search needs in order to tell an
  exhaustive pass from one that stopped early, comes from
  `pyphi.cost.partition_sweep_count`; it is memoized. Unfolding the distinctions
  of the IIT 4.0 Fig 6D system takes 187 s rather than 233 s and peaks at 0.8 GiB
  rather than 2.0 GiB, with every φ, MIP, specified state and partition margin
  unchanged.
- `pyphi.combinatorics.num_subsets_larger_than_one_element` is no longer memoized,
  and so no longer carries `cache_info()` / `cache_clear()`. It evaluates
  `2**n - n - 1` in about 109 ns, against roughly 250 ns for the cache lookup that
  was wrapping it, so caching it cost more than it saved.
- Passing more than one iterable to `map_reduce()` no longer disables cost sampling. Multi-iterable workloads are now sampled on zipped argument tuples, so they get a cost-based chunksize instead of the previous silent fallback to one item per chunk (which dispatched one future per item) and stay sequential when the sampled cost is too small to amortize dispatch.
- The cost sampler that chooses a chunksize for `map_reduce()` no longer discards the results it computes: when collection order is unconstrained (no `ordered=True` and no short-circuit predicate), the sampled items' results are folded into the output instead of being computed a second time.
- The process scheduler now hashes the configuration snapshot only when chunks are actually submitted to worker processes. `map_reduce()` calls that resolve to sequential execution no longer pay the ~1 ms snapshot-hashing cost per call.
- `unconstrained_forward_cause_repertoire` and
  `unconstrained_forward_effect_repertoire` are no longer memoized:
  `intrinsic_information` requests each `(mechanism, purview)` pair once, so
  their caches stored entries that were never read back. The per-state loop
  that carries the real cost keeps its own caching one level down, in
  `effect_repertoire`. No computed value changes.

### Fixes

- Fixed the IIT 4.0 (2026) intrinsic-information requirement (Eq. 23) being
  applied per partition inside the system-MIP search, which could change the
  selected MIP and make the reported 2026 system φ exceed the 2023 value.
  The MIP is now selected on the integration value exactly as in IIT 4.0
  (2023) and the requirement is applied once to the chosen MIP, so 2026 φₛ
  ≤ 2023 φₛ always holds.
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
- `Account` no longer declares itself orderable. It never implemented an ordering, so comparisons raised `NotImplementedError`; they now raise the standard `TypeError` for unorderable types.
- `pyphi.utils.all_states(())` now yields the single empty state instead of
  nothing — the empty product has exactly one assignment — fixing crashes on
  computations over empty unit sets and fully-clamped systems.
- Isomorphism and canonical-form comparisons in `pyphi.automorphism` now
  bucket values at `config.numerics.precision` instead of a hardcoded 12
  decimals, and the canonicalization cache is keyed on the precision.
- The welcome message is written to stderr rather than stdout. The `pyphi-mcp`
  server speaks JSON-RPC over stdout, so importing PyPhi emitted the banner into
  the protocol stream ahead of the first message unless the user had set
  `PYPHI_WELCOME_OFF`. Scripts that captured the banner from stdout will no longer
  see it there.
- The in-memory cache byte bound now charges an `ndarray` view the buffer it keeps alive, instead of zero bytes. The bound previously undercounted memory whenever the cache held a view whose base array was not itself a cache entry.
- `resolve_ties.cascade` now honors its documented contract when the
  escalation budget blocks a level: a lone surviving candidate resolves
  instead of returning unresolved, and `on_unresolved='fail'`/`'warn'` raise
  or warn on a budget-blocked tie just as they do when the cascade exhausts
  its levels. The `"NONE"` tie-resolution strategy is also accepted in list
  form (`["NONE"]`), matching the bare-string form instead of raising
  `NotImplementedError`.
- `pyphi.cost.estimate_analysis(substrate, compute="ces")` now counts the
  system-partition axis under IIT 4.0, where unfolding a cause-effect structure
  computes a system irreducibility analysis before it unfolds anything (Eq. 57).
  It previously reported `system_partitions=None` for that analysis under every
  formalism, which was right only for IIT 3.0, whose cause-effect structure is
  the bare distinctions.

  The MCP server's `analyze` guard inherited the undercount, and refused an
  analysis on whichever single axis matched the requested `compute`. It now
  weighs every axis the analysis walks against that axis's own limit. A sparse
  substrate could be trivial on the distinction axis and enormous on the
  system-partition axis, so a `compute="ces"` request that ran for hours was
  waved through on a count of a few dozen mechanism-partition sweeps. Such a
  request is now refused without `confirm_large`, and the refusal points at
  `compute="distinctions"`, which skips that axis.
- `InducedSubstructure` and `PhiFold` views no longer compare equal to the `CauseEffectStructure` they view (or to views of another kind): equality is exact-type, since views cannot be saved and a fold's relations are incident rather than closed.
- The cache memory limit is now measured against the memory the process is
  actually allowed. `memory_ceiling_percentage` took its denominator from
  total physical memory, which is no bound at all on a process confined to a
  smaller allocation — a scheduler-managed job, a container, a cgroup. It now
  reads the process's cgroup allowance (v2 `memory.max`, falling back to v1
  `memory.limit_in_bytes`, and to the hierarchy root inside a container's cgroup
  namespace, taking the smallest limit along the hierarchy), and uses physical
  memory only when no allowance is reported.

  Because the ceiling follows the memory actually granted, asking a scheduler for
  more memory now grows the caches to match rather than leaving the extra as free
  headroom. Pin `memory_ceiling_bytes` alongside the larger request to buy
  headroom without growing them.

- The conditional-independence check on state-by-state TPMs now runs at the
  configured `numerics.precision` (absolute tolerance) instead of numpy's
  loose defaults, which silently accepted dependence up to ~1e-5.
- `pyphi.cache.clear_all()` no longer deletes the persistent on-disk result cache; it clears only in-memory caches, which is its purpose (recovering memory). Clear the disk store explicitly with `pyphi.cache.clear("disk.results")` or by deleting the `__pyphi_cache__/` directory.
- `CompositionalState` fixes: an empty (no-argument) state is fully usable; a
  purview claimed in only one direction no longer raises KeyError from
  `conflicts_with`; and `resolve_conflicts` ranks candidates by live conflict
  counts as resolution proceeds, keeping mechanisms the stale ranking used to
  discard.
- Loading a `pyphi_config.yml` with an empty layer section no longer crashes
  with a bare AttributeError, and the unknown-option error now points at the
  migration guide — naming the replacement when the option was renamed.
- Documented that `config.override()` applies to the whole process, not the
  current thread: while an override is active, every thread reads the
  overridden values, so concurrent computations under different configurations
  must use separate processes (PyPhi's process-based parallel backends already
  give each worker its own configuration copy). Also fixed the one internal
  misuse: the macro grain search opened an override inside each parallel
  worker, which raced on the shared configuration under the thread backend;
  the override is now a single parent-side scope around the dispatch.
- `pyphi.cost.estimate_analysis` now accounts for
  `system_partition_include_total`: the partition-count memo is keyed on the
  option, so estimates are correct (and the memo cannot be cross-poisoned)
  when the total cut is included.
- The one-line display of a `DirectedBipartition` now always points the cut arrow from `from_nodes` to `to_nodes` — the connections the cut severs — and annotates the causal direction textually. Previously a CAUSE-labeled cut drew the arrow reversed relative to the severed-connections grid.
- Disk-cache writes now use a per-call temporary filename, so two threads writing the same key no longer collide on a shared temp path (which raised `FileNotFoundError`).
- `pyphi.dynamics` now handles explicit-alphabet TPMs (the `(*alphabet_sizes, n_units, max_alphabet)` layout produced by `Substrate.joint_tpm()`): `simulate()`, `settle()`, `mean_dynamics()`, `most_probable_next_state()`, and `number_of_units()` previously misread the layout as a binary state-by-node TPM and returned wrong-length binary states. Random initial states are now drawn from each unit's own alphabet, and `simulate()`/`settle()` reject an `initial_state` of the wrong length.
- Tie-resolution strategy names (`state_tie_resolution`, `mip_tie_resolution`, `purview_tie_resolution`, `sia_tie_resolution`) are now validated against the registered strategies at configuration time, so a typo fails immediately instead of mid-computation.
- The `EDGE_CUT_BIDIRECTIONAL` system partition scheme now applies the same disconnection filter as `EDGE_CUT_ALL`: cuts that leave the system strongly connected are excluded from the MIP search, as required by Eq. 14 of Albantakis et al. (2023). Previously a non-disconnecting cut could win the MIP and report φ_s = 0 for an irreducible system.
- `EdgeCut` hashing now normalizes the cut-matrix dtype, so equal cuts built from matrices of different dtypes hash equally, as required by their dtype-insensitive equality.
- Corrected four figure-citation and content errors in `pyphi/examples.py` example docstrings: `disjunction_conjunction_substrate` now cites Actual Causation Figure 9 (disjunction of two conjunctions), not Figure 7 (which shows separate disjunction/conjunction/biconditional/prevention panels); `prevention_transition` now cites Actual Causation Figure 7D, not Figure 5D (which is the unrelated OR/AND irreducibility example); `iit4_2023_fig6e_substrate` now names the units whose inputs were perturbed relative to Fig 6D as C, D, and E (matching the weight matrix), not C, D, and F; `iit4_2023_fig7_substrate` now describes the perturbed connection as A <- D, not D <- A (matching the weight matrix, which perturbs D's output into A).
- Fixed `FactoredTPM` storing read-only views (e.g. from `numpy.broadcast_to`) without copying: mutating the view's source array after construction could silently change the stored factors, computed results, and the hash. Read-only views are now copied; read-only arrays that own their data are still stored without a copy.
- Actual-causation background conditions now match the paper's causal model in
  both directions: units outside the cause set have their inputs to the
  transition fixed at the observed before-state (Albantakis et al. 2019,
  Section 3.3 — the background U is set to u throughout), for cause repertoires
  as well as effect repertoires. Previously the cause direction integrated the
  background's past states under the posterior implied by the observed present,
  which deviates whenever the background's own dynamics are informative
  (Figure 8B's cause link came out 1.2345 bits instead of 3.0 on such
  backgrounds; every published example has static backgrounds, where the two
  readings coincide). `noise_background=True` now marginalizes background
  inputs uniformly on the cause side too, as documented. `System` gains a
  `background_state` field for conditioning external units at a state other
  than the evaluation state.
- The `JOINT_PARTITION_ALL` mechanism partition scheme now yields each induced edge cut exactly once. Structurally distinct part assignments severing the same edges (e.g. the complete cut written as one mechanism part or several, each over an empty purview) describe the same physical partition; the redundant forms tied exactly in actual causation, exhausting the tie cascade so the MIP search returned `None` and silently dropped the purview from the causal-link search. The AC cascade also gained a backstop that resolves identical-cut survivors instead of returning `None`. Deduplication shrinks mechanism partition sweeps (e.g. 146 → 121 forms at mechanism size 3, purview size 3); .
- `pyphi.actual.account()` now honors `allow_neg=True`; previously the flag was silently dropped on the bidirectional path.
- `Account` equality and hashing are now order-insensitive: two accounts holding the same causal links compare equal regardless of construction order.
- Fixed three thread-safety and accounting defects in the cache layer. The `@cache()` decorator's hit path could raise `KeyError` when worker threads hit the same key concurrently, and a byte-bounded cache's eviction loop could raise `RuntimeError` when a concurrent hit moved an entry during eviction; both paths are now safe under the thread scheduler. Clearing a cache through `pyphi.cache.clear()` or `pyphi.cache.clear_all()` now resets its byte-weight accounting and admission budget along with its entries — previously a cleared cache could report stale occupancy and permanently refuse new entries.
- The configuration validator now checks `ces_measure` against the active formalism: IIT 3.0 accepts `EMD` and `SUM_SMALL_PHI` (the Gómez et al. 2020 multi-valued variant); IIT 4.0 accepts `SUM_SMALL_PHI`. Previously an incompatible pairing was accepted and silently computed a different Φ.
- The complete cut's normalization factor now agrees with the number of
  connections it severs: 1/n² (all connections, self-loops included), by the
  same rule as every other edge cut, instead of the previous 1/n. The complete
  cut represents total unconstraining of the system's cause-effect power and
  is distinct from the all-singletons directional partition (self-loops
  intact), which the default partition family already enumerates with its own
  correct normalization. Single-unit systems are unaffected (the factor is 1
  either way), so default-configuration results do not change; the complete
  cut becomes proportionally more competitive as a MIP candidate under
  `system_partition_include_total` and the edge-cut schemes.
- `Complex` equality now includes `node_indices` (the micro footprint), so equal complexes hash equal and macro complexes over different micro constituents compare unequal.
- `complexes()` now evaluates candidates in deterministic enumeration order under parallel complex evaluation; previously worker-completion order could report a different major complex run-to-run when candidates tie.
- Loading a saved result now restores its `config` as a `ConfigSnapshot` instead of a plain dict, so `diff()`, `as_overrides()`, and display labels work on loaded results exactly as on fresh ones. Previously a loaded result's `diff()` raised `AttributeError`, the documented rerun recipe `pyphi.config.override(**result.config.as_overrides())` failed, and a loaded IIT 3.0 analysis displayed its Φ value under the φ_s label. The parallel-evaluation mappings in the stored config are now saved losslessly (files written earlier, which stored them as repr strings, still load).
- The `DIRECTED_BIPARTITION_CUT_ONE` system partition scheme (and its temporal variant) no longer yields the same two-node bipartition twice.
- `Distinction` equality and hash now include the specified cause/effect
  purview states. Two readings of the same purview specifying different states
  carry different cause-effect power — they support different relations and
  different structure Φ — but previously compared equal and collapsed in sets,
  so two Φ-structures with different Φ could compare equal.
- Distinction tie resolution now follows the S1 postulate cascade over every
  tied reading. Congruence with the system's specified state is a requirement
  (a non-congruent reading is excluded); among the congruent readings of each
  distinction, the combination that maximizes the structure integrated
  information Φ is selected (computed jointly across distinctions via the
  analytical Σφ_r, since a reading's relation support depends on the other
  distinctions' readings), with residual Φ-ties closed deterministically.
  Beyond 4096 tied combinations a greedy per-distinction pass approximates the
  joint maximum with a warning. Previously the selection fell to enumeration
  order, which understated Φ (basic: 1.0 → 1.125) and made Φ depend on node
  labels; Φ is now invariant under relabeling.

  Two published 2023-paper figure reproductions change under the exact rule:
  Fig 6D's Φ becomes 12395 (published: 11452) and Fig 7B's relation count and Φ
  become 13498 and 19.32 (published: 13111 and 18.55) — φ_s and the distinction
  counts still match the figures exactly. The published values embed the old
  enumeration-order tie resolution, which is relabeling-dependent and
  sub-maximal under the S1 supplement's own rule.
- `fig5b_substrate` now implements gate B as AND(A, C), matching Figure 5B of the 2014 IIT 3.0 paper and the fixture's own diagram (it was transcribed as OR). The example's distinction count is unchanged. `differentiation_macro_tpm` no longer divides the p² term by 3 — the coarse-grained probability is now p² + 2pε/3, which reduces to p² at ε = 0 as the grouping requires.
- Complexes found with the certified intrinsic-information prune (the default
  under the 2026 formalism) could not be saved: gated excluded candidates carry
  `phi=None`, which crashed the serializer. Gated candidates now serialize, and
  the certification record (`ii_ceiling`, `gated`) survives the round-trip.
- `Substrate` fingerprints now include each TPM factor's shape, so substrates with identical flat factor values but different dependence structure no longer collide in the content-addressed repertoire cache.
- Fixed two defects in the `INTRINSIC_INFORMATION` composite measure (reachable
  via `mechanism_phi_measure` / `specification_measure`; the default pipeline
  is unaffected). The cause-side intrinsic differentiation was computed from
  the unnormalized forward likelihoods instead of the Bayes posterior of
  Eq. 11, overstating ii by −log₂ of the normalizer wherever the
  differentiation term binds. And the differentiation operand was squeezed
  while the specification operand kept the repertoire's canonical rank, so the
  elementwise minimum broadcast across singleton axes — producing wrong ii
  values, wrong-length specified states, and an IndexError on the
  config-routed distinction path.
- A `MacroSystem` no longer compares equal to a plain `System` over its macro
  substrate. The two are different analyses (the macro construction overrides
  the cause TPM and yields a different φ), but the fallback comparison saw only
  the shared fields, breaking the equality/hash contract and making set and
  dict membership inconsistent.
- Fixed a save/load defect where a MICE that carries a purview-tie tuple it is not itself a member of (as happens for state- and partition-tied MICE, which share the winner's tie tuple) gained a spurious duplicate member on load, changing its `num_purview_ties` from 0 to 1. The round trip now preserves the tie tuple faithfully.
- Fixed `NullRelations` equality: instances compared by identity, so two identical IIT 3.0 cause-effect structures compared unequal, and a saved and reloaded structure never compared equal to the original. All `NullRelations` instances now compare equal and hash consistently.
- `numpy_aware_eq()` now compares arrays of different shapes as unequal, as documented; previously broadcastable shape pairs (e.g. `(1, n)` vs `(n,)`) compared equal.
- Fixed structural equality on mappings: `numpy_aware_eq` compared dicts by
  zipping their keys positionally, so two dicts with identical keys but
  different values compared equal (and equal dicts with different insertion
  order compared unequal). Mappings now compare by key set with values compared
  recursively.
- Fixed `RelationFace` pickling: pickling, `copy.copy`, and `copy.deepcopy` raised `ValueError: phi keyword argument is required`. Because a `Relation` caches its faces once they are computed (e.g. by `repr()` or `Relations.num_faces()`), this also made any relation — or an entire cause-effect structure containing one — unpicklable afterward, breaking process-based parallelism and saving results with `pickle`.
- `Relation` and `RelationFace` now order by φ: `max()`, `min()`, `sorted()`, and comparisons follow φ instead of accidentally using `frozenset` subset comparison. Equality and hashing keep set semantics.
- Distinction-level normalized φ was not stored on serialization: it was
  recomputed from the ambient `distinction_phi_normalization` option at load
  time, so a result computed under one formalism and loaded under another
  silently changed value (e.g. an IIT 3.0 result reloaded under the 2026
  default: 0.5 → 0.1667). The signed normalized φ is now stored in the schema
  and restored on load; files written before the field existed keep the old
  recompute fallback.
- Fixed a race in the serializer's first-use type registration: a thread that started serializing while another thread was still registering the serializable types could fail with `TypeError: No serializer registered for ...`. Registration is now atomic to concurrent observers.
- Sharded campaign merges are now exact. Campaign strides report every
  specified-state candidate's local minimum — one entry per pin at the
  distinction level, one per (cause, effect) pair at the SIA level — so the
  merge takes the cross-stride minimum per candidate before running the same
  selection the unsharded search runs. Previously each stride reported only its
  local winners, so a sharded campaign could report a reducible distinction as
  real (φ = 0.2075 where the full sweep gives φ = 0), select a different system
  MIP, or resolve congruence against a different specified system state. Under
  IIT 4.0 (2026) the intrinsic-information requirement is now applied at merge time,
  after the global MIP per pair is chosen, matching the unsharded definition.
- `System.sia(system_state=...)` shared the plain `sia()` disk-cache entry, so
  with `disk_cache_results` enabled a caller-supplied (possibly non-canonical)
  state specification could poison — and be served by — the cached canonical
  result, persisting across processes. Forced-state calls now bypass the disk
  result cache entirely.
- `pyphi.dynamics.simulate()` draws its random initial state from the TPM's own state labels for state-by-state TPMs, so non-binary units can start in any state of their alphabet; previously the draw was hardcoded binary.
- `sweep(formalisms=None)` now computes under the active configuration exactly as `pyphi.analyze()` does. Previously it silently replaced the ambient config with the complete version preset, discarding runtime customizations for the duration of the sweep.
- `TransitionSystem` equality and hashing now include `noise_background`, so frozen- and noised-background views of the same transition no longer compare equal.
- `TransitionSystem.save()` previously delegated to the underlying `System`, so
  loading the file silently returned a `System` and lost the transition data
  (before/after states, cause/effect sets, direction). `TransitionSystem` now
  has its own serialization schema and round-trips faithfully.
- Fixed the build configuration so wheels and sdists actually contain the package source: the hatchling `include` allowlists shipped only `pyphi/data/`, producing artifacts with zero `.py` files that still imported as an empty namespace package.
- `AnalyticalFoldRelations.save()` now raises a clear error pointing at the
  parent structure, matching the other view types, instead of an opaque
  serializer TypeError.
- `FrozenMap` now hashes its key–value pairs together rather than hashing the key
  set and the value set separately. The previous hash satisfied the equality
  contract but did not distinguish mappings that differ only in which key holds
  which value, so the 2ⁿ mechanism-state conditions the repertoire cache keys on
  collapsed onto three hashes: every cache operation degenerated to a linear scan
  of the bucket under `Mapping.__eq__`, making the cache quadratic in its own
  size. No computed value changes; the specified-state search over a 10-unit
  system drops from 55 seconds to 0.2, and one over 16 units from days to
  roughly a minute.
- Fixed `JointTPM` views returned by `condition()`: `alphabet_sizes` now reports the true per-unit output alphabets instead of the collapsed input-axis sizes, so `to_pandas()` and the display grid list every next state (probabilities sum to 1 per input state) rather than truncating non-binary units.
- `JointTPM.to_array()` and `numpy.asarray(joint_tpm)` now return a read-only array, so the documented read-only value type can no longer be mutated through its own buffer (which silently changed its hash and equality).
- Fixed `pyphi.parallel.map_reduce` draining generator inputs up front on the process backend: unknown-length workloads are now consumed lazily up to `sequential_threshold`, so a short-circuit predicate that fires early no longer pulls the entire generator before the sequential/parallel decision.
- Capped the `mcp` extra below version 2.0. The 2.0 release removes `mcp.server.fastmcp`, which `pyphi-mcp` is built on, so `pip install "pyphi[mcp]"` resolved to a server that failed at import.
- Fixed nondeterministic mechanism MIP reporting under `shortcircuit_sia=False` with parallel partition evaluation: results were collected in completion order, so tie resolution could select different (φ-equivalent) partitions across identical runs. Partition sweeps now always collect in enumeration order.
- A partial per-level parallelization dict (e.g. `parallel_relation_evaluation={"parallel": True}`) now merges over that level's defaults instead of replacing them wholesale, so omitted keys keep their tuned values (the relation level's `sequential_threshold` of 8192 previously collapsed to 1). Unknown keys are rejected with an error naming the valid keys.
- Null causes and effects produced by distinction short-circuiting now carry the system's node labels and mechanism state, so `repr()`, `.mechanism_label`, and `.to_pandas()` work on reducible distinctions just as they do on fully evaluated ones.
- Null IIT 4.0 system irreducibility analyses no longer fabricate an
  `intrinsic_differentiation` of zero. A null SIA carrying a real
  `system_state` previously reported `intrinsic_information == 0.0` as if it
  had been computed, even when the true ii(s) is nonzero; both fields now
  report `None` when the intrinsic differentiation was not computed.
- The `parallel_kwargs` allowlist now exactly mirrors `map_reduce`'s keyword surface: `inflight_limit` (advertised but rejected by `map_reduce`) is removed, and `size_func`/`backend` (accepted by `map_reduce` but silently dropped by the filter) are allowed through.
- `map_reduce()` now passes `shortcircuit_callback` the same payload on every backend and dispatch path: when `shortcircuit_callback_args` is not given, the callback receives the list of results collected so far, ending with the triggering result. Previously the payload was a partially consumed iterator on the sequential path and a list of executor futures on the process, thread, and dask parallel paths.
- `Part` ordering no longer compares `node_labels`, matching its equality and hash semantics. Previously, comparing equal `Part`s that differed only in labels raised `TypeError`.
- Registering a partition scheme now validates the callable's signature against the registry's call shape (`(mechanism, purview)` for mechanism schemes, `(nodes,)` for system schemes), so wrong-shape registrations fail at registration instead of at the bottom of a phi computation.
- Result cards no longer label φₛ as Φ under IIT 4.0. A `CauseEffectStructure`,
  `Analysis`, or `Complex` card now shows `φ_s` for the system irreducibility
  value, and the Φ-structure cards additionally show `Φ` — the structure
  integrated information, the sum of `Σφ_d` and `Σφ_r` printed beneath it. Under
  IIT 3.0, whose system-level value is that formalism's Φ, the label is unchanged.
  Added `Analysis.big_phi` (Φ, raising under IIT 3.0) and `big_phi` / `sum_phi_d`
  columns to `Analysis.to_pandas()`. The MCP server's result summary drops its
  ambiguous `phi` key — a duplicate of `system_phi` — adds a `formalism` key, and
  renders the MIP concisely instead of as a full card.
  `SystemIrreducibilityAnalysis.explain()`/`.diff()` and the `PhiFold` summary
  use the same labels.
- `pointwise_mutual_information_vector()` now returns 0 where the log-ratio is undefined (`p = 0` or `q = 0`), as documented, instead of substituting the maximum finite float for infinite ratios.
- Fixed `potential_purviews()` (and `System.mic()`/`mie()` with an explicit `purviews=` argument) silently returning no purviews when given a one-shot iterable such as a generator: the candidates are now materialized before being scanned.
- `System.proper_cause_marginal` and `System.proper_effect_marginal` now derive
  the background from `external_indices` rather than from the complement of
  `node_indices`. When the two differ — as in actual-causation
  `TransitionSystem`s, where the external set may overlap the system — the
  proper marginals previously conditioned on the wrong units, and with
  `external_indices=()` (`Transition(..., noise_background=True)`)
  `proper_cause_marginal` crashed. External units are conditioned at the
  background reference state; substrate units neither in the system nor
  external are marginalized uniformly. For plain `System`s, where the external
  set is the complement of the system, results are unchanged.
- The provenance writers no longer crash when `params` contains
  `"seed": None`.
- IIT 3.0 computations now reject an incompatible `mechanism_partition_scheme` (e.g. the IIT 4.0 `JOINT_PARTITION_ALL` family) at the dispatch boundary when the configuration was assembled by per-field assignment, instead of silently computing phi over a different partition family.
- `SweepResult` and `OptimizationResult` equality no longer raises: the generated dataclass `__eq__` compared DataFrame/ndarray fields elementwise, whose truth value is ambiguous. Both now define value equality (`DataFrame.equals`, `np.array_equal`).
- `RepertoireIrreducibilityAnalysis.ties` now preserves co-optimal tied MIPs whose partitions differ. Deduplication keyed on RIA equality alone, which deliberately ignores the partition, so partition-distinct tied MIPs collapsed to one and `diff()` reported a MIP change instead of a tie.
- Fixed TPM row-stochasticity validation admitting relative slack: `numpy.allclose`'s default relative tolerance let row sums off by up to ~1.1e-5 pass while the error message claimed the configured absolute tolerance (1e-13 by default). Row sums are now checked against the absolute tolerance only.
- Fixed `pyphi.timescale.run_tpm` silently discarding conditional dependence for `time_scale >= 2`: the round trip through state-by-node form dropped dependencies introduced by iteration, returning a TPM with the wrong multi-step dynamics. It now raises `ConditionallyDependentError` when the exact iterated dynamics cannot be expressed in state-by-node form, and returns the exact iterated TPM when they can.
- The runner-up partition reported on a `SystemIrreducibilityAnalysis` is now ranked by the same quantity that selects the MIP (the primary φ-valued component of `sia_tie_resolution`, normalized φ by default) instead of always by raw φ, so the reported runner-up is the actual nearest competitor. When ranked by normalized φ, `RunnerUp.normalized_phi` carries the value and the gap finding reports the normalized gap.
- Node labels now survive serialization everywhere they are displayed. Mechanism
  and system partitions (`Part`, `JointPartition` and its variants, `NullCut`,
  `DirectedBipartition`, `DirectedJointPartition`), state specifications, and a
  substrate's factored TPM previously lost their labels on round-trip, so a
  reloaded result rendered its MIP and purviews with bare indices while a fresh
  one showed labels. Conversely, an object saved with no labels no longer
  inherits the document's label frame on load, which could attach another
  object's labels to it.
- A reducible `CausalLink` (α = 0, whose analysis carries no purview, partition,
  or probabilities) now serializes. Previously encoding it raised a `TypeError`.
- A `StateSpecification` whose tie family is just itself no longer loses that
  family on round-trip: `ties` was restored as empty instead of the documented
  self-containing tuple whenever there were no tied peers.
- Campaign shards bound their in-memory caches by the memory they are granted.
  A shard evaluates every mechanism it carries against one long-lived
  `System`, whose cached repertoires are released only when that `System` is
  collected, so without a ceiling a shard packing many mechanisms accumulated
  cache entries. The `memory_ceiling_bytes` option gives an absolute ceiling,
  set automatically during shard execution from the smallest cgroup limit along
  the hierarchy, else the memory the generated submit file requested (the
  `PYPHI_SHARD_MEMORY` environment variable it exports), else the request
  recorded at planning time; `shard_memory_bytes` includes the cache allowance
  it grants, so the request and the enforced ceiling come from the same figure.
- `sia.ties` now contains only the specified-state readings whose φ_s is tied with the winner (up to the configured numerical precision). Previously every evaluated reading was attached as a tie, including readings whose φ_s lost the state-resolution cascade outright.
- Fixed a crash (`ValueError: cascade requires at least one candidate`) in `sia(directions=[...])` when the requested direction's specified state was tied. Single-direction analyses now resolve specified-state ties within that direction alone.
- Fixed a `ZeroDivisionError` when analyzing a single-unit system under the `DIRECTED_BIPARTITION_CUT_ONE` system partition scheme. A single unit has no bipartition with two nonempty parts, so the scheme now yields no partitions and the analysis reports φ_s = 0, matching the other schemes.
- `from pyphi import *` no longer rebinds the stdlib names `warnings` and
  `types` to PyPhi submodules, and `estimate_analysis` / `AnalysisEstimate`
  are now included in `pyphi.__all__`.
- Fixed the thread scheduler ignoring the chunking policy: it submitted one future per item (10,000 submissions for `chunksize=4096`) and never consulted `size_func`. It now submits chunked futures honoring `chunksize` and `size_func`, like the process backend, with unchanged result-order semantics.
- Fixed a thread-backend `map_reduce` permanently disabling config-snapshot installs in its process: the parent-PID latch that suppresses in-thread snapshot application was never reset, so a nested thread dispatch inside a process-pool worker made that worker silently ignore every later configuration snapshot. The latch is now restored when the thread dispatch completes.
- `timescale.run_cm` no longer mutates the caller's connectivity matrix (and
  accepts read-only input), and the `sparse` heuristic is fixed: it was
  inverted (dense matrices took the scipy-sparse branch) and measured on the
  state-by-node TPM rather than the state-by-state matrix actually raised to
  a power. Results are unchanged; only which backend computes them.
- The full-state repertoire sweeps — the unconstrained forward effect repertoire
  and the forward cause repertoire — no longer store their per-state
  intermediates in the kernel cache. Each intermediate is a full repertoire read
  exactly once, so caching them cost the product of the state count and the
  repertoire size (order 4ⁿ cells for an n-unit system) for no hits at all: a
  16-unit specified-state search exhausted memory before finishing. The sweeps
  now run under the new `pyphi.core.repertoire_algebra.transient_repertoires`
  scope, which returns computed repertoires without admitting them. Peak memory
  for a 14-unit search falls from 1.2 GiB to 0.14 GiB, and a 16-unit search
  completes in 0.19 GiB where it previously ran out of memory.

  The size bound on full-state sweeps now covers the cause direction as well as
  the effect one. Only the unconstrained forward effect repertoire checked it,
  and `Direction.both()` walks the cause direction first, so an oversized system
  spent its entire cause sweep before anything refused.
- IIT 3.0 configurations now require `background_conditioning="CONDITION_CURRENT_STATE"`, the convention of the shipped preset and the post-2014 literature. Pairing IIT 3.0 with `CAUSAL_MARGINALIZATION` previously passed validation and silently changed phi on proper-subset systems. A `System` that pins its own convention is checked against the pinned value.
- Configuration validation now covers `formalism.iit.specification_measure`: a measure the active formalism does not accept (e.g. `EMD` under IIT 4.0) is rejected when the configuration is applied, and again at the dispatch boundary for configurations assembled by per-field assignment. Previously such a value was accepted and silently changed Φ and φₛ.
- The shipped `pyphi_config_3.0.yml` reference config failed to load (it used a
  retired field name) and had drifted from the `IIT_3_0` preset. It is now an
  exact mirror of the preset.
- Configuring `version: IIT_3_0` without also setting `sia_tie_resolution`
  left the IIT 4.0 default (`NORMALIZED_PHI`, ...) in place, and `analyze()`
  crashed with an `AttributeError` deep inside tie resolution (IIT 3.0 SIA
  results have no normalized φ). Incompatible SIA tie strategies are now
  rejected with a `ConfigurationError` naming the field and a fix — eagerly on
  `config.override()` / `load_yaml()`, and at the analysis dispatch boundary
  for configs assembled by per-field assignment.
- `IIT3SystemIrreducibilityAnalysis` no longer includes `distinctions` in its hash. The compute path assigns `distinctions` to tie peers after construction, which changed the hash mid-lifetime.
- `formalism.iit.version="IIT_4_0_2026"` now requires a `system_phi_measure` that applies the intrinsic-information requirement (Eq. 23), i.e. `INTRINSIC_INFORMATION`. Previously pairing version 2026 with `GENERALIZED_INTRINSIC_DIFFERENCE` passed validation and computed the 2023 quantity while results reported version 2026.
- `Substrate` now rejects a non-square 2-D `tpm=` array whose row count is not `2**n` for its `n` columns (for example 8 rows with 2 columns), naming the expected row count. Such arrays were previously reshaped into a scrambled substrate over the wrong number of nodes.
- Passing a 2-D transition probability matrix with a multi-valued `alphabet=` or
  `state_space=`, or a state-by-state matrix whose state count is not a power of
  two, now raises an error that says the 2-D forms describe binary units and
  points at the factored form, instead of failing inside a conversion with
  "expected integer".

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

- `ConfigSnapshot.as_kwargs()` documentation now states that the flat form cannot reproduce a snapshot whose formalism differs from the ambient default (colliding fields like `version` are excluded); use `as_overrides()` to reproduce a snapshot.
- Documentation accuracy sweep with an executed build: theory pages, the worked
  example, and the getting-started guide no longer narrate 2023-formalism φₛ
  values over cells executing under the 2026 default (the worked example now
  pins the paper's formalism and reproduces its published numbers); `analyze()`
  is no longer described as searching for the complex; the migration guides'
  IIT 3.0 recipes use complete presets; the README example uses the IIT 4.0
  paper's system and labels φₛ and Φ correctly; and the MCP reference content's
  complexity formulas, tie-resolution description, state-ordering example, and
  configuration recipes are corrected.
- A `reproducible-work` reference topic covering seeding, the `pyphi.provenance`
  writers, the no-clobber filename convention, and saving per-trial values
  alongside summaries. Available through `get_iit_reference` and as the
  `pyphi://theory/reproducible-work` resource.
- Public docstrings, the documentation, and the bundled MCP reference now
  consistently call Mayner et al. (2026) Eq. 23 the intrinsic-information
  requirement ("with" / "without the requirement") rather than a cap.

- A configuration reference page lists every option with its layer and default,
  generated from the configuration classes, followed by their documentation.

- A reference page lists every registered example network with its size and
  source, generated from the registry at build time.
- A FAQ and troubleshooting page: zero φₛ, numbers that differ from a paper or
  from 1.x, unreachable states, conditional dependence, capped estimates, long
  runs, ties, and configuration drift.
- A glossary of the terms the documentation and the IIT 4.0 papers use, each
  entry pointing to the page that treats it.
- New how-to, "Build a substrate": from a transition probability matrix, from a
  weight matrix with logistic units, from a function per unit, and from recorded
  transitions, with the checks to run before analyzing.
- New how-to, "Estimate the cost before you run": the practical ceilings,
  `estimate_analysis` with its counting budget, and what to reduce.

- New how-to, "Read a result": every row of the analysis and system cards, the
  letter-case convention of purview labels, and the two different reasons a
  system's φₛ can be zero.
- The MCP `estimate_cost` tool and the `performance` reference say which axes
  `estimated_cpu_seconds` covers: the distinction axis only, so a system-φ
  estimate reports counts without a time and a full estimate is a lower bound.

- The migration guide gains the example-network renames, a table from every 1.x
  configuration option to its 2.0 location, and a table of the quantities (1.x
  `compute.phi` is `analysis.phi` under IIT 3.0).
- The IIT 4.0 demo notebook is now rendered in full on its tutorial page from
  stored outputs.
- The documentation site has a new visual design: a palette anchored on the
  result cards' cause and effect colours, IBM Plex type, Selenized code blocks
  in both themes, restyled cards, tables, and admonitions, a landing page that
  leads with the section cards, an announcement bar, a version switcher, and a
  sidebar citation.
- Every configuration option carries a docstring stating what it controls, its
  accepted values, and which preset changes it; the configuration reference
  lists them by layer with their defaults and renders the full descriptions.
- Reference has an executed gallery of the example networks, rendered from the
  objects themselves, and the hand-copied TPM and connectivity-matrix tables
  were removed from the `pyphi.examples` docstrings.
- The conditional-independence theory page now opens with the reasoning behind
  the assumption: a substrate is a complete causal model whose transitions are
  defined by intervention, so units act only across steps and their joint
  transition factors per unit.
- Docs describe an excluded candidate with higher φₛ in plain words instead of a
  coined term, and refer to the default formalism as IIT 4.0 (2026) rather than
  a refinement.
- Docstrings, the bundled MCP reference topics, the documentation pages, the
  changelog, and the demo notebook no longer carry development narrative
  (verification stories, work-item labels, test-suite references, design
  defences); they describe what the code is and does.
- The documentation site now publishes `llms.txt` and `llms-full.txt` (the
  narrative pages as one markdown file) and a sitemap, and the landing page and
  README point AI assistants at the MCP server, the per-page sources, and the
  intersphinx inventory.
- The documentation site is restyled: a teal accent, figures drawn while
  building the docs (and the precomputed complexity figures) on a transparent
  ground in the site palette so they read in both themes, φₛ and the other
  subscripted quantities written as math in prose, page titles of the form "Page
  — PyPhi", Open Graph tags for link previews, an install line and a four-line
  example on the landing page, Ctrl+B (Cmd+B) to toggle the sidebar, and the
  how-to guides and the what's-new page ordered by how soon a reader needs each
  item, the latter with a new part on ergonomics and quality of life.

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
  `parallel_backend="auto"` selects threads on free-threaded runtimes).


1.2.0
-----
_2019-06-21_

### Fixes

- Fixed a bug introduced into `pyphi.utils.load_data()` by a breaking change
  in recent versions of NumPy that caused an error on import.
- Fixed a bug where changing `config.PRECISION` dynamically did not change
  `constants.EPSILON`, causing some comparisons that relied on
  `constants.EPSILON` to not reflect the new precision.
- Changing `config.FS_CACHE_DIRECTORY` and `config.FS_CACHE_VERBOSITY` now
  causes a new `joblib.Memory` cache to be created. Previously, changing these
  options dynamically had no effect.
- Made test suite compatible with stricter usage of `pytest` fixtures
  required by recent versions of `pytest`.

### API additions

- Added `pyphi.tpm.reconstitute_tpm()`.

### API changes

- Renamed `pyphi.partition.partition_registry` to
  `pyphi.partition.partition_types`.
- Renamed `pyphi.distance.bld()` to `pyphi.distance.klm()`.
- Fixed the connectivity matrix of the `disjunction_conjunction_network()`.
- Removed `'expanded_*_reperotire'` attributes of JSON-serialized `Concept`s.

### Config

- Added the `WELCOME_OFF` option to turn off the new welcome message.

### Documentation

- Added documentation for the `partition_types` registry.
- Added documentation for the filesystem and database caches.


1.1.0
-----
_2018-05-30_

### Fixes

- Fixed a memory leaked when concepts returned by parallel CES computations
  were returned with distinct subsystem objects. Now all objects in a CES share
  the same subsystem reference.
- Fixed a race condition caused by newly introduced `tqdm` synchronization.
  Removed the existing `ProgressBar` implementation and pinned `tqdm` to
  version >= 4.20.0.
- Made model hashes deterministic (6b59061). This fixes an issue with the Redis
  MICE cache in which cached values were not shared between processes and
  program invocations.
- Fixed the connectivity matrix in `examples.disjunction_conjunction.network()`.

### API additions

- Added a `NodeLabels` object for managing the labels of network elements. Most
  models now carry a `NodeLabels` instance that is used for string formatting.
- Added the `cut_node_labels` property to `Subsystem` and `MacroSubsystem`.
- Added `utils.time_annotated` decorator to measure execution speed.

### API changes

- Specifying the nodes of a `Subsystem` is now optional. If not provided, the
  subsystem will cover the entire network.
- Removed the `labels2indices`, `indices2labels` and `parse_node_indices`
  methods from `Network`, and the `indices2labels` method from `Subsystem`.
- Renamed `config.load_config_file` to `config.load_file`, and
  `config.load_config_dict` to `config.load_dict`
- Removed backwards-compatible `Direction` import from `constants` module.
- Renamed `macro.coarse_grain` to `coarse_graining`.
- Exposed `coarse_grain`, `blackbox`, `time_scale`, `network_state` and
  `micro_node_indices` as attributes of `MacroSubsystem`.

### Config

- Removed the `LOG_CONFIG_ON_IMPORT` configuration option.


1.0.0 :tada:
------------
_2017-12-21_

### API changes

#### Modules

- Renamed:
  - `compute.big_phi` to `compute.network`
  - `compute.concept` to `compute.subsystem`
  - `models.big_phi` to `models.subsystem`
  - `models.concept` to `models.mechanism`

#### Functions

- Renamed:
  - `compute.main_complex()` to `compute.major_complex()`
  - `compute.big_mip()` to `compute.sia()`
  - `compute.big_phi()` to `compute.phi()`
  - `compute.constellation()` to `compute.ces()`
  - `compute.conceptual_information()` to `compute.conceptual_info()`
  - `subsystem.core_cause()` to `subsystem.mic()`
  - `subsystem.core_effect()` to `subsystem.mie()`
  - `subsystem.mip_past()` to `subsystem.cause_mip()`
  - `subsystem.phi_mip_past()` to `subsystem.phi_cause_mip()`
  - `subsystem.phi_mip_future()` to `subsystem.phi_effect_mip()`
  - `distance.small_phi_measure()` to `distance.repertoire_distance()`
  - `distance.big_phi_measure()` to `distance.system_repertoire_distance()`
  - For all functions in `convert`:
    - `loli` to `le` (little-endian)
    - `holi` to `be` (big-endian)
- Removed `compute.concept()`; use `Subsystem.concept()` instead.

#### Arguments

- Renamed `connectivity_matrix` keyword argument of `Network()` to `cm`

#### Objects

- Renamed `BigMip` to `SystemIrreducibilityAnalysis`
  - Renamed the `unpartitioned_constellation` attribute to `ces`
  - `sia` is used throughout for attributes, variables, and function names
    instead of `big_mip`
- Renamed `Mip` to `RepertoireIrreducibilityAnalysis`
  - Renamed the `unpartitioned_repertoire` attribute to `repertoire`
  - `ria` is used throughout for attributes, variables, and function names
    instead of `mip`
- Renamed `Constellation` to `CauseEffectStructure`
  - `ces` is used throughout for attributes, variables, and function names
    instead of `constellation`
- Renamed `Mice` to `MaximallyIrreducibleCauseOrEffect`
  - `mic` or `mie` are used throughout for attributes, variables, and function
    names instead of `mip`

- Similar changes were made to the `actual` and `models.actual_causation`
modules.

#### Configuration settings

- Changed configuration settings as necessary to use the new object names.

#### Constants

- Renamed `Direction.PAST` to `Direction.CAUSE`
- Renamed `Direction.FUTURE` to `Direction.EFFECT`

### API additions

#### Configuration settings

- Added `CACHE_REPERTOIRES` to control whether cause/effect repertoires are
  cached. Single-node cause/effect repertoires are always cached.
- Added `CLEAR_SUBSYSTEM_CACHES_AFTER_COMPUTING_SIA` to control whether
  subsystem caches are cleared after calling `compute.sia()`.

#### Objects

- Added two new objects, `MaximallyIrreducibleCause` and
  `MaximallyIrreducibleEffect`, that are subclasses of
  `MaximallyIrreducibleCauseOrEffect` with a fixed direction.

### Refactor

- Moved network-level functions in `compute.big_phi` to
  `pyphi.compute.network`
- Moved subsystem-level functions in `compute.big_phi` and `compute.concept` to
  `compute.subsystem`

### Documentation

- Added a description of TPM representations.
- Improved the explanation of conditional independence and updated the example
  to reflect that PyPhi now raises an error if a conditionally-dependent TPM is
  provided.
- Added detailed installation instructions.
- Little-endian and big-endian replace LOLI and HOLI terminology
- Added documentation for the following modules:
  - `distribution`
  - `cache`
  - `compute.parallel`
  - `compute` top-level module
  - `module` top-level module


0.9.1
-----
_2017-12-21_

### Fixes
- Refactored parallel processing support to fix an intermittent deadlock.


0.9.0
-----
_2017-12-04_

### API changes
- Many functions have been refactored to different modules; see the "Refactor"
  section for details.
- `compute.possible_complexes` no longer includes the empty subsystem.
- Made `is_cut` a property.
- Renamed `macro.list_all_partitions` and `macro.list_all_groupings` to
  `all_partitions` and `all_groupings`. Both are now generators and return
  nested tuples instead of lists.
- Moved `macro.make_mapping` to `CoarseGrain.make_mapping`.
- Moved `macro.make_macro_tpm` to `CoarseGrain.macro_tpm`.
- Added blackbox functionality to `macro.emergence`. Blackboxing and coarse-
  graining are now parametrized with the `blackbox` and `coarse_grain`
  arguments.
- Removed `utils.submatrix`.
- Made `Network.tpm` and `Network.cm` immutable properties.
- Removed the `purview` argument from `Subsystem.expand_repertoire`.
- Moved `validate.StateUnreachableError` and `macro.ConditionallyDependentError`
  to the `exceptions` module.
- Removed perturbation vector support.
- Changed `tpm.marginalize_out` to take a list of indices.
- Fixed `macro.effective_info` to use the algorithm from the macro-micro paper.
- Replace `constants.DIRECTIONS`, `constants.PAST`, and `constants.FUTURE` with
  a proper `Enum` class: `constants.Direction`. Past and future are now
  represented by `constants.Direction.PAST` and `constants.Direction.FUTURE`.
- Simplified logging config to use `config.LOG_STDOUT_LEVEL`,
  `config.LOG_FILE_LEVEL` and `config.LOG_FILE`.
- Removed the `location` property of `Concept`.

### API additions
- Added `subsystem.evaluate_partition`. This returns the φ for a particular
  partition.
- Added `config.MEASURE` to choose between EMD, KLD, or L1 for distance
  computations.
- Added `macro.MacroSubsystem`. This subclass of `Subsystem` is used to performs
  macro computations.
- Added `macro.CoarseGrain` to represent coarse-grainings of a system.
- Added `macro.Blackbox` to represent system blackboxes.
- Added `validate.blackbox` and `validate.coarse_grain`.
- Added `macro.all_coarse_grains` and `macro.all_blackboxes` generators.
- Added `Subsystem.cut_indices` property.
- Added `Subsystem.cm` connectivity matrix alias.
- Added `utils.all_states`, a generator over all states of an `n`-element
  system.
- Added `tpm.is_state_by_state` for testing whether a TPM is in state-by-state
  format.
- `Network` now takes an optional `node_labels`  argument, allowing nodes to be
  referenced by a canonical name other than their indices. The nodes of a
  `Subsystem` can now be specified by either their index or their label.
- Added `models.normalize_constellation` for deterministically ordering a
  constellation.
- Added a `Makefile`.
- Added an `exceptions` module.
- Added `distribution.purview` for computing the purview of a repertoire.
- Added `distribution.repertoire_shape`.
- Added `config.PARTITION_TYPE` to control the ways in which φ-partitions are
  generated.
- Added more functions to the `convert` module:
  - `holi2loli` and `loli2holi` convert decimal indices between **HOLI** and
    **LOLI** formats.
  - `holi2loli_state_by_state` and `loli2holi_state_by_state` convert between
    **HOLI** and **LOLI** formats for state-by-state TPMs.
  - Added short aliases for some functions:
    - `h2l` is `holi2loli`
    - `l2h` is `loli2holi`
    - `l2s` is `loli_index2state`
    - `h2s` is `holi_index2state`
    - `s2h` is `state2loli_index`
    - `s2l` is `state2holi_index`
    - `h2l_sbs` is `holi2loli_state_by_state`
    - `l2h_sbs` is `loli2holi_state_by_state`
    - `sbn2sbs` is `state_by_node2state_by_state`
    - `sbs2sbn` is `state_by_state2state_by_node`
- Added the `Constellation.mechanisms`, `Constellation.labeled_mechanisms`, and
  `Constellation.phis` properties.
- Add `BigMip.print` method with optional `constellations` argument that allows
  omitting the constellations.

### Refactor
- Refactored the `utils` module into the `connectivity`, `distance`,
  `distribution`, `partition`, `timescale`, and `tpm` modules.
- Existing macro coarse-grain logic to use `MacroSubsystem` and `CoarseGrain`.
- Improved string representations of PyPhi objects.
- Refactored JSON support. The `jsonify` module now dumps PyPhi models to a
  a format which can be loaded to reproduce the full object graph of PyPhi
  objects. This causes backwards incompatible changes to the JSON format of
  some model representations.
- Refactored `pyphi.config` to be an object. Added validation and callbacks for
  config options.

### Optimizations
- Added an analytic solution for the EMD computation between effect
  repertoires.
- Improved the time complexity of `directed_bipartition_of_one` from
  exponential to linear.

### Documentation
- Updated documentation and examples to reflect changes made to the `macro` API
  and usage.
- Added documentation pages for new modules.


0.8.1
------------------
_2016-02-11_

### Fixes
- Fixed a bug in `setup.py` that prevented installation.


0.8.0
------------------
_2016-02-06_

### API changes
- Mechanisms and purviews are now passed to all functions and methods in node
  index form (e.g. `(0, 1, 3)`). Previously, many functions took these
  arguments as `Node` objects. Since nodes belong to a specific `Subsystem` it
  was possible to pass nodes from one subsystem to another subsystem's methods,
  leading to incorrect results.
- `constellation_distance` no longer takes a `subsystem` argument because
  concepts in a constellation already reference their subsystems.
- Moved `utils.cut_mechanism_indices` and `utils.mechanism_split_by_cut` to
  to `Cut.all_cut_mechanisms` and `Cut.splits_mechanism`, respectively;
  moved `utils.cut_mice` to `Mice.damaged_by_cut`.
- `Concept.__eq__`: when comparing concepts for equality, we no longer directly
  check equality of their subsystems. Concept equality is now defined as
  follows:
    - Same φ
    - Same mechanism node indices cause/effect purview node indices
    - Same mechanism state
    - Same cause/effect repertoires
    - Same networks
  This allows two concepts to be equal when _e.g._ the only difference between
  them is that one's subsystem is a superset of the other's subsystem.
- `Concept.__hash__`: the above notion of concept equality is also implemented
  for concept hashing, so two concepts that differ only in that way will have
  the same hash value.
- Disabled concept caching; removed the `config.CACHE_CONCEPTS` option.

### API Additions
- Added `config.REPR_VERBOSITY` to control whether `__reprs__` of PyPhi models
  use pretty string formatting and control the verbosity of the output.
- Added a `Constellation` object.
- Added `utils.submatrix` and `utils.relevant_connections` functions.
- Added the `macro.effective_info` function.
- Added the `utils.state_of` function.
- Added the `Subsystem.proper_state` attribute. This is the state of the
  subsystem's nodes, rather than the entire network state.
- Added an optional Redis-backed cache for Mice objects. This is enabled with
  `config.REDIS_CACHE` and configured with `config.REDIS_CONFIG`.
- Enabled parallel concept evaluation with `config.PARALLEL_CONCEPT_EVALUATION`.

### Fixes
- `Concept.eq_repertoires` no longer fails when the concept has no cause or
  effect.
- Fixed the `Subsystem.proper_state` attribute.

### Refactor
- Subsystem Mice and cause/effect repertoire caches; Network purview caches.
  Cache logic is now handled by decorators and custom cache objects.
- Block reducibility tests and Mice connection computations.
- Rich object comparisons on phi-objects.

### Documentation
- Updated documentation and examples to reflect node-to-index conversion.


0.7.5 [unreleased]
------------------
_2015-11-02_

### API changes
- Subsystem states are now validated rather than network states. Previously,
  network states were validated, but in some cases there can be a
  globally-impossible network state that is locally possible for a subsystem
  (or vice versa) when considering the subsystem's TPM, which is conditioned
  on the external nodes (i.e., background conditions). It is now impossible to
  create a subsystem in an impossible state (a `StateUnreachableError` is
  thrown), and accordingly no 𝚽 values are calculated for such subsystems; this
  may change results from older versions, since in some cases the calculated
  main complex was in fact in an impossible. This functionality is enabled by
  default but can be disabled via the `VALIDATE_SUBSYSTEM_STATES` option.


0.7.4 [unreleased]
------------------
_2015-10-12_

### Fixes
- Fixed a caching bug where the subsystem's state was not included in its hash
  value, leading to collisions.


0.7.3 [unreleased]
------------------
_2015-09-08_

### API changes
- Heavily refactored the `pyphi.json` module and renamed it to `pyphi.jsonify`.


0.7.2 [unreleased]
------------------
_2015-07-01_

### API additions
- Added `convert.nodes2state` function.
- Added `constrained_nodes` keyword argument to `validate.state_reachable`.

### API changes
- Concept equality is now more permissive. For two concepts to be considered
  equal, they must only have the same φ, the same mechanism and purviews (in
  the same state), and the same repertoires.


0.7.1
------------------
_2015-06-30_

### API additions
- Added `purviews`, `past_purviews`, `future_purviews` keyword arguments to
  various concept-calculating methods. With these, the purviews that are
  considered in the concept calculation can be restricted.

### API changes
- States are now associated with subsystems rather than networks. Functions in
  the `compute` module that operate on networks now also take a state.

### Fixes
- Fixed a bug in `compute._constellation_distance_emd` where partitioned
  concepts were unable to be moved to the null concept for the EMD calculation.
  In some cases, the partitioned system has *greater* ∑φ than the unpartitioned
  system; therefore it must be possible for the φ of partitioned-constellation
  concepts to be moved to the null concept, not just vice versa.
- Fixed a bug in `compute._constellation_distance_emd` where it was possible to
  move concepts around within their own constellation; the distance matrix now
  disallows any such intraconstellation paths. This is important because in
  some cases paths from a concept in one constellation to a concept the other
  can actually be shorter if a detour is taken through a different concept in
  the same constellation.
- Fixed a bug in `validate.state_reachable` where network states were
  incorrectly validated.
- `macro.emergence` now always returns a macro-network, even when 𝚽 = 0.
- Fixed a bug in `repr(Network)` where the perturbation vector and connectivity
  matrix were switched.

### Documentation
- Added example describing “magic cuts” that, counterintuitively, can create
  more concepts.
- Updated existing documentation to the new subsystem-state association.


0.7.0
------------------
_2015-05-08_

### API additions
- `pyphi.macro` provides several functions to analyze networks over different
  spatial scales.
- `convert.conditionally_independent(tpm)` checks if a TPM is conditionally
  independent.

### API changes
- Φ and φ values are now rounded to `config.PRECISION` when stored on objects.

### Fixes
- Tests for `Subsystem_find_mip_parallel` and `Subsystem_find_mip_sequential`.
- Slow tests for `compute.big_mip`.

### Refactor
- Subsystem cause and effect repertoire caching.

### Documentation
- Added XOR and Macro examples.


0.6.0
------------------
_2015-04-20_

### Optimizations
- Pre-compute and cache possible purviews.
- Compute concept distance over least-common-purview rather than whole system.
- Store `relevant_connections` on MICE objects for MICE cache checking.
- Only recheck concepts and cut mechanisms after a system cut.

### API additions
- The new configuration option `CUT_ONE_APPROXIMATION` gives an approximation
  of Φ by only considering cuts that cut off a single node.
- Formerly, the configuration was always printed when PyPhi was imported. Now
  this can be suppressed by setting the `LOG_CONFIG_ON_IMPORT` option to
  `false` in the `pyphi_config.yml` file.

### Fixes
- Bipartition function.
- MICE caching.


0.5.0
------------------
_2015-03-02_

### Optimizations
- Concepts are only recomputed if they could have been changed by a cut.
- Cuts are evaluated individually, rather than in bidirectional pairs, which
  allows for better parallel performance.

### API changes
- Removed the unused `validate.nodelist` function.

### API additions
- The new configuration option `ASSUME_CUTS_CANNOT_CREATE_NEW_CONCEPTS` gives
  an approximation of Φ by only recomputing concepts that exist in the
  unpartitioned constellation. This is much faster in certain cases.
- The methods used in determining whether a cut could effect a concept are
  exposed via the `utils` module as:
    - `utils.cut_mechanism_indices`
    - `utils.cut_concepts`
    - `utils.uncut_concepts`
- Added the `pyphi.Subsystem.connections_relevant_for_concept` method.
- Added the `pyphi.Subsystem.cut_matrix` property.

### Fixes
- `pyphi.compute.main_complex` now returns an empty `BigMip` if there are no
  proper complexes.
- No longer using LRU-caches implemented as a circular, doubly-linked list;
  this was causing a huge number of recursive calls when pickling a `Subsystem`
  object (since caches are stored on subsystems since v0.3.6) as `pickle`
  traversed the (potentially very large) cache.
- `pyphi.json.make_encodable` now properly handles NumPy numeric types.


0.4.0
-----
_2015-02-23_

### Optimizations
- `compute.big_mip` is faster for reducible networks when executed in parallel;
  it returns immediately upon finding a reducible cut, rather than evaluating
  all cuts. **NOTE:** This introduces a race condition in cases where there is
  more than one reducible cut; there is no guarantee as to which cut will be
  found first and returned.
- `compute.complexes` prunes out subsystems that contain nodes without either
  inputs or outputs (any subsystem containing such a node must necessarily have
  zero Φ).

### API changes
- `compute.complexes`: returns only irreducible MIPs; see optimizations.
- `compute.big_mip`
  - New race condition with cuts; see optimizations.
  - The single-node and null `BigMip`'s constellations are now empty tuples
    instead of empty lists and `None`, respectively.
- `models.Concept.eq_repertoires` no longer ensures that the networks of each
  concept are equal; it only checks if the repertoires are the same.

### API additions
- `compute.all_complexes` returns the `BigMip` of every subsystem in the
  network's powerset (including reducible ones).
- `compute.possible_main_complexes` returns the subsystems that survived the
  pruning described above.

### Fixes
- Network tests.

### Refactor
- Network attributes. They're now implemented as properties (with getters and
  setters) to facilitate changing them properly. It should be possible to use
  the same network object with different states.
- Network state validation.
- `utils.phi_eq` is used wherever possible instead of direct comparisons to
  `constants.EPSILON`.
