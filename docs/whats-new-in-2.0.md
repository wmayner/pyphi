# What's new in PyPhi 2.0

Highlights of the 2.0 release. Each section is a short tour of one
capability; the complete list of changes is in the changelog.

<!--
Candidate sections to add as they land / get written up:
- Precision architecture: tolerance at decision sites, tie reporting (.ties, margins)
- Formalism objects: IIT 3.0 / IIT 4.0 (2023, 2026) / Actual Causation as first-class, switchable strategies
- k-ary (multi-valued) units end to end
- pyphi.analyze(): one entry point for substrate → complexes → structure
- Macro/grain search: intrinsic units, blackboxing, temporal grains, cost pre-flight
- Rich display cards for every result type
- New serialization (msgspec): compact, typed, binary-capable
- Repertoire-algebra kernel rewrite: ~18-20x faster SIA partition evaluation
- Substrate landscape tools (pyphi.landscape)
-->

## Query the relational structure without enumerating it

Relations dominate a cause-effect structure at any interesting scale. The
IIT 4.0 paper's specialized-lattice example (Fig. 6D) has 27 distinctions
and 1,537,080 relations; enumerating them takes about a minute and more
than a gigabyte of memory, and past roughly 35 distinctions enumeration
stops fitting in memory at all. But the relation set is fully determined
by the distinctions, so most questions about it can be answered exactly
without ever building it.

PyPhi 2.0 turns that fact into an interface. With
`config.formalism.iit.relation_computation = "ANALYTICAL"`, a relation
set answers structural questions in closed form, in milliseconds, at any
scale:

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
```

Three more tools cover what closed forms can't:

- `relations.strongest(k)` yields relations one at a time in exact
  descending φ_r order, so the top ten of a million-relation structure
  cost about a millisecond — useful for plots and reports that only ever
  show the strongest few.
- `relations.sample(n, seed=...)` draws an unbiased sample of relations
  and returns estimates with standard errors for any per-relation
  quantity you define — including questions no closed form reaches.
- `relations.materialize(max_degree=..., min_phi=...)` is the explicit
  escape hatch back to concrete `Relation` objects, with bounds so you
  can't accidentally ask for all 2^27 of them.

The same queries work on Φ-folds (restricted to the relations touching a
set of distinctions), and
`ces.distinction_importance()` ranks each distinction by its additive
contribution to Φ — the contributions sum to Φ exactly, so the ranking is
a true decomposition of the structure's integrated information.
