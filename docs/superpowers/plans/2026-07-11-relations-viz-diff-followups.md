# Analytical-safe relations viz and diff — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the relation-visualization projection and the cause-effect-structure diff work correctly when relations are the non-enumerable `AnalyticalRelations` backend, without losing any detail on the concrete backend.

**Architecture:** Statistics that both backends answer in closed form become the always-present relation summary (node sizing, the spectrum view, and the diff's relation deltas). Per-relation detail — the rendered edge/face set and the diff's gained/lost rows — is drawn from the top-k strongest relations for viz and kept only where relations are enumerable for the diff.

**Tech Stack:** Python 3.13+, pytest, plotly (optional, guarded by `importorskip`), the existing `pyphi.relations` / `pyphi.visualize` / `pyphi.models` modules.

## Global Constraints

- Python 3.13+ only; no backward-compatibility shims.
- `uv run` for all Python commands (e.g. `uv run pytest ...`).
- NumPy-style docstrings, final-state impersonal voice, Unicode symbols (`Σφ_r`, `φ`); no process narrative, no planning-artifact references (no "Task N", "Wave 7", "per spec") in source, comments, docstrings, or the changelog.
- Numeric comparisons of φ use `pyphi.numerics.eq` (tolerant at the configured precision), never `==` on floats.
- Never overwrite existing output files; not applicable here (no experiment outputs).
- Commit after each task. Do not push. Do not use `--no-verify`.

---

### Task 1: `sum_phi_by_distinction` on the relation backends

Each distinction's incident Σφ_r (the sum of φ_r over every relation containing it, self-relation included), computed by iteration on the base/concrete backend and in closed form on the analytical backend. Node sizing in Task 2 consumes it.

**Files:**
- Modify: `pyphi/relations.py` — add `sum_phi_by_distinction` to the `Relations` base class (after `degree_spectrum`, ~line 466) and override it on `AnalyticalRelations` (after its `degree_spectrum`, ~line 891).
- Test: `test/test_relations_queries.py`

**Interfaces:**
- Produces: `Relations.sum_phi_by_distinction(distinctions) -> tuple[float, ...]`, where `distinctions` is an iterable of `Distinction` objects; the result is parallel to it. Defined on `Relations` (iterating), `ConcreteRelations` and `NullRelations` (inherited), and `AnalyticalRelations` (closed-form override).

- [ ] **Step 1: Write the failing test**

Add to `test/test_relations_queries.py` (the `structures` fixture yields `(name, distinctions, concrete, analytical)`):

```python
def test_sum_phi_by_distinction_parity(structures):
    """Per-distinction incident Σφ_r agrees between the iterating and
    closed-form backends, and the iterating sum conserves Σ_r φ_r·degree(r)."""
    _, distinctions, concrete, analytical = structures
    dl = list(distinctions)
    conc = concrete.sum_phi_by_distinction(dl)
    anal = analytical.sum_phi_by_distinction(dl)
    assert len(conc) == len(dl)
    for c, a in zip(conc, anal, strict=True):
        assert c == pytest.approx(a)
    # Independent oracle: each relation contributes φ_r to each of its relata.
    assert sum(conc) == pytest.approx(sum(float(r.phi) * len(r) for r in concrete))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_relations_queries.py::test_sum_phi_by_distinction_parity -x`
Expected: FAIL with `AttributeError: 'ConcreteRelations' object has no attribute 'sum_phi_by_distinction'`.

- [ ] **Step 3: Add the base (iterating) implementation**

In `pyphi/relations.py`, inside `class Relations`, immediately after the `degree_spectrum` method:

```python
    def sum_phi_by_distinction(self, distinctions) -> tuple[float, ...]:
        """Return each distinction's incident Σφ_r, aligned to ``distinctions``.

        A distinction's incident Σφ_r is the sum of φ_r over every relation
        that contains it, including its self-relation. The result is a tuple
        parallel to ``distinctions``; a distinction that no relation reaches
        contributes ``0.0``.
        """
        position = {tuple(d.mechanism): i for i, d in enumerate(distinctions)}
        sums = [0.0] * len(position)
        for relation in self:  # type: ignore[attr-defined]  # iterable in subclasses
            phi = float(relation.phi)
            for mechanism in relation.mechanisms:
                index = position.get(tuple(mechanism))
                if index is not None:
                    sums[index] += phi
        return tuple(sums)
```

- [ ] **Step 4: Add the analytical (closed-form) override**

In `pyphi/relations.py`, inside `class AnalyticalRelations`, immediately after its `degree_spectrum` method:

```python
    def sum_phi_by_distinction(self, distinctions) -> tuple[float, ...]:
        """Return each distinction's incident Σφ_r in closed form.

        A relation either contains a given distinction or does not, so its
        incident Σφ_r is ``total − Σφ_r(relations avoiding it)``: the full
        total differenced against the total over the remaining distinctions.
        No relations are enumerated. The result is parallel to
        ``distinctions``.
        """
        from pyphi.models.distinctions import ResolvedDistinctions

        total = self.sum_phi()
        result = []
        for distinction in distinctions:
            mechanism = tuple(distinction.mechanism)
            others = ResolvedDistinctions(
                d for d in self.distinctions if tuple(d.mechanism) != mechanism
            )
            result.append(total - AnalyticalRelations(others).sum_phi())
        return tuple(result)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest test/test_relations_queries.py::test_sum_phi_by_distinction_parity -v`
Expected: PASS for every parametrized network (`xor`, `basic`, `rule110`, `fig4`, `grid3`), including `basic` (zero relations → all zeros on both backends).

- [ ] **Step 6: Commit**

```bash
git add pyphi/relations.py test/test_relations_queries.py
git commit -m "Add sum_phi_by_distinction to relation backends

Per-distinction incident Σφ_r, by iteration on the concrete backend and
closed form (total minus seed-free total) on the analytical backend."
```

---

### Task 2: Project through `strongest(k)` with faithful sizing and a closed-form spectrum

`project_ces` renders edges and faces from the top-k strongest relations, sizes nodes from the faithful `sum_phi_by_distinction`, and carries a closed-form `degree_spectrum`. Uncapped projection of a non-enumerable backend raises.

**Files:**
- Modify: `pyphi/visualize/projection/__init__.py` — add `field` import; add `degree_spectrum` to `CESProjection`; rewrite `project_ces`; rewrite `_faces`; delete `_sum_phi_relations`.
- Test: `test/visualize/test_visualize_projection.py`

**Interfaces:**
- Consumes: `Relations.sum_phi_by_distinction` (Task 1); `Relations.strongest(k=...)` and `Relations.degree_spectrum()` (existing).
- Produces: `project_ces(ces, node_labels=None, max_relations=None) -> CESProjection`; `CESProjection.degree_spectrum: dict[int, tuple[int, float]]`. Raises `ValueError` when `max_relations is None` and `ces.relations` is not iterable.

- [ ] **Step 1: Write the failing tests**

In `test/visualize/test_visualize_projection.py`, delete the import `from pyphi.visualize.projection import _sum_phi_relations` and the test `test_sum_phi_relations_exact` (both obsolete — the summing logic now lives in `Relations.sum_phi_by_distinction`, tested in Task 1). Add:

```python
def _xor_ces(analytical):
    import pyphi
    from pyphi import examples

    if not analytical:
        return examples.xor_system().ces()
    with pyphi.config.override(relation_computation="ANALYTICAL"):
        return examples.xor_system().ces()


def test_project_ces_carries_degree_spectrum():
    from pyphi.visualize.projection import project_ces

    ces = _xor_ces(analytical=False)
    proj = project_ces(ces)
    assert proj.degree_spectrum == ces.relations.degree_spectrum()


def test_project_ces_analytical_requires_cap():
    from pyphi.relations import AnalyticalRelations
    from pyphi.visualize.projection import project_ces

    ces = _xor_ces(analytical=True)
    assert isinstance(ces.relations, AnalyticalRelations)  # precondition
    with pytest.raises(ValueError, match="max_relations"):
        project_ces(ces)


def test_project_ces_analytical_matches_concrete_and_sizes_faithfully():
    from pyphi.visualize.projection import project_ces

    concrete = project_ces(_xor_ces(analytical=False))
    n = len(concrete.edges)
    analytical = project_ces(_xor_ces(analytical=True), max_relations=n)
    # Same rendered relations (as relata sets) when the cap covers all.
    assert {e.relata for e in analytical.edges} == {e.relata for e in concrete.edges}
    # Node sizing is faithful: identical to the concrete incident Σφ_r.
    assert [node.sum_phi_relations for node in analytical.nodes] == pytest.approx(
        [node.sum_phi_relations for node in concrete.nodes]
    )
    # And independent of the cap: a tight cap does not change node sizes.
    capped = project_ces(_xor_ces(analytical=True), max_relations=1)
    assert [node.sum_phi_relations for node in capped.nodes] == pytest.approx(
        [node.sum_phi_relations for node in analytical.nodes]
    )
    assert len(capped.edges) <= 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest test/visualize/test_visualize_projection.py -k "degree_spectrum or analytical" -x`
Expected: FAIL — `project_ces()` has no `max_relations` parameter / `CESProjection` has no `degree_spectrum`.

- [ ] **Step 3: Add the `field` import and the `degree_spectrum` field**

In `pyphi/visualize/projection/__init__.py`, change the dataclasses import:

```python
from dataclasses import dataclass
from dataclasses import field
```

In `class CESProjection`, add the field after `faces`:

```python
    endpoints: tuple[EndpointNode, ...] = ()
    faces: tuple[RelationFaceEdge, ...] = ()
    degree_spectrum: dict[int, tuple[int, float]] = field(default_factory=dict)
```

- [ ] **Step 4: Delete `_sum_phi_relations` and rewrite `_faces`**

Delete the whole `_sum_phi_relations` function. Replace the `_faces` function with a version that takes an iterable of `Relation` objects instead of reading `faces_by_degree`:

```python
def _faces(relations, mechanism_to_id) -> tuple[RelationFaceEdge, ...]:
    faces = []
    for relation in relations:
        for face in relation.faces:
            endpoint_ids = tuple(
                sorted(
                    2 * mechanism_to_id[tuple(relatum.mechanism)]
                    + (0 if relatum.direction.name == "CAUSE" else 1)
                    for relatum in face
                )
            )
            faces.append(
                RelationFaceEdge(
                    endpoints=endpoint_ids,
                    degree=len(face),
                    phi=float(face.phi),
                    overlap=_unit_indices(face.overlap),
                )
            )
    faces.sort(key=lambda f: (f.degree, f.endpoints, f.phi))
    return tuple(faces)
```

- [ ] **Step 5: Rewrite `project_ces`**

Replace the body of `project_ces` (keep the leading `relation_closed` guard and its `TypeError` unchanged) so it reads:

```python
def project_ces(ces, node_labels=None, max_relations=None) -> CESProjection:
    """Project a :class:`~pyphi.models.ces.CauseEffectStructure` into plot-ready data.

    Parameters
    ----------
    ces : CauseEffectStructure
        The cause-effect structure to project. Must be relation-closed.
    node_labels : NodeLabels, optional
        Labels for substrate units. Defaults to the labels carried by the first
        distinction.
    max_relations : int, optional
        Render only the ``max_relations`` strongest relations (and their faces),
        in descending φ_r order. If None, render every relation; a relation set
        that cannot be enumerated (the analytical backend) then raises, since
        "every relation" is unbounded. Node marker sizes and the degree spectrum
        are always computed over the full structure, independent of this cap.

    Returns
    -------
    CESProjection

    Raises
    ------
    TypeError
        If ``ces`` is not relation-closed (e.g. a :class:`PhiFold`, whose
        relations may reference distinctions outside it).
    ValueError
        If ``max_relations`` is None and ``ces.relations`` is not enumerable.
    """
    if not getattr(ces, "relation_closed", True):
        raise TypeError(
            "cannot project a view that is not relation-closed (e.g. a PhiFold, "
            "whose relations may reference distinctions outside it); project "
            "the parent structure or an induced substructure, or use "
            "highlight_phi_fold to visualize a fold"
        )
    distinctions = list(ces.distinctions)
    if node_labels is None:
        node_labels = distinctions[0].node_labels
    mechanism_to_id = {tuple(d.mechanism): i for i, d in enumerate(distinctions)}
    if max_relations is None:
        try:
            iter(ces.relations)
        except TypeError:
            raise ValueError(
                "relations are not enumerable (analytical backend); pass "
                "max_relations=N to render the strongest N relations by φ_r"
            ) from None
    top = list(ces.relations.strongest(k=max_relations))
    edges = tuple(
        RelationEdge(
            relata=tuple(sorted(mechanism_to_id[tuple(m)] for m in relation.mechanisms)),
            degree=len(relation),
            phi=float(relation.phi),
            overlap=_unit_indices(relation.purview),
        )
        for relation in top
    )
    mechanism_inclusion = _inclusion_order(
        tuple(frozenset(d.mechanism) for d in distinctions)
    )
    unions = tuple(
        frozenset(getattr(u, "index", u) for u in d.purview_union) for d in distinctions
    )
    purview_union_inclusion = _inclusion_order(unions)
    sums = ces.relations.sum_phi_by_distinction(distinctions)
    nodes = tuple(
        DistinctionNode(
            id=i,
            mechanism=tuple(d.mechanism),
            label=str(d.mechanism_label),
            cause_purview=tuple(d.cause_purview),
            effect_purview=tuple(d.effect_purview),
            mechanism_state=tuple(d.mechanism_state),
            phi=float(d.phi),
            sum_phi_relations=sums[i],
            includes=bool(purview_union_inclusion.covers[i]),
            included=any(i in c for c in purview_union_inclusion.covers),
        )
        for i, d in enumerate(distinctions)
    )
    return CESProjection(
        nodes=nodes,
        edges=edges,
        mechanism_inclusion=mechanism_inclusion,
        purview_union_inclusion=purview_union_inclusion,
        node_labels=node_labels,
        endpoints=_endpoints(distinctions, node_labels),
        faces=_faces(top, mechanism_to_id),
        degree_spectrum=ces.relations.degree_spectrum(),
    )
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest test/visualize/test_visualize_projection.py -v`
Expected: PASS (including the pre-existing `test_project_xor_*` tests, whose edge/face content is unchanged when uncapped).

- [ ] **Step 7: Commit**

```bash
git add pyphi/visualize/projection/__init__.py test/visualize/test_visualize_projection.py
git commit -m "Project relations via strongest(k) with faithful sizing

Edges and faces come from the top-k strongest relations, so an analytical
(non-enumerable) structure renders with a cap and raises without one. Node
sizing uses sum_phi_by_distinction and the new degree_spectrum field is
closed-form, both independent of the cap."
```

---

### Task 3: Thread `max_relations` through the plot entry points

`plot_ces` and `highlight_phi_fold` accept and forward `max_relations` so callers can render analytical structures.

**Files:**
- Modify: `pyphi/visualize/__init__.py` — `plot_ces` (signature ~line 46, `project_ces` call ~line 154) and `highlight_phi_fold` (signature ~line 202, `project_ces` call ~line 257).
- Test: `test/visualize/test_visualize_projection.py` (a plotly-guarded entry-point test).

**Interfaces:**
- Consumes: `project_ces(..., max_relations=...)` (Task 2).
- Produces: `plot_ces(ces_, *, ..., max_relations=None)` and `highlight_phi_fold(ces_, phi_fold=None, *, ..., max_relations=None)` forwarding to `project_ces`.

- [ ] **Step 1: Write the failing test**

Add to `test/visualize/test_visualize_projection.py`:

```python
def test_plot_ces_forwards_max_relations():
    pytest.importorskip("plotly")
    import pyphi
    from pyphi import examples
    from pyphi.visualize import plot_ces

    with pyphi.config.override(relation_computation="ANALYTICAL"):
        ces = examples.xor_system().ces()
        with pytest.raises(ValueError, match="max_relations"):
            plot_ces(ces)
        fig = plot_ces(ces, max_relations=2)
    assert fig is not None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest test/visualize/test_visualize_projection.py::test_plot_ces_forwards_max_relations -x`
Expected: FAIL — `plot_ces()` got an unexpected keyword argument `max_relations` (or the no-cap call does not raise).

- [ ] **Step 3: Add `max_relations` to `plot_ces`**

In `pyphi/visualize/__init__.py`, add the parameter to the keyword-only block of `plot_ces` (after `star_min_degree=None`):

```python
    star_min_degree=None,
    max_relations=None,
```

Change its `project_ces` call:

```python
    projection = project_ces(ces_, node_labels=node_labels, max_relations=max_relations)
```

Add to the `plot_ces` docstring, after the `star_min_degree` parameter entry:

```
    max_relations : int, optional
        Render only the strongest ``max_relations`` relations by φ_r. Required
        when the structure's relations are computed analytically (the set is
        not enumerable); node sizes and the spectrum view remain exact
        regardless. If None, every relation is rendered.
```

- [ ] **Step 4: Add `max_relations` to `highlight_phi_fold`**

Add the parameter to the keyword-only block of `highlight_phi_fold` (after `show=None`):

```python
    show=None,
    max_relations=None,
```

Change its `project_ces` call:

```python
    projection = project_ces(ces_, node_labels=node_labels, max_relations=max_relations)
```

Add to the `highlight_phi_fold` docstring, after the `show` parameter entry:

```
    max_relations : int, optional
        Render only the strongest ``max_relations`` relations by φ_r; required
        for analytically-computed relations.
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest test/visualize/test_visualize_projection.py::test_plot_ces_forwards_max_relations -v`
Expected: PASS (or SKIP if plotly is not installed).

- [ ] **Step 6: Commit**

```bash
git add pyphi/visualize/__init__.py test/visualize/test_visualize_projection.py
git commit -m "Forward max_relations through plot_ces and highlight_phi_fold"
```

---

### Task 4: Spectrum view reads the closed-form degree spectrum

`render_relation_spectrum` uses `projection.degree_spectrum` instead of iterating `projection.edges`, so the spectrum is exact and independent of any relation cap.

**Files:**
- Modify: `pyphi/visualize/render/spectrum.py`
- Test: `test/visualize/test_visualize_spectrum.py`

**Interfaces:**
- Consumes: `CESProjection.degree_spectrum` (Task 2).

- [ ] **Step 1: Write the failing test**

Add to `test/visualize/test_visualize_spectrum.py`:

```python
def test_spectrum_is_cap_independent_on_analytical():
    pytest.importorskip("plotly")
    import pyphi
    from pyphi import examples
    from pyphi.visualize import plot_ces

    with pyphi.config.override(relation_computation="ANALYTICAL"):
        ces = examples.xor_system().ces()
        spectrum = ces.relations.degree_spectrum()
        fig = plot_ces(ces, view="spectrum", max_relations=1)
    # Bars reflect the full closed-form spectrum, not the single rendered edge.
    (bar,) = fig.data
    assert list(bar.x) == sorted(spectrum)
    assert list(bar.y) == pytest.approx([spectrum[d][1] for d in sorted(spectrum)])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest test/visualize/test_visualize_spectrum.py::test_spectrum_is_cap_independent_on_analytical -x`
Expected: FAIL — the bar heights reflect only the single capped edge (or the y-values do not match the full spectrum). SKIP if plotly absent — in that case verify against the concrete backend instead by temporarily setting `max_relations=None`; the test as written is the target once plotly is present.

- [ ] **Step 3: Rewrite the spectrum renderer**

Replace the top of `render_relation_spectrum` in `pyphi/visualize/render/spectrum.py`. Remove `from collections import defaultdict`. Replace the count/sum loop:

```python
def render_relation_spectrum(
    projection: CESProjection, theme: Theme, fig: go.Figure | None = None
) -> go.Figure:
    """A 2-D bar panel of relation count and sum of φ per relation degree.

    Reads the projection's closed-form ``degree_spectrum`` (count and Σφ_r per
    degree), so the high-degree structure that is hard to read in the 3-D
    hypergraph view is summarized exactly, whatever relation cap the other
    views use.
    """
    spectrum = projection.degree_spectrum
    degrees = sorted(spectrum)
    count = {d: spectrum[d][0] for d in degrees}
    sum_phi = {d: spectrum[d][1] for d in degrees}
    figure = go.Figure() if fig is None else fig
```

Leave the rest of the function (the `add_trace` and `update_layout` calls) unchanged — it already reads `count`, `sum_phi`, and `degrees`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest test/visualize/test_visualize_spectrum.py -v`
Expected: PASS (the pre-existing spectrum tests still pass on concrete relations, since `degree_spectrum` equals what the edge loop produced when uncapped).

- [ ] **Step 5: Commit**

```bash
git add pyphi/visualize/render/spectrum.py test/visualize/test_visualize_spectrum.py
git commit -m "Render the relation spectrum from the closed-form degree spectrum

Reading projection.degree_spectrum keeps the spectrum exact under an
analytical relation cap, where iterating the capped edge set would show a
truncated census."
```

---

### Task 5: Relation statistic deltas in the cause-effect-structure diff

`CauseEffectStructure._changes` always emits relation statistic deltas (Σφ_r, count, per-degree) — closed-form on both backends — and keeps per-relation gained/lost rows where both sides are enumerable.

**Files:**
- Modify: `pyphi/models/ces.py` — the relation block of `_changes` (~lines 417-430).
- Test: `test/models/test_result_diff.py`

**Interfaces:**
- Consumes: `Relations.sum_phi()`, `num_relations()`, `degree_spectrum()` (existing on every backend).
- Produces: new `Change.kind` values `"relation_sum_phi"`, `"relation_count"`, `"relation_degree"` on the `ResultDiff` returned by `CauseEffectStructure.diff`.

- [ ] **Step 1: Write the failing test**

Add to `test/models/test_result_diff.py`:

```python
def test_ces_diff_relation_statistic_deltas_on_analytical():
    """Analytical relations produce relation statistic deltas (Σφ_r, count,
    per-degree) even though they cannot be enumerated for gained/lost rows."""
    import pyphi
    from pyphi import examples
    from pyphi.conf import presets

    substrate = examples.basic_substrate()
    state_a = examples.basic_state()
    state_b = tuple(1 - v if i == 0 else v for i, v in enumerate(state_a))
    with pyphi.config.override(**presets.iit4_2023, relation_computation="ANALYTICAL"):
        a = substrate.ces(state_a)
        b = substrate.ces(state_b)
    from pyphi.relations import AnalyticalRelations

    assert isinstance(a.relations, AnalyticalRelations)  # precondition
    rd = a.diff(b)
    kinds = {c.kind for c in rd.changes}
    # At least one statistic delta is present, and no per-relation rows (the
    # analytical backend is not enumerable).
    assert kinds & {"relation_sum_phi", "relation_count", "relation_degree"}
    assert not (kinds & {"relation_gained", "relation_lost"})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest test/models/test_result_diff.py::test_ces_diff_relation_statistic_deltas_on_analytical -x`
Expected: FAIL — no statistic-delta kinds are present (today the analytical branch yields no relation changes).

- [ ] **Step 3: Rewrite the relation block of `_changes`**

In `pyphi/models/ces.py`, replace the block that currently builds `a_rels` / `b_rels` and the `relation_lost` / `relation_gained` extends (the code from `a_rels = (` through the second `changes.extend(...)`), with:

```python
        a_rels, b_rels = self.relations, other.relations
        if not numerics.eq(float(a_rels.sum_phi()), float(b_rels.sum_phi())):
            changes.append(
                Change(
                    "relation_sum_phi",
                    None,
                    a_value=a_rels.sum_phi(),
                    b_value=b_rels.sum_phi(),
                )
            )
        if a_rels.num_relations() != b_rels.num_relations():
            changes.append(
                Change(
                    "relation_count",
                    None,
                    a_value=a_rels.num_relations(),
                    b_value=b_rels.num_relations(),
                )
            )
        a_spec, b_spec = a_rels.degree_spectrum(), b_rels.degree_spectrum()
        for degree in sorted(a_spec.keys() | b_spec.keys()):
            av, bv = a_spec.get(degree), b_spec.get(degree)
            same = (
                av is not None
                and bv is not None
                and av[0] == bv[0]
                and numerics.eq(av[1], bv[1])
            )
            if not same:
                changes.append(
                    Change("relation_degree", degree, a_value=av, b_value=bv)
                )
        if hasattr(a_rels, "__iter__") and hasattr(b_rels, "__iter__"):
            a_set, b_set = set(a_rels), set(b_rels)
            changes.extend(
                Change("relation_lost", tuple(sorted(r.mechanisms)), a_value=r.phi)
                for r in a_set - b_set
            )
            changes.extend(
                Change("relation_gained", tuple(sorted(r.mechanisms)), b_value=r.phi)
                for r in b_set - a_set
            )
        return tuple(changes)
```

Update the `Change` docstring `kind` list in `pyphi/models/diff.py` to include the new keys, inserting after `"relation_lost"`:

```python
    ``"relation_lost"`` / ``"relation_sum_phi"`` / ``"relation_count"`` /
    ``"relation_degree"`` /
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest test/models/test_result_diff.py -v`
Expected: PASS — the new analytical test passes; `test_ces_diff_distinctions_and_relations` (identical CESs → `changes == ()`) still passes because every statistic is equal; `test_ces_diff_with_differing_relation_sets` (concrete) still finds `relation_gained` / `relation_lost` and now additionally carries statistic deltas.

- [ ] **Step 5: Commit**

```bash
git add pyphi/models/ces.py pyphi/models/diff.py test/models/test_result_diff.py
git commit -m "Report relation statistic deltas in the cause-effect-structure diff

Σφ_r, relation count, and the per-degree spectrum are differenced on every
backend, so an analytical structure produces relation deltas instead of the
silent empty set. Per-relation gained/lost rows are kept where both sides are
enumerable."
```

---

### Task 6: Changelog fragment

**Files:**
- Create: `changelog.d/relations-viz-diff-analytical.feature.md`

- [ ] **Step 1: Write the fragment**

```bash
cat > changelog.d/relations-viz-diff-analytical.feature.md <<'EOF'
`plot_ces` now renders cause-effect structures backed by analytically-computed
relations: pass `max_relations=N` to draw the N strongest relations by φ_r
(node sizes and the spectrum view stay exact over the full structure).
`CauseEffectStructure.diff` now reports relation statistic deltas (Σφ_r, count,
and the per-degree spectrum) on both relation backends, alongside per-relation
gained/lost rows where relations are enumerable.
EOF
```

- [ ] **Step 2: Verify towncrier accepts it**

Run: `uv run towncrier build --draft --version 0.0.0`
Expected: the draft output includes the new fragment text under a features heading; no error.

- [ ] **Step 3: Commit**

```bash
git add changelog.d/relations-viz-diff-analytical.feature.md
git commit -m "Add changelog fragment for analytical-safe relations viz and diff"
```

---

### Final verification

- [ ] **Full suite with doctests (no path argument, per project policy):**

Run: `uv run pytest`
Expected: green. Pay attention to `pyphi/relations.py`, `pyphi/visualize/projection/__init__.py`, and `pyphi/models/ces.py` doctests (collected only in this unqualified invocation).

- [ ] **Type check:**

Run: `uv run pyright pyphi/relations.py pyphi/visualize/projection/__init__.py pyphi/visualize/__init__.py pyphi/visualize/render/spectrum.py pyphi/models/ces.py`
Expected: no new errors.

## Self-review notes

- **Spec coverage:** viz `strongest(k)` + `max_relations` + uncapped error (Task 2/3); faithful node sizing via `sum_phi_by_distinction` (Task 1/2); closed-form spectrum (Task 2/4); diff statistic deltas + per-relation detail where enumerable (Task 5); changelog (Task 6). All spec sections map to a task.
- **Type consistency:** `sum_phi_by_distinction(distinctions) -> tuple[float, ...]` and `degree_spectrum() -> dict[int, tuple[int, float]]` are used identically across tasks; `project_ces(ces, node_labels=None, max_relations=None)` matches its two call sites in Task 3.
- **IIT 3.0 path:** `NullRelations` answers `sum_phi()`/`num_relations()` as `0` and `degree_spectrum()` as `{}`, and is iterable, so the diff on identical IIT 3.0 structures still yields `changes == ()`.
