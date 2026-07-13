# Documentation Overhaul for the IIT 4.0 (2026) Default — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** `docs/superpowers/specs/2026-07-12-docs-overhaul-2026-default-design.md`

**Goal:** Rework the user-facing documentation so it genuinely teaches the IIT 4.0 (2026) default — a nonzero first result on the paper's own Fig 1A example, one worked Fig 1→2→4 arc, and the determinism ⇒ φ_s = 0 property taught as an explicit lesson instead of read as a bug.

**Architecture:** One code change (promote the Fig 1A logistic builder from `test/integration/test_paper_reproduction.py` to public `pyphi.examples.iit4_2023_fig1a_substrate()` / `iit4_2023_fig1a_system()`, with the test importing it back), then a docs-only rework: getting-started and the theory pipeline pages move onto Fig 1A; the two overlapping tutorials collapse into one worked arc on Fig 1A; a new theory page teaches the intrinsic-information cap using `xor`; the remaining docs are either example-swapped, explicitly pinned with a note, or verified safe. All executable MyST pages regenerate at docs build time (`nb_execution_raise_on_error = True`), so hardcoded prose numbers and the standalone demo notebook are the only stale-φ risks.

**Tech Stack:** Sphinx + myst-nb (executable MyST pages, `nb_execution_mode = "cache"`), jupytext 1.19.4 (paired `.md` ↔ `.ipynb` tutorials), pytest, `uv run` for everything Python, `just docs` for the build.

## Global Constraints

Copied from the spec — every task's requirements implicitly include these:

- **Invariant: no doc silently shows a φ value that is wrong under the shipping default.** Every user-facing doc that computes under the default and shows a φ value must either (a) use Fig 1A or another probabilistic example, or (b) explicitly pin the formalism it means to demonstrate (`with config.override(**presets.iit4_2023)` etc.) with a one-line note.
- **No `pyphi/` computation logic changes** except adding the Fig 1A example functions. No re-deriving or changing any φ value or formalism behavior.
- **Executable MyST pages regenerate at build time**, so the risk is committed notebook outputs and hardcoded φ numbers in prose. (Verified: all paired `.ipynb` files have zero committed outputs; the only notebook with content risk is `docs/examples/IIT_4.0_demo.ipynb`, which is excluded from the Sphinx build and runs under the ambient default when a user executes it.)
- **`uv run` for all Python commands**; `just docs` for the docs build.
- **Non-goals:** the analytical-relations default (`c511e8bc`, separate follow-up); the ii-gate build; `docs/superpowers/` specs and plans (historical records, not user docs).

Repository conventions that bind this plan:

- Paired tutorial pages carry `formats: md:myst,ipynb` frontmatter. After editing such a `.md`, regenerate its `.ipynb` with `uv run jupytext --sync <file>.md`. Paired pages: `getting-started/first-computation.md` and `tutorials/{worked-example,cause-effect-structure,macro,actual-causation,recursive-exclusion,causal-reductionism}.md`. Theory and how-to pages are executable but unpaired — no sync needed.
- **Do not touch concurrent-session files:** `docs/examples/relations_query_surface.ipynb` (untracked, in flight), any `docs/examples/analytical_relations.ipynb` that appears, `experiments/`, `benchmarks/iit_3_vs_4/results/`. Stage only files this plan names; never `git add -A`.
- Docstrings follow the repo standard (NumPy sections, final-state voice, Unicode symbols, paper-verified citations — see `CLAUDE.md`).
- User-facing changes need `changelog.d/` fragments (Tasks 1 and 10).

## Verified reference values (ground truth, computed 2026-07-12 under `IIT_4_0_2026` default)

All doc prose in this plan cites these values; they were computed on this machine with the exact builder being promoted. Cells in executable pages regenerate at build time and must match.

**Fig 1A** (weights below; state `(0, 1, 1)` = paper's "aBC"; labels A, B, C):

| Quantity | Value |
| --- | --- |
| φ_s of subset `(0,)` (a) | 0.0396 (paper Fig 1E: 0.04) |
| φ_s of subset `(0, 1)` (aB) | 0.1719 (paper: 0.17) |
| φ_s of subset `(0, 1, 2)` (aBC, whole) | 0.1339 (paper: 0.13) |
| φ_s of subset `(2,)` (C) | 0.2122 |
| aB: φ_c / φ_e | 0.2446 / 0.1719 (paper Fig 1D: 0.24 / 0.17); φ_s = min = φ_e |
| `substrate.complexes((0,1,1))` | `((2,), φ=0.2122, maximal)` and `((0,1), φ=0.1719)` — non-overlapping; aB **is a complex** (beats every overlapping candidate), {C} is the global maximum |
| aB distinctions | mech `(0,)`: φ_d 0.3327, cause `(1,)`, effect `(1,)`; mech `(1,)`: φ_d 0.3236, cause `(0,)`, effect `(0,1)`; mech `(0,1)`: φ_d 0.0714, cause `(1,)`, effect `(0,1)` (paper Fig 2: 0.33 / 0.32 / 0.07) |
| aB relations | 7 total (including self-relations); r({a, aB}): φ_r 0.0357, 9 faces (paper Fig 4: 0.035) |
| aB CES sums | Σφ_d = 0.7278, Σφ_r = 0.8349, Φ = 1.5627 |
| whole-substrate (aBC) CES | 6 distinctions, 45 relations, Φ = 4.3657 |
| identical under `iit4_2023` | yes (probabilistic; the cap does not bind) |

**Deterministic classics and contrasts:**

| System, state | 2026 (default) φ_s | 2023 φ_s | IIT 3.0 | Notes |
| --- | --- | --- | --- | --- |
| `xor`, (0,0,0) | 0.0 | 1.5 | 1.875 | φ_c=1.5, φ_e=3.0; specified-state ii: cause 1.0, effect 2.0; **intrinsic differentiation: 0.0 both directions**; CES unchanged: 4 distinctions, 15 relations, Φ = 9.5 |
| `basic`, (1,1,0) | 0.0 | 0.2075 | 0.1875 | CES: 3 distinctions, Φ = 1.8568 |
| `basic`, (1,0,0) | 0.0 | 0.4150 | — | CES: 2 distinctions, Φ = 1.0 |
| `grid3`, (0,0,0) | 0.0247 | — | — | probabilistic; 7 distinctions, 39 relations, Φ = 6.4986 |

**`howto/landscape.md` claims re-verified under the default** (state (1,0,0), Fig 1A weights, A→B axis): MIP switch bracketed at (0.45, 0.475) ✓ ("near θ ≈ 0.45" holds); `switch_distances["cause_state"]` = 0.00166 ✓ ("about 0.0017" holds); crossing at 0.704 collapses φ to 0 with cause state (0,1,1) → (0,1,0) ✓. Page needs no numeric changes.

**Builder equivalence verified:** `build_substrate([ising.probability] * 3, weights, temperature=1/4)` produces a `Substrate` equal (TPM, cm, labels A/B/C) to the test's manual sigmoid construction (Eq. 60, k = 4).

---

### Task 1: Promote the Fig 1A example to `pyphi.examples`

**Files:**
- Modify: `pyphi/examples.py` (insert before `iit4_2023_fig6a_substrate`, currently line ~1556)
- Modify: `test/integration/test_paper_reproduction.py:91-160` (module docstring lines 15-25 and 66-69 also)
- Create: `changelog.d/iit4-fig1a-example.feature.md`

**Interfaces:**
- Consumes: `pyphi.substrate_generator.build_substrate`, `pyphi.substrate_generator.ising` (already imported at top of `examples.py`), `pyphi.system.System` (already imported), `register_example` decorator.
- Produces (every doc task below calls these):
  - `pyphi.examples.iit4_2023_fig1a_substrate() -> Substrate` — the 3-unit Fig 1A logistic substrate, labels `("A", "B", "C")`.
  - `pyphi.examples.iit4_2023_fig1a_system() -> System` — the whole substrate in the paper's state `(0, 1, 1)`, `node_indices=(0, 1, 2)` (matches the `iit4_2023_fig6*_system` convention: full system in the canonical state).
  - Registered as `EXAMPLES["substrate"]["iit4_2023_fig1a"]` / `EXAMPLES["system"]["iit4_2023_fig1a"]` (automatic via the name-derived registry).

Naming note (spec left names TBD): the spec's suggested `iit4_fig1a_*` would sit ambiguously next to the existing IIT-3.0-era `fig1a_substrate()` (examples.py line ~785) and break the established `iit4_2023_fig6a/6b/.../fig7` family convention. `iit4_2023_fig1a_*` is the consistent choice; the docs refer to it as "the Fig 1A example".

- [ ] **Step 1: Write the failing test re-point.** In `test/integration/test_paper_reproduction.py`, delete `_FIG1_K` and `_FIG1_WEIGHTS` (lines 110-117) and the `_fig1_substrate` function (lines 145-155) together with the weight-derivation comment block (lines 94-109); keep `_FIG1_STATE`, `_FIG1_PUBLISHED_PHI_S`, and `_FIG2_DISTINCTIONS`. Add the alias so no other line changes:

```python
# Fig 1A: the substrate is shown in state aBC = (a off, B on, C on). Lowercase
# denotes state "-1" (PyPhi 0), uppercase "+1" (PyPhi 1). The substrate itself
# ships as a public example.
_fig1_substrate = examples.iit4_2023_fig1a_substrate
_FIG1_STATE = (0, 1, 1)
```

Also update the module docstring: in the Fig 1 bullet (lines 15-25), note the substrate ships as `pyphi.examples.iit4_2023_fig1a_substrate`; extend the closing provenance paragraph (lines 66-69) to include Fig 1A alongside the Fig 6/7 substrates.

- [ ] **Step 2: Run the re-pointed tests to verify they fail** (the example does not exist yet):

Run: `uv run pytest test/integration/test_paper_reproduction.py -k "fig1 or fig2 or fig4" -x -q`
Expected: FAIL with `AttributeError: module 'pyphi.examples' has no attribute 'iit4_2023_fig1a_substrate'`

- [ ] **Step 3: Add the example functions.** In `pyphi/examples.py`, immediately before `iit4_2023_fig6a_substrate`, insert:

```python
@register_example
def iit4_2023_fig1a_substrate():
    """The 3-unit logistic substrate of Fig 1A.

    The example the IIT 4.0 paper introduces the theory with (Albantakis et
    al. 2023, Figs 1, 2 and 4). Each unit's activation is a logistic function
    of its weighted inputs in {−1, +1} (Eq. 60) with slope k = 4. The
    connection weights are read from the Fig 1A causal model::

        A→A = −0.2   A→B = +0.7   A→C = +0.2
        B→A = +0.7   B→B = −0.2   (no B→C)
        (no C→A)     C→B = −0.8   C→C = +0.2

    This reading is self-validating: in the canonical state aBC = (0, 1, 1),
    the three φₛ values published in Fig 1E — 0.04 for {A}, 0.17 for the
    complex {A, B}, 0.13 for {A, B, C} — all reproduce to the paper's
    two-decimal precision, which they would not if any weight were misread.
    Because the substrate is probabilistic, these values are identical under
    the 2023 and 2026 formalisms.
    """
    # fmt: off
    weights = np.array([
        [-0.2, 0.7, 0.2],
        [0.7, -0.2, 0.0],
        [0.0, -0.8, 0.2],
    ])
    # fmt: on
    return build_substrate([ising.probability] * 3, weights, temperature=1 / 4)


@register_example
def iit4_2023_fig1a_system():
    return System(
        iit4_2023_fig1a_substrate(),
        state=(0, 1, 1),
        node_indices=(0, 1, 2),
    )
```

(`build_substrate([ising.probability] * 3, weights, temperature=1/4)` is verified equal to the test's manual sigmoid construction — TPM, cm, and default labels A/B/C all match.)

- [ ] **Step 4: Run the paper-reproduction tests to verify they pass:**

Run: `uv run pytest test/integration/test_paper_reproduction.py -k "fig1 or fig2 or fig4" -q -m "not slow"`
Expected: PASS (the fig1/fig2/fig4 tests are not slow-marked; the five published φₛ pins, the aB-is-a-complex test, the Fig 2 distinctions, and the Fig 4 relation all reproduce).

- [ ] **Step 5: Run the examples registry and doctest sweep:**

Run: `uv run pytest pyphi/examples.py test/test_examples.py -q` (if `test/test_examples.py` does not exist, run `uv run pytest pyphi/examples.py -q` — this exercises `--doctest-modules` on the module and the registry name validation at import).
Expected: PASS.

- [ ] **Step 6: Create the changelog fragment:**

```bash
echo 'Added `pyphi.examples.iit4_2023_fig1a_substrate()` and `iit4_2023_fig1a_system()`: the 3-unit logistic network the IIT 4.0 paper introduces the theory with (Figs 1, 2, 4), previously available only inside the test suite.' > changelog.d/iit4-fig1a-example.feature.md
```

- [ ] **Step 7: Commit** (only the three named files):

```bash
git add pyphi/examples.py test/integration/test_paper_reproduction.py changelog.d/iit4-fig1a-example.feature.md
git commit -m "Promote the IIT 4.0 Fig 1A network to pyphi.examples

iit4_2023_fig1a_substrate()/iit4_2023_fig1a_system() join the
iit4_2023_fig6*/fig7 example family; the N1 paper-reproduction tests
import the example instead of a private duplicate builder."
```

---

### Task 2: Getting started on Fig 1A

**Files:**
- Modify: `docs/getting-started/first-computation.md` (full rewrite of the computing sections; frontmatter, download/Colab links, and install section unchanged)
- Regenerate: `docs/getting-started/first-computation.ipynb` (jupytext sync)

The page currently builds `basic_substrate()` at `(1, 0, 0)` and shows `analysis.phi` — which regenerates to **0.0** under the default. Rewrite it on Fig 1A so the newcomer's first φ is nonzero and the complex is the "aha".

- [ ] **Step 1: Rewrite the page body.** Keep lines 1-38 (frontmatter, title, download/Colab links, ten-minute framing, install, import + progress-bars cell) except: in the intro paragraph (lines 20-23), replace "build a small substrate" with "build a small substrate from the IIT 4.0 paper". Replace everything from `## Build a substrate` through `## Where to go next` with:

````markdown
## Build a substrate

A {class}`~pyphi.substrate.Substrate` is a set of interacting units defined by a
transition probability matrix and a connectivity matrix. PyPhi ships the
example systems used in the IIT literature; we will use the three-unit network
the IIT 4.0 paper introduces the theory with (Albantakis et al. 2023, Fig 1A) —
three units `A`, `B`, and `C`, each a noisy logistic function of its inputs.

```{code-cell} python
substrate = pyphi.examples.iit4_2023_fig1a_substrate()
substrate
```

The connectivity matrix shows which units influence which, and the transition
probability matrix gives the probability that each unit turns on given the
current state of the system. Note the probabilities are strictly between 0
and 1: the network is probabilistic, not deterministic.

## Analyze a state

Integrated information is a property of a substrate *in a particular state*. We
use the state analyzed in the paper — `A` off, `B` and `C` on — ordered
`(A, B, C)`, and hand it, together with the substrate, to
{func}`~pyphi.analyze`.

```{code-cell} python
state = (0, 1, 1)
analysis = pyphi.analyze(substrate, state)
analysis
```

That single call runs the whole analysis of the three-unit system: it measures
the system's irreducibility, finds its distinctions, and computes their
relations.

## Read the results

The scalar `analysis.phi` is $\varphi_s$, the system integrated information —
how much the system, as a whole, is irreducible to its parts.

```{code-cell} python
round(analysis.phi, 4)
```

The value is positive: the three units hang together as one system. (It
reproduces the value published in the paper's Fig 1E for this system, 0.13.)

## Find the complexes

Not every subset of units exists as a whole of its own. Subsets *compete*:
among overlapping candidates, only the one with maximal $\varphi_s$ — a
**complex** — exists. {meth}`~pyphi.substrate.Substrate.complexes` runs that
competition over every subset:

```{code-cell} python
for complex_ in substrate.complexes(state):
    print(complex_.node_indices, round(float(complex_.phi), 4))
```

The substrate condenses into two complexes: the single unit `C`, and the pair
`{A, B}` — the complex the paper features in Fig 1E, written "aB". Every other
candidate, including the full three-unit system we just analyzed, is excluded
by one of these two. This is IIT's exclusion postulate in action: $\varphi_s >
0$ makes a candidate *eligible*; being a local maximum among everything it
overlaps makes it a complex.

## The Φ-structure

The richer object is the $\Phi$-structure, available as `analysis.ces` (a
{class}`~pyphi.models.ces.CauseEffectStructure`). It is the collection of
*distinctions* — the irreducible mechanisms the system specifies — together
with the *relations* among them.

```{code-cell} python
ces = analysis.ces
print("distinctions:", len(ces.distinctions))
print("relations:   ", len(ces.relations))
```

Its `big_phi` attribute is the structure integrated information $\Phi$, the
summed $\varphi$ of the distinctions and relations.

```{code-cell} python
round(float(ces.big_phi), 4)
```

## Save the result

Analyses can be expensive, so it is worth saving them. {func}`~pyphi.save`
writes any PyPhi result object to JSON, and {func}`~pyphi.load` reads it back.

```{code-cell} python
pyphi.save(ces, "ces.json")
```

## Where to go next

That is a full PyPhi computation. From here:

- The {doc}`worked example <../tutorials/worked-example>` follows this same
  network through the paper's Figures 1, 2, and 4, reproducing the published
  numbers.
- The theory page {doc}`../theory/overview` explains what these quantities mean
  and how they are defined.
- If you analyze a *deterministic* network and see $\varphi_s = 0$: that is a
  theorem of the default formalism, not a bug — see
  {doc}`../theory/intrinsic-information`.
````

- [ ] **Step 2: Sync the notebook pair:**

Run: `uv run jupytext --sync docs/getting-started/first-computation.md`
Expected: `first-computation.ipynb` regenerated (no outputs).

- [ ] **Step 3: Verify the page's cells produce the plan's numbers.** Run the page end to end:

Run: `uv run --all-extras --group docs jupyter execute docs/getting-started/first-computation.ipynb`
Expected: exit 0. Spot-check by running the key lines in `uv run python`: `round(analysis.phi, 4)` → `0.1339`; complexes → `(2,) 0.2122` and `(0, 1) 0.1719`; `len(ces.distinctions)` → 6; `len(ces.relations)` → 45; `big_phi` → `4.3657`.

- [ ] **Step 4: Commit:**

```bash
git add docs/getting-started/first-computation.md docs/getting-started/first-computation.ipynb
git commit -m "Rewrite getting-started first computation on the Fig 1A example

The previous example (basic_substrate, deterministic) computes phi_s = 0
under the IIT 4.0 (2026) default; Fig 1A is probabilistic, nonzero, and
carries the paper's published values."
```

---

### Task 3: The worked arc — one tutorial following the paper's Fig 1 → 2 → 4

**Files:**
- Modify: `docs/tutorials/worked-example.md` (full rewrite; same path, so the published URL survives)
- Regenerate: `docs/tutorials/worked-example.ipynb`
- Delete: `docs/tutorials/cause-effect-structure.md`, `docs/tutorials/cause-effect-structure.ipynb`
- Modify: `docs/tutorials/index.md` (drop the `cause-effect-structure` toctree entry)
- Modify: `docs/tutorials/iit-4.0-demo.md:34-36` (re-point the `{doc}` cross-reference)

Spec decision resolved here: the arc is **one page**, at the existing `worked-example` path. The current `worked-example.md` (XOR) and `cause-effect-structure.md` (basic) overlap heavily — both tour the CES object — and both compute φ_s = 0 under the default. The XOR material moves to the determinism lesson (Task 5); the CES-object API tour folds into this arc.

- [ ] **Step 1: Rewrite `docs/tutorials/worked-example.md`.** Keep the jupytext frontmatter (lines 1-13) verbatim. Replace everything from the title down with (the download/Colab lines are unchanged — the notebook path survives):

````markdown
# A complete worked example

{download}`Download this page as a Jupyter notebook <worked-example.ipynb>`
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wmayner/pyphi/blob/main/docs/tutorials/worked-example.ipynb)

This page follows the worked example of the IIT 4.0 paper (Albantakis et al.
2023) from start to finish, reproducing its published numbers under PyPhi's
default formalism: **Figure 1** — is a set of units a complex, and how
irreducible is it ($\varphi_s$)? **Figure 2** — what distinctions compose its
cause-effect structure? **Figure 4** — how do those distinctions bind into
relations? One small network carries all three:
{func}`pyphi.examples.iit4_2023_fig1a_substrate`.

```{code-cell} python
import pyphi

pyphi.config.progress_bars = False
```

## The substrate: three logistic units

Fig 1A defines three units $A$, $B$, $C$. Each unit's probability of turning
ON is a logistic function (slope $k = 4$) of its weighted inputs, with the
inputs read as $\pm 1$ (paper Eq. 60). $A$ and $B$ excite each other strongly
($\pm 0.7$), $C$ inhibits $B$ ($-0.8$), and each unit weakly affects itself.

```{code-cell} python
substrate = pyphi.examples.iit4_2023_fig1a_substrate()
substrate
```

The paper analyzes the state written "aBC": $A$ off, $B$ and $C$ on.
(Lowercase marks an OFF unit.)

```{code-cell} python
state = (0, 1, 1)
```

## Figure 1: integration and exclusion

Fig 1E asks which candidate systems exist. We compute $\varphi_s$ for the
three candidates the paper compares — the single unit $\{A\}$, the pair
$\{A, B\}$ ("aB"), and the whole substrate:

```{code-cell} python
for subset in [(0,), (0, 1), (0, 1, 2)]:
    analysis = pyphi.analyze(substrate, state, subset=subset)
    print(subset, round(analysis.phi, 4))
```

These reproduce the paper's Fig 1E values: $0.04$, $0.17$, and $0.13$. The
pair aB beats both its subset and its superset — and in fact every candidate
that overlaps it — so **aB is a complex**. PyPhi's exhaustive competition
confirms it, and finds one other, non-overlapping complex:

```{code-cell} python
for complex_ in substrate.complexes(state):
    print(complex_.node_indices, round(float(complex_.phi), 4))
```

The single unit $\{C\}$ has the globally maximal $\varphi_s$ here; since it
does not overlap aB, both exist. The paper presents aB as *a* complex —
maximal among the candidates that share its units — which is exactly what
PyPhi finds.

Where does aB's $\varphi_s$ come from? Fig 1D splits it by temporal
direction. Integration is measured separately over the system's causes and
its effects, and the system is only as integrated as its weaker direction:

```{code-cell} python
aB = pyphi.analyze(substrate, state, subset=(0, 1))
print("φ_c =", round(float(aB.sia.cause.phi), 4))
print("φ_e =", round(float(aB.sia.effect.phi), 4))
print("φ_s =", round(aB.phi, 4))
```

$\varphi_c = 0.24$ and $\varphi_e = 0.17$, the paper's published split; the
effect side is the weaker one, so $\varphi_s = \varphi_e$. The partition
responsible — the minimum information partition — is on the analysis as
`aB.sia.partition`.

## Figure 2: the distinctions

With the complex fixed, the composition postulate unfolds what exists *within*
it. Every subset of aB's units — every **mechanism** — is tested for an
irreducible cause and effect. The irreducible ones are the complex's
**distinctions**, and they live on the cause-effect structure:

```{code-cell} python
ces = aB.ces
for d in ces.distinctions:
    print(
        f"{d.mechanism_label:>3}  φ_d = {float(d.phi):.4f}  "
        f"cause {d.cause_purview}  effect {d.effect_purview}"
    )
```

Three distinctions, matching Fig 2: the first-order mechanisms $a$
($\varphi_d = 0.33$) and $B$ ($0.32$), and the second-order mechanism $aB$
($0.07$). Each specifies the *purviews* shown — the units its cause and effect
power is about. A single distinction prints its full detail, including the
repertoires (the probability distributions it specifies over its purviews):

```{code-cell} python
ces.distinctions[1]
```

The whole set collapses to a table with `to_pandas`, handy for sorting,
filtering, or exporting:

```{code-cell} python
ces.to_pandas()
```

## Figure 4: the relations

Distinctions whose purviews overlap congruently — same units, same specified
state — bind together into **relations**. Fig 4 works out the relation between
the distinctions $a$ and $aB$, which overlap over unit $b$:

```{code-cell} python
relation = next(
    r for r in ces.relations
    if {tuple(m) for m in r.mechanisms} == {(0,), (0, 1)}
)
print("φ_r =", round(float(relation.phi), 4))
print("faces:", relation.num_faces)
```

$\varphi_r = 0.036$ with all $9$ faces, the paper's Fig 4 relation (quoted
there as $0.035$, from the rounded $\varphi_d(aB) = 0.07$ divided over the
two-unit purview union). The structure has seven relations in all — including
*self-relations*, where a single distinction's own cause and effect purviews
overlap:

```{code-cell} python
for r in ces.relations:
    mechs = [tuple(m) for m in r.mechanisms]
    print(f"φ_r = {float(r.phi):.4f}  mechanisms {mechs}")
```

## The Φ-structure, summed

Distinctions and relations together are the complex's $\Phi$-structure, and
their summed $\varphi$ is the **structure integrated information** $\Phi$:

```{code-cell} python
print("Σ φ_d =", round(float(ces.sum_phi_distinctions), 4))
print("Σ φ_r =", round(float(ces.sum_phi_relations), 4))
print("Φ     =", round(float(ces.big_phi), 4))
```

Note that $\Phi$ (the structure's total, $1.56$ here) is a different quantity
from $\varphi_s$ (the system's irreducibility over its minimum partition,
$0.17$ here). Both are reported on the analysis: `aB.phi` is $\varphi_s$;
`aB.ces.big_phi` is $\Phi$.

## Summary

For the paper's Fig 1A network in state aBC, PyPhi reproduces, under the
default formalism:

- $\varphi_s = 0.04 / 0.17 / 0.13$ for $\{A\}$ / aB / aBC (Fig 1E), with
  $\varphi_c = 0.24$, $\varphi_e = 0.17$ for aB (Fig 1D);
- aB as a complex, alongside the non-overlapping complex $\{C\}$;
- aB's three distinctions with $\varphi_d = 0.33, 0.32, 0.07$ and their
  purviews (Fig 2);
- the relation $r(\{a, aB\})$ with $\varphi_r = 0.035$ and 9 faces (Fig 4);
- the summed structure, $\Phi = 1.56$.

## Where to go next

- {doc}`../theory/index` — the same pipeline, quantity by quantity, with the
  paper-to-code map.
- {doc}`../theory/intrinsic-information` — why a *deterministic* network
  computes $\varphi_s = 0$ under this formalism.
- {doc}`iit-4.0-demo` — the paper's own supplementary notebook, going deeper
  into the algorithm.
````

- [ ] **Step 2: Delete the merged tutorial and update references:**

```bash
git rm docs/tutorials/cause-effect-structure.md docs/tutorials/cause-effect-structure.ipynb
```

In `docs/tutorials/index.md`, remove the `cause-effect-structure` line from the toctree (keep `worked-example`). In `docs/tutorials/iit-4.0-demo.md` line 36, change "for a shorter hands-on tour of the cause-effect structure, see {doc}`cause-effect-structure`." to "for a shorter hands-on tour, see {doc}`worked-example`."

- [ ] **Step 3: Sync the notebook pair:**

Run: `uv run jupytext --sync docs/tutorials/worked-example.md`

- [ ] **Step 4: Execute the page and check the published numbers:**

Run: `uv run --all-extras --group docs jupyter execute docs/tutorials/worked-example.ipynb`
Expected: exit 0; the φ printouts match the reference table (0.0396 / 0.1719 / 0.1339; 0.2446 / 0.1719; distinctions 0.3327 / 0.3236 / 0.0714; relation 0.0357, 9 faces; sums 0.7278 / 0.8349 / 1.5627).

- [ ] **Step 5: Commit:**

```bash
git add docs/tutorials/worked-example.md docs/tutorials/worked-example.ipynb docs/tutorials/index.md docs/tutorials/iit-4.0-demo.md
git commit -m "Replace the XOR worked example and CES tutorial with one Fig 1->2->4 arc

One tutorial now follows the IIT 4.0 paper's own worked example on the
promoted Fig 1A network, reproducing the published numbers under the
default formalism. The CES object tour folds into it; the XOR specimen
moves to the intrinsic-information theory lesson."
```

---

### Task 4: Theory pipeline pages onto Fig 1A

**Files:**
- Modify: `docs/theory/overview.md` (lines 82-117, "The worked example")
- Modify: `docs/theory/substrate-and-system.md` (lines 37-95)
- Modify: `docs/theory/system-integration.md` (lines 20-131)
- Modify: `docs/theory/distinctions-and-relations.md` (lines 20-89)
- Modify: `docs/theory/phi-structure.md` (lines 26-80)

These five pages share one running example — currently `basic_substrate()` at `(1, 1, 0)`, **misattributed** as "the three-unit example system from the IIT 4.0 paper", with hardcoded numbers (φ_s ≈ 0.208, Φ ≈ 1.857) that regenerate to 0 under the default. Move the running example to the aB analysis of Fig 1A (which *is* the paper's example), so every quantity on these pages is paper-corroborated. The shared setup cell on each page becomes:

```python
import pyphi

pyphi.config.progress_bars = False

substrate = pyphi.examples.iit4_2023_fig1a_substrate()
analysis = pyphi.analyze(substrate, (0, 1, 1), subset=(0, 1))
```

- [ ] **Step 1: `overview.md`.** Replace the "The worked example" section (lines 82-117) with:

````markdown
## The worked example

One substrate carries this section: the three-unit logistic network the IIT
4.0 paper itself uses to introduce the theory (Figs 1, 2 and 4), available as
`pyphi.examples.iit4_2023_fig1a_substrate()`.

```{code-cell} python
import pyphi

pyphi.config.progress_bars = False

substrate = pyphi.examples.iit4_2023_fig1a_substrate()
substrate
```

The paper analyzes the candidate system $\{A, B\}$ — written "aB" — in the
state where $A$ is off and $B$ and $C$ are on. Analyzing it runs the whole
pipeline:

```{code-cell} python
analysis = pyphi.analyze(substrate, (0, 1, 1), subset=(0, 1))
analysis.phi  # the system integrated information, φ_s
```

This value, $\varphi_s \approx 0.172$, answers the first question: aB is a
complex (the paper's Fig 1E reports it as $0.17$), and $\varphi_s$ measures
how irreducible it is. The second question — the structure — is answered by
the $\Phi$-structure it specifies:

```{code-cell} python
(len(analysis.ces.distinctions), len(analysis.ces.relations), round(float(analysis.ces.big_phi), 3))
```

Three distinctions, seven relations, and a structure integrated information
$\Phi \approx 1.563$. The pages ahead unpack each of these: the [substrate and
system](substrate-and-system.md) it starts from, the [system integrated
information](system-integration.md) $\varphi_s$ that finds the complex, the
[distinctions and relations](distinctions-and-relations.md) that compose the
structure, and [the $\Phi$-structure](phi-structure.md) itself with the full map
from paper symbols to PyPhi types.
````

- [ ] **Step 2: `substrate-and-system.md`.** Replace the substrate cell (lines 39-46) to build `pyphi.examples.iit4_2023_fig1a_substrate()` with the sentence "In PyPhi a substrate is a `Substrate`. The worked example is the paper's three-unit logistic network:". Then rework the "A system is an intrinsic point of view" section (lines 62-95) to *use* the background machinery instead of noting its absence — replace the `System` construction cell and following paragraph with:

````markdown
A `System` is a candidate subset of a substrate in a definite state. The worked
example's candidate is the pair $\{A, B\}$, so unit $C$ is background:

```{code-cell} python
system = pyphi.System(substrate, (0, 1, 1), node_indices=(0, 1))
system
```

Here `node_indices` selects the subset $S = \{A, B\}$; the remaining unit $C$
is causally marginalized, so the analysis sees $A$ and $B$ from their own
intrinsic perspective, with $C$'s influence averaged out.
````

Keep the cause/effect-TPM cell (line 89) and closing paragraph as is (they operate on `system` and regenerate).

- [ ] **Step 3: `system-integration.md`.** Four changes:
  1. Setup cell (lines 20-27): the shared Fig 1A setup above.
  2. The Integration section's worked numbers (lines 56-67): the cell `(analysis.sia.cause.phi, analysis.sia.effect.phi, analysis.phi)` stays; replace the prose "For the worked example the cause direction is the binding one: $\varphi_c \approx 0.208$ is smaller than $\varphi_e \approx 0.415$, so $\varphi_s = \varphi_c \approx 0.208$." with: "For the worked example the effect direction is the binding one: $\varphi_e \approx 0.172$ is smaller than $\varphi_c \approx 0.245$, so $\varphi_s = \varphi_e \approx 0.172$ — the paper's Fig 1D split ($\varphi_c = 0.24$, $\varphi_e = 0.17$)."
  3. Add one sentence at the end of the Integration section, before "Selection margins": "Under the default formalism, $\varphi_s$ is additionally capped by the system's *intrinsic information* — a system must furnish itself alternatives, not merely specify one state. The cap and its consequences (deterministic systems have $\varphi_s = 0$) have {doc}`their own page <intrinsic-information>`."
  4. The Exclusion section (lines 115-131) currently claims "`pyphi.analyze` performs this search" — false (analyze evaluates one candidate). Replace the section body after the postulate sentence with:

````markdown
IIT resolves this by keeping the sets whose integrated information is maximal
among everything they overlap — the **complexes** (Albantakis et al., 2023).
{meth}`~pyphi.substrate.Substrate.complexes` runs this competition over the
whole substrate:

```{code-cell} python
for complex_ in substrate.complexes((0, 1, 1)):
    print(complex_.node_indices, round(float(complex_.phi), 4))
```

Two non-overlapping complexes survive: the single unit $\{C\}$ and the pair
$\{A, B\}$ — the complex the paper features. With a complex fixed, the final
postulate — *composition* — unfolds its internal structure: the
[distinctions and relations](distinctions-and-relations.md) it specifies.
````

  The margins section (lines 74-113) needs no prose change (it hardcodes no values); its cell regenerates.

- [ ] **Step 4: `distinctions-and-relations.md`.** Setup cell → shared Fig 1A setup. Then:
  - The distinctions list cell (lines 44-47) stays; the introduction sentence "For the worked example the complex specifies three distinctions" **stays true** (aB: 3 distinctions).
  - Line 55's comment `# the mechanism-(2,) distinction, φ_d = 0.5` → `# the mechanism-B distinction, φ_d ≈ 0.32`.
  - The Relations section (lines 64-84): replace "The worked example has two relations:" with: "A relation can also bind a *single* distinction with itself, where its own cause and effect purviews overlap congruently — a self-relation. Counting these, the worked example has seven relations:". The list cell and sums cells stay (they regenerate: 7 rows, Σφ_r = 0.835).
  - Closing sentence: "the three distinctions and the two relations" → "the three distinctions and the seven relations".

- [ ] **Step 5: `phi-structure.md`.** Setup cell → shared Fig 1A setup (built as `analysis = pyphi.analyze(pyphi.examples.iit4_2023_fig1a_substrate(), (0, 1, 1), subset=(0, 1))`). Then:
  - Replace the sums prose (lines 39-41): "The three distinction $\varphi_d$ values sum to $0.728$, the seven relation $\varphi_r$ values to $0.835$, and their total is $\Phi \approx 1.563$."
  - Replace the closing answer paragraph (lines 47-50): "…the complex exists with system integrated information $\varphi_s \approx 0.172$, and the experience it specifies has the $\Phi$-structure above, of quantity $\Phi \approx 1.563$."
  - In the paper-to-code map table (lines 58-76): the example reference in the header sentence becomes `analysis = pyphi.analyze(substrate, (0, 1, 1), subset=(0, 1))`; row $U$'s example becomes `pyphi.examples.iit4_2023_fig1a_substrate()`; update the $\mathit{ii}$ row to "`analysis.sia.system_state` (per-direction `intrinsic_information`)"; update the $\varphi_s$ row's Quantity column to "system integrated information, $\min(\varphi_c, \varphi_e)$, capped by $\mathit{ii}(s)$ under the 2026 default (see {doc}`intrinsic-information`)"; update the "complex" row's In-PyPhi column to "`Substrate.complexes`; the analyzed candidate is `analysis.system`".

- [ ] **Step 6: Verify all five pages execute with the expected values.** Run this snippet and compare against the reference table:

```bash
uv run python - <<'EOF'
import pyphi
pyphi.config.progress_bars = False
substrate = pyphi.examples.iit4_2023_fig1a_substrate()
a = pyphi.analyze(substrate, (0, 1, 1), subset=(0, 1))
assert round(a.phi, 4) == 0.1719
assert round(float(a.sia.cause.phi), 4) == 0.2446
assert round(float(a.sia.effect.phi), 4) == 0.1719
assert (len(a.ces.distinctions), len(a.ces.relations)) == (3, 7)
assert round(float(a.ces.big_phi), 3) == 1.563
assert [tuple(c.node_indices) for c in substrate.complexes((0, 1, 1))] == [(2,), (0, 1)]
print("theory-page ground truth OK")
EOF
```

Expected: `theory-page ground truth OK`.

- [ ] **Step 7: Commit:**

```bash
git add docs/theory/overview.md docs/theory/substrate-and-system.md docs/theory/system-integration.md docs/theory/distinctions-and-relations.md docs/theory/phi-structure.md
git commit -m "Move the theory section's running example to the paper's Fig 1A network

basic_substrate was misattributed as the IIT 4.0 paper's example and
computes phi_s = 0 under the 2026 default; the pipeline pages now walk
the genuine paper example (complex aB), so every quantity shown is
paper-corroborated and nonzero. Also corrects the exclusion section:
the complex search is Substrate.complexes, not pyphi.analyze."
```

---

### Task 5: The intrinsic-information theory centerpiece (the determinism lesson)

**Files:**
- Create: `docs/theory/intrinsic-information.md`
- Modify: `docs/theory/index.md` (toctree: insert `intrinsic-information` after `system-integration`)

This page is the new conceptual centerpiece: what the 2026 cap is, and why deterministic systems have φ_s = 0 — with `xor` as the specimen, repurposed from headline example. (`basic` need not appear here; one clean illustration suffices, and `basic` already appears in the formalism-versions comparison.)

- [ ] **Step 1: Verify the theoretical claims against the paper.** Before writing prose, open `papers/2026__mayner-et-al__intrinsic-cause-effect-power.pdf` and confirm: (a) the equation number for the cap φ_s = min{φ_c, φ_e, ii(s)} (the codebase cites it as Eq. 23 — `pyphi/formalism/iit4/__init__.py:_apply_ii_cap`); (b) the definition of ii(s) as the per-direction minimum of *specification* (intrinsic information of the specified state) and *differentiation* (the code implements `ii(s) = min_d min(i_spec_d, i_diff_d)`); (c) the paper's own statement of the deterministic ⇒ zero-differentiation property, and its equation/section number. Cite exactly what the paper says; do not paraphrase the formalism from memory. If the paper's terminology differs from the draft below, follow the paper.

- [ ] **Step 2: Write the page:**

````markdown
---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# The intrinsic-information cap

PyPhi's default formalism, IIT 4.0 (2026), refines the system integrated
information of the preceding pages in one way: beyond *specifying* an
irreducible cause–effect state, a system must provide itself with a
*repertoire of alternatives* — intrinsic **differentiation** (Mayner,
Marshall, Tononi 2026). The two requirements trade off, and the system
integrated information is capped by their minimum, the system's **intrinsic
information** $\mathit{ii}(s)$:

$$ \varphi_s = \min\{\varphi_c,\ \varphi_e,\ \mathit{ii}(s)\}. $$

<!-- Implementer: cite the verified equation number from Step 1 here, e.g.
"(Mayner et al., 2026, Eq. 23)", and state ii(s)'s definition exactly as the
paper gives it. -->

This page shows the cap in action, and its most consequential theorem:
**a deterministic system has $\varphi_s = 0$.**

## Determinism means zero differentiation

A deterministic system in a state pins its cause and effect down completely —
maximal specification. But differentiation asks the opposite question: what
repertoire of alternatives does the system furnish itself? A deterministic
transition offers exactly one, so its intrinsic differentiation is zero, the
cap binds at zero, and $\varphi_s = 0$ — however tightly the units are wired
together.

The classic three-XOR network makes this concrete. Under the *uncapped* 2023
formalism it is the textbook integrated system:

```{code-cell} python
import pyphi

pyphi.config.progress_bars = False

xor = pyphi.examples.xor_substrate()
pyphi.analyze(xor, (0, 0, 0), formalism="IIT_4_0_2023").phi
```

Under the default it computes zero:

```{code-cell} python
analysis = pyphi.analyze(xor, (0, 0, 0))
analysis.phi
```

The analysis records exactly where the zero comes from. Both directions carry
substantial integration and a fully specified state — and zero
differentiation:

```{code-cell} python
sia = analysis.sia
print("φ_c =", float(sia.cause.phi), "  φ_e =", float(sia.effect.phi))
print("differentiation:",
      {str(d): float(v) for d, v in sia.intrinsic_differentiation.items()})
```

$\varphi_c = 1.5$ and $\varphi_e = 3.0$ survive from the 2023 analysis; the
differentiation term is $0$ in both directions, so the minimum — and
$\varphi_s$ — is $0$.

## This is a theorem, not a bug

Almost the entire classic IIT teaching repertoire is deterministic — the XOR
network above, the `basic` OR/COPY/XOR gates, the cellular-automaton rules —
so under the default formalism *all of them* compute $\varphi_s = 0$. If you
port an analysis from the literature or from an earlier PyPhi and see zero
where a paper printed $1.5$: nothing is broken. The system is deterministic,
the default formalism is the 2026 refinement, and the published number is the
uncapped 2023 quantity — still available by pinning that formalism, as in the
first cell above or with `pyphi.config.override(**pyphi.conf.presets.iit4_2023)`.

Any indeterminism restores a repertoire of alternatives. The paper's own
worked example — the {doc}`Fig 1A logistic network <../tutorials/worked-example>`
threaded through this section — is probabilistic, which is why its published
values are identical under 2023 and 2026. Even slight noise suffices: the
three-unit noisy grid computes a small but positive value under the default.

```{code-cell} python
pyphi.analyze(pyphi.examples.grid3_substrate(), (0, 0, 0)).phi
```

## What the cap does and does not change

The cap applies to the *system-level* quantity only. Mechanism-level
quantities — distinctions, relations, and their summed structure $\Phi$ — are
computed exactly as in 2023, so the XOR network's cause-effect structure is
as rich as ever:

```{code-cell} python
ces = analysis.ces
(len(ces.distinctions), len(ces.relations), float(ces.big_phi))
```

What changes is the system's claim to existence: with $\varphi_s = 0$, the
candidate is not a complex, so under the 2026 formalism this structure is not
specified by any existing whole. The structure remains available for analysis
and comparison; the theory's verdict on the deterministic system itself is
$\varphi_s = 0$.

The minimum information partition is also unaffected: the MIP is selected on
the *uncapped* normalized integrated information, exactly as in 2023, and the
cap is applied once to the selected partition's value. Margins and
tie-breaking therefore behave identically across the two formalisms (see
{doc}`Control tie-breaking <../howto/tie-breaking>`).

For choosing between formalism versions — and reproducing published 2023 or
IIT 3.0 numbers — see {doc}`formalism-versions`.
````

Remove the HTML comment after completing the Step 1 citation check; the final page must carry the verified citation, not the placeholder comment.

- [ ] **Step 3: Add to the toctree.** In `docs/theory/index.md`, insert `intrinsic-information` on its own line directly after `system-integration`.

- [ ] **Step 4: Verify the page's cells:**

```bash
uv run python - <<'EOF'
import pyphi
pyphi.config.progress_bars = False
assert pyphi.analyze(pyphi.examples.xor_substrate(), (0,0,0), formalism="IIT_4_0_2023").phi == 1.5
a = pyphi.analyze(pyphi.examples.xor_substrate(), (0,0,0))
assert a.phi == 0.0
assert all(float(v) == 0.0 for v in a.sia.intrinsic_differentiation.values())
assert (float(a.sia.cause.phi), float(a.sia.effect.phi)) == (1.5, 3.0)
assert (len(a.ces.distinctions), len(a.ces.relations), float(a.ces.big_phi)) == (4, 15, 9.5)
assert round(pyphi.analyze(pyphi.examples.grid3_substrate(), (0,0,0)).phi, 4) == 0.0247
print("intrinsic-information page ground truth OK")
EOF
```

- [ ] **Step 5: Commit:**

```bash
git add docs/theory/intrinsic-information.md docs/theory/index.md
git commit -m "Add the intrinsic-information theory page: the 2026 cap, taught

The determinism => phi_s = 0 property of the default formalism becomes
an explicit lesson (xor as specimen, cap mechanics shown on the SIA)
instead of a hidden gotcha."
```

---

### Task 6: Formalism versions, default-statement sweep, and release notes

**Files:**
- Modify: `docs/theory/formalism-versions.md` (rework)
- Modify: `docs/howto/configure.md:109` (stale "the default formalism" label)
- Modify: `docs/migration/migration-2.0.md` (add a determinism note under "changed defaults")
- Modify: `docs/index.md:5-11` (formalism citation)
- Modify: `docs/whats-new-in-2.0.md` (add the default-formalism section)

- [ ] **Step 1: Rework `docs/theory/formalism-versions.md`.** Full replacement of the body (keep frontmatter):

````markdown
# Formalism versions

The preceding pages describe IIT 4.0 in its 2026 refinement — PyPhi's default
formalism. PyPhi also implements the 2023 formulation and IIT 3.0, and a
separate analysis of actual causation. A **formalism** is the set of rules
that turns a system into results; which one applies is a matter of
configuration.

The cleanest way to select a formalism is the `formalism` argument to
`pyphi.analyze`, which sets the compatible measures for you. The same
substrate and state yield a different system integrated information under
each formalism, because each defines that quantity differently — here on the
three-XOR network:

```{code-cell} python
import pyphi

pyphi.config.progress_bars = False

substrate = pyphi.examples.xor_substrate()

{version: round(float(pyphi.analyze(substrate, (0, 0, 0), formalism=version).phi), 4)
 for version in ("IIT_3_0", "IIT_4_0_2023", "IIT_4_0_2026")}
```

## IIT 4.0 (2026)

`"IIT_4_0_2026"` — the default. It refines the account of system integrated
information to require not only that a system *specify* a cause–effect state
but also that it provide itself with a *repertoire of alternatives* —
intrinsic **differentiation** — capping $\varphi_s$ by the system's intrinsic
information (Mayner, Marshall, Tononi 2026). The XOR network above is the
consequence in miniature: it is deterministic, so it furnishes no
alternatives, and its $\varphi_s$ is $0$ — where the uncapped 2023 value is
$1.5$. See {doc}`intrinsic-information` for the full lesson.

## IIT 4.0 (2023)

`"IIT_4_0_2023"` — the formulation of Albantakis et al. (2023), identical to
the default except *uncapped*: $\varphi_s = \min(\varphi_c, \varphi_e)$, with
no differentiation requirement. Distinctions, relations, and $\Phi$ are the
same under both. Choose it to reproduce numbers published against the 2023
paper — most classic worked examples in the literature are deterministic, and
their published nonzero $\varphi_s$ values are 2023 quantities.

## IIT 3.0 (2014)

`"IIT_3_0"` is the earlier formalism (Oizumi et al., 2014). It computes a
cause–effect structure of *concepts* rather than distinctions and relations,
and its integrated information is defined differently — hence the third value
above. It remains available for reproducing older results and for comparison;
see the [IIT 3.0 overview](iit-3.0.md).

## Actual causation

Actual causation answers a different question — not *how integrated is this
system*, but *which past events actually caused a given present event, and
which effects will it actually cause* (Albantakis et al., 2019). It operates
on a `Transition` (a substrate observed across two time steps) rather than a
`System`, and lives in `pyphi.actual`. It is its own formalism, unaffected by
the IIT versions above (in particular, the 2026 cap does not apply to it),
and is documented with the tutorials.

## Under the hood

`formalism=` sets `pyphi.config.formalism.iit.version` together with the
distance measures each version requires. You can also set the version through
configuration directly, but the measures must be made compatible with it; the
`formalism` argument — or applying a whole preset from `pyphi.conf.presets`
(`iit3`, `iit4_2023`, `iit4_2026`) with `config.override` — is the reliable
path. The three IIT versions correspond to the namespaces `pyphi.iit3`,
`pyphi.iit4_2023`, and `pyphi.iit4_2026`.
````

Expected regenerated dict: `{'IIT_3_0': 1.875, 'IIT_4_0_2023': 1.5, 'IIT_4_0_2026': 0.0}`.

- [ ] **Step 2: Fix `docs/howto/configure.md` line 109.** Change the preset list to:

```markdown
- `iit3` — IIT 3.0 (Oizumi et al. 2014)
- `iit4_2023` — IIT 4.0 (Albantakis et al. 2023), uncapped
- `iit4_2026` — IIT 4.0 with the intrinsic-information cap (Mayner,
  Marshall, Tononi 2026), the default formalism
```

- [ ] **Step 3: Add the migration note.** In `docs/migration/migration-2.0.md`, find the "changed defaults" item (line ~188: "The default formalism changed from IIT 3.0 (1.x) to IIT 4.0 (2026)…") and append after it:

```markdown
A practical consequence: **deterministic networks compute φ_s = 0 under the
2026 default.** The classic examples (`xor`, `basic`, the cellular-automaton
rules) are all deterministic, so analyses ported from 1.x or from the
literature will show 0 where papers print nonzero values. This is the 2026
formalism's intrinsic-information cap, not a regression; pin
`formalism="IIT_4_0_2023"` to reproduce published 2023 numbers. See
{doc}`../theory/intrinsic-information`.
```

- [ ] **Step 4: Update the root `docs/index.md` citation block.** After the Albantakis et al. (2023) citation (lines 5-11), add:

```markdown
The default formalism includes the 2026 refinement of system integrated
information, described in:

> Mayner WGP, Marshall W, Tononi G. (2026). <!-- Implementer: copy the exact
> title and venue from papers/2026__mayner-et-al__intrinsic-cause-effect-power.pdf -->
```

Replace the comment with the verified citation from the PDF's title page; do not invent the venue.

- [ ] **Step 5: Add the what's-new section.** In `docs/whats-new-in-2.0.md`, insert before the "Query the relational structure" section (line 19):

```markdown
## IIT 4.0 (2026) is the default formalism

PyPhi 2.0 computes the 2026 refinement of IIT 4.0 by default: system
integrated information is capped by the system's intrinsic information
(Mayner, Marshall, Tononi 2026), so a system must furnish itself a repertoire
of alternatives, not merely specify one state. One consequence to know before
comparing against published numbers: **deterministic systems compute
φ_s = 0** under the default. The 2023 formulation and IIT 3.0 remain fully
supported — `pyphi.analyze(..., formalism="IIT_4_0_2023")` or the presets in
`pyphi.conf.presets` reproduce published values exactly. See the theory page
[The intrinsic-information cap](theory/intrinsic-information.md).
```

Also delete the "Formalism objects: …" line from the HTML comment block (lines 8-9) — this section now covers it.

- [ ] **Step 6: Verify the comparison cell values:**

Run: `uv run python -c "import pyphi; pyphi.config.progress_bars=False; print({v: round(float(pyphi.analyze(pyphi.examples.xor_substrate(), (0,0,0), formalism=v).phi), 4) for v in ('IIT_3_0','IIT_4_0_2023','IIT_4_0_2026')})"`
Expected: `{'IIT_3_0': 1.875, 'IIT_4_0_2023': 1.5, 'IIT_4_0_2026': 0.0}`

- [ ] **Step 7: Commit:**

```bash
git add docs/theory/formalism-versions.md docs/howto/configure.md docs/migration/migration-2.0.md docs/index.md docs/whats-new-in-2.0.md
git commit -m "Teach the formalism-version choice 2026-first

formalism-versions.md leads with the default (capped) formalism and
makes the determinism contrast concrete on xor (0 vs 1.5 vs 1.875);
stale 'iit4_2023 is the default' claims fixed in configure.md; the
determinism consequence added to the migration guide, what's-new, and
the landing-page citation."
```

---

### Task 7: How-to guides — example swaps onto Fig 1A

**Files:**
- Modify: `docs/howto/save-load.md` (lines 30-40)
- Modify: `docs/howto/cache.md` (lines 55-56)
- Modify: `docs/howto/export.md` (lines 27-34)
- Modify: `docs/howto/parallel.md` (lines 75-91)
- Modify: `docs/howto/sweep.md` (restructure, lines 28-56 + 79-92)

These pages compute under the default on deterministic examples and *display* φ_s = 0 (or, for sweep, whole tables of zeros). Swap to the promoted example; none of their teaching depends on the old examples except where noted.

- [ ] **Step 1: `save-load.md`.** Replace the compute cell (lines 30-40):

````markdown
```{code-cell} python
from pyphi import examples

system = examples.iit4_2023_fig1a_system()
analysis = pyphi.analyze(system.substrate, system.state)

ces = analysis.ces  # cause-effect structure (distinctions + relations)
sia = analysis.sia  # system-irreducibility analysis (Φ_s)

print(f"Φ_s = {float(sia.phi):.4f}   |   {len(ces.distinctions)} distinctions")
```
````

Expected regenerated line: `Φ_s = 0.1339   |   6 distinctions`. The rest of the page operates on `ces`/`sia`/`system` and needs no change (it displays no φ).

- [ ] **Step 2: `cache.md`.** Replace lines 55-56:

```python
substrate = pyphi.examples.iit4_2023_fig1a_substrate()
system = pyphi.System(substrate, (0, 1, 1))
```

Line 90's `round(uncached.phi, 4)` then regenerates to `0.1339` (nonzero — previously `0.0`). No other change.

- [ ] **Step 3: `export.md`.** Replace lines 32-33:

```python
substrate = examples.iit4_2023_fig1a_substrate()
state = (0, 1, 1)
```

Change line 27's sentence to "We use the IIT 4.0 paper's three-unit Fig 1A substrate throughout." The `analysis.to_pandas()` cell then shows a nonzero `phi` row. The DBN/xarray sections work on any substrate; the `ds["unit_0"].sel(u0=0, u1=0, u2=0)` cell (line 82) remains valid (binary units, coordinates 0/1).

- [ ] **Step 4: `parallel.md`.** Replace lines 76-77:

```python
substrate = pyphi.examples.iit4_2023_fig1a_substrate()
state = (0, 1, 1)
```

The parallel-vs-sequential equality demo then compares `0.133873` to `0.133873` (previously `0.0` vs `0.0` — a vacuous-looking identity). No other change.

- [ ] **Step 5: Restructure `sweep.md`.** Under the default, the current first sweep (`basic`, all states) renders a table whose `phi` column is all zeros. Restructure:
  1. **First sweep** (lines 28-38): switch to the Fig 1A substrate —

````markdown
```{code-cell} python
substrate = pyphi.examples.iit4_2023_fig1a_substrate()

result = pyphi.sweep(substrate, states="all")
result.df.round(6)
```
````

  The table now shows eight rows with state-varying nonzero φ.
  2. **Skipped-states demo** (lines 48-56): Fig 1A is probabilistic, so every state is reachable and `result.skipped` is empty. Keep the teaching by switching specimen — replace the paragraph and cell with:

````markdown
States that cannot be reached from any previous state have no defined
repertoire, so their $\Phi$ is undefined. When you enumerate an axis with
`"all"`, those cells are dropped rather than raised, and the dropped cells
are recorded on the result. Our probabilistic substrate reaches every state,
so nothing was dropped here:

```{code-cell} python
result.skipped
```

A deterministic substrate shows the mechanism — two of the eight states of
the three-gate `basic` network are unreachable:

```{code-cell} python
pyphi.sweep(pyphi.examples.basic_substrate(), states="all").skipped
```
````

  Expected regenerated output for the second cell: `[('IIT_4_0_2026', (0, 1, 2), (0, 1, 0)), ('IIT_4_0_2026', (0, 1, 2), (0, 1, 1))]`.
  3. **Subsystem sweep** (line 79): `pyphi.sweep(substrate, states=(0, 1, 1), subsets="all").df.round(6)` (Fig 1A substrate, paper state — rows now show the 0.0396/0.1719/0.1339/0.2122 spread).
  4. **Formalism sweep** (lines 84-93): keep `basic` deliberately — it is the determinism contrast. Replace the cell and add a note:

````markdown
```{code-cell} python
pyphi.sweep(
    pyphi.examples.basic_substrate(),
    states=(1, 0, 0),
    formalisms=["IIT_4_0_2023", "IIT_4_0_2026"],
).df.round(6)
```

The deterministic `basic` network is chosen deliberately: the two rows differ
(φ = 0.415 under 2023, 0 under 2026) because the 2026 formalism caps
deterministic systems at zero — see
{doc}`../theory/intrinsic-information`.
````

  5. **Remaining cells** (lines 100-148: `compute="ces"`, tied-cells, parallel, seed): they reference `substrate` and `result` — now Fig 1A; they regenerate. In the tied-cells section, the `result.df[result.df.effectively_tied...]` filter may legitimately return an empty frame for Fig 1A; that is fine (the section's point is the one-liner), but change its lead-in sentence to "…find the cells whose selections were effectively tied — none, for this asymmetric substrate:" **only if** the regenerated table is empty; if rows survive, leave the sentence as is. (Check during Step 6.)

- [ ] **Step 6: Verify all five pages.** Run each page's cells via a scratch execution; the fastest complete check is the docs build in Task 10, but verify the two key sweep facts now:

```bash
uv run python - <<'EOF'
import pyphi
pyphi.config.progress_bars = False
r = pyphi.sweep(pyphi.examples.iit4_2023_fig1a_substrate(), states="all")
assert len(r.df) == 8 and (r.df.phi > 0).any() and not r.skipped
b = pyphi.sweep(pyphi.examples.basic_substrate(), states="all")
assert len(b.skipped) == 2
tied = r.df[r.df.effectively_tied.astype(bool)]
print("fig1a sweep OK; tied rows:", len(tied))
EOF
```

Note the printed tied-row count and finalize the Step 5.5 sentence accordingly.

- [ ] **Step 7: Commit:**

```bash
git add docs/howto/save-load.md docs/howto/cache.md docs/howto/export.md docs/howto/parallel.md docs/howto/sweep.md
git commit -m "Swap how-to examples onto Fig 1A so displayed phi is nonzero

save-load, cache, export, and parallel showed phi_s = 0 under the 2026
default (deterministic specimens); sweep's first table was all zeros.
basic is retained where it teaches (unreachable states; the 2023/2026
determinism contrast, now annotated)."
```

---

### Task 8: How-to and tutorial verification pass — pins, notes, and cross-references

**Files:**
- Modify: `docs/howto/landscape.md` (lines 33-49: reference the promoted example)
- Modify: `docs/tutorials/recursive-exclusion.md` (line ~74: one-line pin note)
- Modify: `docs/tutorials/macro.md` (line ~38: extend the existing pin note)
- Modify: `docs/howto/grain-search.md` (line ~54: extend the existing pin note)
- Modify: `docs/tutorials/actual-causation.md` (line ~41: one clause)
- Modify: `docs/howto/visualize.md` (line ~50: one-line note)
- Verify only (no planned edits): `docs/howto/tie-breaking.md`, `docs/howto/query-relations.md`, `docs/howto/mcp-server.md`, `docs/howto/configure.md` (rest), `docs/theory/iit-3.0.md`, `docs/theory/computational-complexity.md`, `docs/theory/macro-units.md`, `docs/theory/conditional-independence.md`, `docs/tutorials/causal-reductionism.md`, `docs/migration/from-substrate-modeler.md`
- Regenerate pairs: `docs/tutorials/{recursive-exclusion,macro,actual-causation}.ipynb`

Each edit below is the complete change for that file; the verify-only files were read during planning and hold under the default for the stated reason — the implementer re-confirms each via the Task 10 build.

- [ ] **Step 1: `landscape.md`** — verified safe under the default (all three prose claims re-checked; see reference table). One improvement: tie it to the promoted example. Replace the sentence at lines 34-37 ("Here we use the three-unit substrate of Figure 1A … as the axis:") with: "Here we use the three-unit substrate of Figure 1A of the IIT 4.0 paper — the same network as {func}`pyphi.examples.iit4_2023_fig1a_substrate` — taking its A→B coupling (published value 0.7) as the axis:". Keep the inline `weights` construction (the page needs the raw weights for `weight_axis`).

- [ ] **Step 2: `recursive-exclusion.md`** — computes `complexes` under an explicit `presets.iit4_2023` pin (line 77) but never says why. Insert, immediately before the "## Finding the complexes" heading:

```markdown
The φₛ values in this tutorial are computed under the 2023 formalism, pinned
explicitly below: the chain's φₛ ladder was constructed for the uncapped
measure, and the exclusion mechanics being demonstrated are identical under
both formalisms.
```

Then sync: `uv run jupytext --sync docs/tutorials/recursive-exclusion.md`.

- [ ] **Step 3: `macro.md`** — every compute cell already pinned to `presets.iit4_2023`. Extend the existing note (lines 38-40, "Throughout we use the configuration preset that reproduces the paper's settings:") with one sentence: "(The pin also matters for the numbers: several specimens here are deterministic or near-deterministic, and the 2026 default's intrinsic-information cap would zero them — see {doc}`../theory/intrinsic-information`.)" Sync the pair.

- [ ] **Step 4: `grain-search.md`** — same situation; extend line 54-55 ("Every computation below runs under the configuration preset that reproduces the settings of Marshall et al. (2024).") with the same parenthetical sentence as Step 3. (Unpaired page; no sync.)

- [ ] **Step 5: `actual-causation.md`** — AC is its own formalism; the cap does not apply (verified: the page computes only α quantities via `pyphi.actual`). Extend the sentence at lines 41-43 to close the question explicitly: "…configured by default to reproduce the 2019 paper. No special configuration is needed — in particular, the IIT formalism version (and the 2026 intrinsic-information cap) does not affect actual causation." Sync the pair.

- [ ] **Step 6: `visualize.md`** — plots `xor_system().ces()` under the default; all plotted quantities (φ_d, φ_r, Σφ_r) are mechanism-level and unchanged by the cap, so the page is compliant. Add one orienting line after the `ces = examples.xor_system().ces()` cell (line 51): "(The XOR system's *system-level* φₛ is 0 under the default formalism — see {doc}`../theory/intrinsic-information` — but its cause-effect structure is rich, which is what the plots show.)"

- [ ] **Step 7: Verify-only files — confirm each holds, fix only if the check fails.**
  - `tie-breaking.md`: displays no system-level φ (mechanism-level φ = 1.0 demos; margins on `basic` still populate under the default — verified: `partition_margin` 0.0755, state margins 3.0/3.0, `grid3` partition tie fires). Run its margin cells if in doubt.
  - `query-relations.md`: uses `grid3` (probabilistic, nonzero) under the default; verified `pyphi.analyze` on grid3 computes a full CES.
  - `iit-3.0.md`: pins `formalism="IIT_3_0"` explicitly; the shown value regenerates.
  - `computational-complexity.md`: compute cells pin `pyphi.iit4_2023` / `presets.iit4_2023`; the counting bounds are partition-scheme facts, not φ values. The line-18 claim "IIT 4.0 is the formalism PyPhi computes by default" remains true.
  - `macro-units.md`, `conditional-independence.md`: no computed φ.
  - `causal-reductionism.md`: fully pinned to its own frog formalism (IIT 3.0 + AC).
  - `mcp-server.md`, `configure.md` (rest), `migration/from-substrate-modeler.md`: no default-computed φ displays.

- [ ] **Step 8: Sync pairs and commit:**

```bash
uv run jupytext --sync docs/tutorials/recursive-exclusion.md docs/tutorials/macro.md docs/tutorials/actual-causation.md
git add docs/howto/landscape.md docs/howto/visualize.md docs/howto/grain-search.md docs/tutorials/recursive-exclusion.md docs/tutorials/recursive-exclusion.ipynb docs/tutorials/macro.md docs/tutorials/macro.ipynb docs/tutorials/actual-causation.md docs/tutorials/actual-causation.ipynb
git commit -m "Annotate formalism pins and cap-adjacent examples across docs

Every page that pins iit4_2023 now says why in one line; the AC
tutorial states the cap does not apply; visualize notes xor's zero
system phi is expected; landscape cross-references the promoted
Fig 1A example."
```

---

### Task 9: Pin the IIT 4.0 demo notebook to the 2023 formalism

**Files:**
- Modify: `docs/examples/IIT_4.0_demo.ipynb` (cell 3, the setup cell)
- Modify: `docs/tutorials/iit-4.0-demo.md` (one sentence)

The notebook is the 2023 paper's supplement and is excluded from the Sphinx build — but it is a download-and-run artifact, and it currently runs under the ambient default. Its Part 1 system (Fig 8C) is **deterministic**: a user running it today gets φ_s = 0 where the paper's numbers are nonzero. Pin the 2023 formalism in the setup cell, *before* the measure-resolution lines (which read the active config).

- [ ] **Step 1: Edit cell 3.** Use `NotebookEdit` (or a `json`-editing script) to insert into the cell-3 source, after `pyphi.config.shortcircuit_sia = False` and before the `specification_measure = ...` block:

```python
# This notebook is the supplement to the IIT 4.0 (2023) paper, so it pins the
# 2023 formalism. PyPhi's default is the 2026 refinement, whose
# intrinsic-information cap gives deterministic systems -- like the Part 1
# network below -- phi_s = 0; see the documentation's theory section.
import warnings

from pyphi.conf import presets

with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # advisory config-change notices
    pyphi.config.iit = presets.iit4_2023["iit"]
```

(`pyphi.config.iit = presets.iit4_2023["iit"]` applies the whole `IITConfig` globally — verified working; the 2023 preset contains only the `iit` key. The existing measure-resolution lines below then resolve GID for both measures, matching the paper.)

- [ ] **Step 2: Execute the notebook end to end:**

Run: `uv run --all-extras --group docs jupyter execute docs/examples/IIT_4.0_demo.ipynb`
Expected: exit 0 (95 cells; a few minutes — Part 2 enumerates relations on a 3-unit system). If any cell errors, diagnose before proceeding; do not commit a notebook that no longer runs.

- [ ] **Step 3: Confirm no outputs were committed:**

Run: `uv run python -c "import json; nb=json.load(open('docs/examples/IIT_4.0_demo.ipynb')); print(sum(1 for c in nb['cells'] if c.get('outputs')))"`
Expected: `0` (jupyter execute must not have written outputs back; if it did, strip them — the file convention is outputs-free).

- [ ] **Step 4: Note the pin in the wrapper page.** In `docs/tutorials/iit-4.0-demo.md`, after the paragraph ending "…reproduces the paper's numbers on the paper's own example systems." (line 22), add: "The notebook pins the IIT 4.0 (2023) formalism it documents; under PyPhi's default 2026 formalism, the deterministic Part 1 system would compute $\varphi_s = 0$ (see {doc}`../theory/intrinsic-information`)."

- [ ] **Step 5: Commit:**

```bash
git add docs/examples/IIT_4.0_demo.ipynb docs/tutorials/iit-4.0-demo.md
git commit -m "Pin the IIT 4.0 demo notebook to the 2023 formalism it documents

The paper-supplement notebook ran under the ambient default; under the
2026 default its deterministic Part 1 system computes phi_s = 0 instead
of the paper's numbers. The setup cell now applies presets.iit4_2023
before resolving measures, with a note pointing at the theory lesson."
```

---

### Task 10: Full verification, changelog, and roadmap

**Files:**
- Create: `changelog.d/docs-overhaul-2026-default.doc.md`
- Modify: `ROADMAP.md` (Status Dashboard row)

- [ ] **Step 1: Full test suite** (no path argument — CI parity, includes the doctest sweep):

Run: `uv run pytest -m "not slow"` in foreground, and `uv run pytest -m slow` in background (`run_in_background`).
Expected: both green. The only code change is Task 1's example promotion; any failure outside `test_paper_reproduction.py` / `examples` indicates an unintended change — stop and diagnose.

- [ ] **Step 2: Docs build under the 2026 default:**

Run: `just docs`
Expected: exit 0 with `-W` (warnings are errors; `nb_execution_raise_on_error` executes every MyST page under the shipping default). This is the binding verification that every executed cell in the reworked docs computes cleanly.

- [ ] **Step 3: Spot-check the built pages.** Open `docs/_build/html/getting-started/first-computation.html` and `docs/_build/html/tutorials/worked-example.html` — confirm nonzero Fig 1A numbers (0.1339 / 0.1719, the two complexes); open `docs/_build/html/theory/intrinsic-information.html` — confirm the xor cells show `1.5` (2023), `0.0` (default), and the zero differentiation dict. Confirm `tutorials/cause-effect-structure.html` is gone and no page links to it (the build's `-W` would have failed on a dangling `{doc}` reference).

- [ ] **Step 4: Stale-φ grep pass.** No committed doc may carry a default-computed φ number that the build no longer produces:

```bash
# Old basic/xor headline numbers must survive only in pinned/contrast contexts:
grep -rn "0\.208\|1\.857\|1\.8568\|0\.415" docs/getting-started docs/tutorials docs/theory docs/howto docs/index.md docs/whats-new-in-2.0.md
# Expected hits: ONLY formalism-versions.md / intrinsic-information.md /
# sweep.md / migration-2.0.md lines that explicitly name the 2023 formalism
# or IIT 3.0. Any other hit is a missed rework — fix it.

# The keystone numbers must be present where the spec requires them:
grep -rln "0\.17\|0\.13" docs/getting-started/first-computation.md docs/tutorials/worked-example.md docs/theory/overview.md
# Expected: all three files hit.

# No doc still headlines the old examples as *the* default-formalism demo:
grep -rn "basic_substrate\|xor_substrate\|rule110" docs/getting-started docs/theory docs/howto docs/tutorials --include="*.md"
# Expected hits: only (a) explicitly pinned/contrast contexts
# (formalism-versions, intrinsic-information, sweep's two basic cells,
# iit-3.0, tie-breaking's mechanism-level demos, visualize's annotated xor,
# conditional-independence's cond_*_tpm) — read each hit and confirm.
```

- [ ] **Step 5: Changelog fragment for the docs overhaul:**

```bash
echo 'Documentation overhauled for the IIT 4.0 (2026) default: getting started and the theory pipeline now run on the IIT 4.0 paper'"'"'s Fig 1A network (nonzero, paper-matching φ); one worked tutorial follows the paper'"'"'s Figs 1→2→4; a new theory page teaches the intrinsic-information cap and why deterministic systems compute φ_s = 0; the IIT 4.0 demo notebook pins the 2023 formalism it documents.' > changelog.d/docs-overhaul-2026-default.doc.md
```

- [ ] **Step 6: ROADMAP row.** In `ROADMAP.md`'s Status Dashboard, the "Documentation overhaul (2.0 / IIT 4.0)" row (currently "✅ landed") predates the default flip. Append to its description: "**2026-default rework (2026-07-12):** teaching moved onto the promoted `iit4_2023_fig1a` example (nonzero under the shipping default); determinism ⇒ φ_s = 0 taught explicitly (`docs/theory/intrinsic-information.md`); demo notebook pinned to 2023. Spec/plan: `docs/superpowers/{specs,plans}/2026-07-12-docs-overhaul-2026-default*`." If a dedicated row for the default flip exists by execution time, update that row instead — verify against the live file, not this plan.

- [ ] **Step 7: Final commit:**

```bash
git add changelog.d/docs-overhaul-2026-default.doc.md ROADMAP.md
git commit -m "Record the 2026-default docs overhaul in changelog and roadmap"
```

Do **not** push; pushing needs its own explicit approval from the maintainer.

---

## Self-Review

**Spec coverage.**
- Keystone example promoted, test re-pointed, N1 values re-verified → Task 1. ✓
- Teaching spine 1 (getting started, nonzero first result + complex aha) → Task 2. ✓
- Spine 2 (one Fig 1→2→4 arc replacing `cause-effect-structure` + `worked-example`; split decision: one page at the surviving URL) → Task 3. ✓
- Spine 3 (intrinsic-information centerpiece; xor as specimen; classics repurposed, not all retained — `rule110`/`rule154`/`fig4`/`fig5*`/`pqr`/`residue` simply no longer appear in user docs, which the spec permits) → Task 5. ✓
- Spine 4 (formalism versions, 2026-first, xor contrast 0 vs 1.5) → Task 6. ✓
- Spine 5 (supporting material under the per-doc principle: macro, AC, how-tos) → Tasks 7, 8. ✓
- Affected-docs inventory: every file in the spec's list has a task (first-computation T2; cause-effect-structure/worked-example T3; actual-causation T8; IIT_4.0_demo T9; theory/{overview, phi-structure, system-integration, distinctions-and-relations} T4, macro-units verified T8; howto/{landscape, save-load, export, tie-breaking, parallel, cache} T7/T8; whats-new T6). Spec's "already safe" list re-verified: formalism-versions was **not** safe (stale "2023 is the default") — reworked in T6; iit-3.0, macro, grain-search verified with notes in T8. Files the spec missed but the live inventory surfaced: sweep.md (all-zero tables, T7), query-relations.md + visualize.md (new since spec, T8), configure.md (stale default claim, T6), migration-2.0.md (T6), root index.md (T6), recursive-exclusion.md (T8). ✓
- Verification section of the spec: pinned N1 values (T1), `uv run pytest` green (T10.1), `just docs` + spot-checks (T10.2-3), stale-φ grep (T10.4). ✓
- Non-goals respected: no analytical-relations changes, no ii-gate, no `docs/superpowers/` edits beyond this plan file, no computation-logic changes beyond the example.

**Placeholder scan.** Two deliberate verify-against-source steps remain (Task 5 Step 1: equation numbers from the 2026 paper; Task 6 Step 4: exact citation from the PDF) — these are verification requirements with explicit instructions and marked insertion points, not deferred decisions; the repo's docstring/citation rules forbid citing paper details from memory, so the plan must not hardcode them. Task 7 Step 5.5 has an explicit either/or resolved by a concrete check in Step 6. No "TBD"/"handle appropriately" items.

**Type/name consistency.** `iit4_2023_fig1a_substrate()` / `iit4_2023_fig1a_system()` used identically in Tasks 1-8; `pyphi.analyze(substrate, (0, 1, 1), subset=(0, 1))` is the canonical aB call everywhere (Tasks 3, 4); `docs/theory/intrinsic-information.md` is the link target used in Tasks 2, 4, 5, 6, 7, 8, 9. Reference values in all tasks trace to the single ground-truth table at the top, computed with the exact promoted builder on 2026-07-12.
