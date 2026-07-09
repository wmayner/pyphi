# Causal-reductionism frog tutorial and example builders — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development
> or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `frog_example` print-demo with `frog_substrate`/
`frog_transition` builders and an executable tutorial reproducing Grasso et al.
(2021), *Causal reductionism and causal structures*.

**Architecture:** Pure example builders in `pyphi/examples.py` (no config side
effects); one executable MyST tutorial in `docs/tutorials/` that sets the
IIT-3.0/AC formalism and computes the actual-causation accounts live.

**Tech Stack:** PyPhi 2.0, `pyphi.actual` (Transition/account), myst-nb +
jupytext for the paired notebook.

## Global Constraints

- The builders carry **no `@config.override`**; constructing a Substrate/
  Transition is config-independent. The IIT-3.0 override (built from the `iit3`
  preset with `mechanism_partition_scheme="WEDGE_TRIPARTITION"`,
  `mechanism_phi_measure="AID"`, `alpha_measure="WPMI"`,
  `validate_system_states=False`) lives in the **tutorial** and in the builder
  **tests**, applied around `account()` only.
- Species set is `{"F1", "F2", "F3"}`; default `"F3"`. Unknown species →
  `ValueError`.
- Node labels, weights, before/after states, and cause/effect unit sets for
  each species come from the current `frog_example` F3/F2/F1 blocks
  (`pyphi/examples.py`), reconciled to the paper's figures (the current F1 block
  uses generic `S1..M2` labels; F2/F3 use anatomical `SL/SC/SR/CL/...` — make
  the labels consistent with the paper).
- Tutorial cells execute at build time; the `-W --keep-going` docs build must
  stay green. Paired `.ipynb` is committed output-free.
- Commit trailer on every commit (see repo convention). Stage only named files.

---

## Task 1: Example builders

**Files:**
- Modify: `pyphi/examples.py` (remove `frog_example`; add `_frog_net`,
  `frog_substrate`, `frog_transition`)
- Test: `test/test_examples.py` (or the existing examples test module)

**Interfaces produced:**
- `frog_substrate(species: str = "F3") -> Substrate`
- `frog_transition(species: str = "F3") -> Transition`

- [ ] **Step 1: Write failing tests.** For each `species in ("F1","F2","F3")`:
  `frog_substrate(species)` returns a `Substrate` with the expected `size` and
  `node_labels`; `frog_transition(species)` returns a `pyphi.actual.Transition`;
  and, inside the IIT-3.0 override, `pyphi.actual.account(frog_transition(species))`
  returns a non-empty account whose causal links include at least one composite
  (size ≥ 2) cause purview. Also assert `frog_substrate("bogus")` raises
  `ValueError`, and that both builders succeed under the **default** config
  (no override needed to construct).

- [ ] **Step 2: Run tests, verify they fail** (`frog_substrate` undefined).

- [ ] **Step 3: Implement.** Lift the substrate-construction helper (`LogFunc`,
  `Gauss`, `NR`, and the `get_net` body) out of `frog_example` into a
  module-level private `_frog_net(mech_func, weights, *, mu, si, node_labels,
  ...) -> Substrate`. Define a small internal table mapping each species to its
  `(mech_func, weights, node_labels, before_state, after_state, cause_units,
  effect_units)` — taken from the current F3/F2/F1 blocks, labels reconciled to
  the paper. `frog_substrate(species)` builds via `_frog_net`; `frog_transition`
  wraps it in `actual.Transition(substrate, before, after, cause_units,
  effect_units)`. Register both with `@register_example`. Delete `frog_example`.

- [ ] **Step 4: Run tests, verify pass.** Also run `uv run pytest test/ -q` for
  the examples module and a quick `grep -rn "frog_example" pyphi/ test/ docs/`
  to confirm no dangling references.

- [ ] **Step 5: Commit** (`pyphi/examples.py`, the test file).

## Task 2: The tutorial page

**Files:**
- Create: `docs/tutorials/causal-reductionism.md` (+ paired
  `causal-reductionism.ipynb` via `jupytext --sync`)
- Modify: `docs/tutorials/index.md` (toctree entry)

- [ ] **Step 1: Author the page.** MyST front matter matching the other
  tutorials (`jupytext: formats: md:myst,ipynb` + `text_representation` +
  `kernelspec`), download link + Colab badge to
  `docs/tutorials/causal-reductionism.ipynb`. Content per the spec's six
  sections: the frog world and the reductionist claim; the IIT-3.0/AC config
  override (built from `iit3` + WEDGE/AID/WPMI/validate-off, one line on each);
  **F3 in depth** (`frog_substrate("F3")` / `frog_transition("F3")` →
  `account()`, walking the first-order links and the composite second-order
  links reductionism misses); **F2 and F1 compared** (lighter); the takeaway;
  and links to the actual-causation theory/reference and how-tos. Prose numbers
  come from the executed cells; LaTeX (`$...$`) for math.

- [ ] **Step 2: Add to the toctree** in `docs/tutorials/index.md`.

- [ ] **Step 3: Sync the paired notebook**
  (`env -u VIRTUAL_ENV uv run --group docs jupytext --sync
  docs/tutorials/causal-reductionism.md`) and confirm it is output-free.

- [ ] **Step 4: Commit** (the `.md`, the `.ipynb`, `index.md`).

## Task 3: Build gate and finalize

- [ ] **Step 1: Clean `-W` build.** `rm -rf docs/_build docs/reference/_autosummary`
  then the full `env -u VIRTUAL_ENV uv run --all-extras --group docs sphinx-build
  -W --keep-going -b html docs docs/_build/html`; confirm `BUILD_RC=0` and that
  the new page rendered with executed output.

- [ ] **Step 2: Full test suite** (`uv run pytest`) green (no-path run, so the
  examples doctests are included).

- [ ] **Step 3: ROADMAP.** If the tutorials set is tracked there, note the added
  causal-reductionism tutorial. Commit any doc/roadmap touch-up.

## Self-review checklist

- Spec builders (frog_substrate/frog_transition, species param, no config) → Task 1. ✓
- Spec tutorial (six sections, F3 deep + F1/F2, executable, paired) → Task 2. ✓
- Spec removal of `frog_example` → Task 1 Step 3 + Step 4 grep. ✓
- Spec verification (builds default config, account under override, -W green,
  suite green) → Task 1 Step 4, Task 3. ✓
