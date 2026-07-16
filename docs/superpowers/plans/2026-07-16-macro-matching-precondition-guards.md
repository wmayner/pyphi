# Macro/Matching Precondition Guards Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert four silently-violated preconditions in the macro and matching modules into enforced guards that raise `ValueError`.

**Architecture:** Guards land at the shared root of each entry family: two small validators in `triggered_tpm.py` used by both matching entry points; the existing macro validators move from `macro/system.py` down to `macro/tpm.py` so `macro_tpms` validates its own inputs; one check added to `MacroUnit.__post_init__`; mechanism/state pairs canonicalized inside `_marginalize_system`.

**Tech Stack:** Python 3.13, numpy, pytest.

## Global Constraints

- Run everything via `uv run` from the worktree root (`.claude/worktrees/macro-matching-preconditions`).
- Commit messages end with the `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe` trailers. Never `--no-verify`.
- Docstrings: NumPy style, final-state impersonal voice, no planning-artifact references. Binary limits are worded "Only binary substrates are currently supported" (an implementation limit, not a theory claim).
- Final gate: pathless `uv run pytest` (no path argument) redirected to a file; read the summary line.

---

### Task 1: Matching entry guards (binary + sorted indices)

**Files:**
- Modify: `pyphi/matching/triggered_tpm.py` (new validators + calls in `build_triggered_tpm`, docstring)
- Modify: `pyphi/matching/system.py` (calls in `PerceptualSystem.__post_init__`, docstring)
- Test: `test/matching/test_triggered_tpm.py`, `test/matching/test_matching_system.py`

**Interfaces:**
- Produces: `_validate_binary_substrate(substrate)` and `_validate_sorted_indices(name, indices)` in `pyphi/matching/triggered_tpm.py`, importable by `matching/system.py`. Both raise `ValueError`; no return value.

- [ ] **Step 0: Install optional extras into the worktree venv** (needed for the final pathless sweep)

```bash
cd /Users/will/projects/pyphi/.claude/worktrees/macro-matching-preconditions
WT_PY="$(uv run python -c 'import sys; print(sys.executable)')"
env -u VIRTUAL_ENV uv pip install --python "$WT_PY" -e ".[visualize,caching,emd,xarray]" pot
```

Expected: installs succeed; `uv run python -c "import xarray, ot"` exits 0.

- [ ] **Step 1: Write the failing tests**

Append to `test/matching/test_triggered_tpm.py` (already imports `numpy as np`, `pytest`, `examples`, `build_triggered_tpm`; add `from pyphi.substrate import Substrate` to the imports in isort order):

```python
def test_build_triggered_tpm_rejects_kary_substrate():
    f0 = np.full((3, 2, 3), 1 / 3)
    f1 = np.full((3, 2, 2), 1 / 2)
    substrate = Substrate(
        marginals=[f0, f1],
        state_space=((0, 1, 2), (0, 1)),
        cm=np.ones((2, 2)),
    )
    with pytest.raises(ValueError, match="binary"):
        build_triggered_tpm(substrate, (0,), (1,), tau=1, tau_clamp=1)


def test_build_triggered_tpm_rejects_unsorted_indices():
    substrate = examples.basic_substrate()
    with pytest.raises(ValueError, match="system_indices"):
        build_triggered_tpm(substrate, (0,), (2, 1), tau=1, tau_clamp=1)
    with pytest.raises(ValueError, match="sensory_indices"):
        build_triggered_tpm(substrate, (1, 0), (2,), tau=1, tau_clamp=1)


def test_build_triggered_tpm_rejects_duplicate_indices():
    substrate = examples.basic_substrate()
    with pytest.raises(ValueError, match="system_indices"):
        build_triggered_tpm(substrate, (0,), (1, 1), tau=1, tau_clamp=1)
```

Append to `test/matching/test_matching_system.py` (already imports `pytest`, `examples`, `PerceptualSystem`; add `import numpy as np` and `from pyphi.substrate import Substrate` in isort order):

```python
def test_perceptual_system_rejects_kary_substrate():
    f0 = np.full((3, 2, 3), 1 / 3)
    f1 = np.full((3, 2, 2), 1 / 2)
    substrate = Substrate(
        marginals=[f0, f1],
        state_space=((0, 1, 2), (0, 1)),
        cm=np.ones((2, 2)),
    )
    with pytest.raises(ValueError, match="binary"):
        PerceptualSystem(substrate, system_indices=(1,), sensory_indices=(0,))


def test_perceptual_system_rejects_unsorted_indices():
    substrate = examples.basic_substrate()
    with pytest.raises(ValueError, match="system_indices"):
        PerceptualSystem(substrate, system_indices=(2, 1), sensory_indices=(0,))
    with pytest.raises(ValueError, match="sensory_indices"):
        PerceptualSystem(substrate, system_indices=(2,), sensory_indices=(1, 0))
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest test/matching/test_triggered_tpm.py test/matching/test_matching_system.py -k "rejects" -v > /tmp/m1_fail.log 2>&1; tail -12 /tmp/m1_fail.log
```

Expected: the five new tests FAIL (`DID NOT RAISE` or garbage-output assertions); pre-existing tests untouched.

- [ ] **Step 3: Add the validators and wire both entry points**

In `pyphi/matching/triggered_tpm.py`, add `import itertools` to the imports (isort order) and insert before `_system_step_tpm`:

```python
def _validate_binary_substrate(substrate) -> None:
    """Raise if the substrate has any non-binary unit.

    The triggered-TPM construction operates on the binary state-by-node
    representation; only binary substrates are currently supported.
    """
    sizes = substrate.factored_tpm.alphabet_sizes
    if any(size != 2 for size in sizes):
        raise ValueError(
            "only binary substrates are currently supported; "
            f"got alphabet sizes {sizes}"
        )


def _validate_sorted_indices(name: str, indices) -> None:
    """Raise unless ``indices`` is strictly increasing (sorted, no duplicates).

    Triggered-TPM axes and stimulus/state tuples are positional relative
    to these index tuples, so only the sorted form is unambiguous.
    """
    if not all(a < b for a, b in itertools.pairwise(indices)):
        raise ValueError(
            f"{name} must be strictly increasing (sorted, without "
            f"duplicates); got {tuple(indices)}"
        )
```

At the top of `build_triggered_tpm` (before `n = len(substrate.node_indices)`):

```python
    _validate_binary_substrate(substrate)
    _validate_sorted_indices("sensory_indices", sensory_indices)
    _validate_sorted_indices("system_indices", system_indices)
```

In the `build_triggered_tpm` docstring, replace the sentence `Assumes a binary substrate.` with `Only binary substrates are currently supported.`

In `pyphi/matching/system.py`, extend the `triggered_tpm` import block:

```python
from .triggered_tpm import TriggeredTPM
from .triggered_tpm import _validate_binary_substrate
from .triggered_tpm import _validate_sorted_indices
from .triggered_tpm import build_triggered_tpm
```

At the end of `PerceptualSystem.__post_init__`:

```python
        _validate_binary_substrate(self.substrate)
        _validate_sorted_indices("system_indices", self.system_indices)
        _validate_sorted_indices("sensory_indices", self.sensory_indices)
```

In the `PerceptualSystem` docstring, replace `Assumes a binary
substrate.` with `Only binary substrates are currently supported.`

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest test/matching/ -q > /tmp/m1_pass.log 2>&1; tail -3 /tmp/m1_pass.log
```

Expected: entire matching suite PASS.

- [ ] **Step 5: Commit**

```bash
git add pyphi/matching/triggered_tpm.py pyphi/matching/system.py test/matching/test_triggered_tpm.py test/matching/test_matching_system.py
git commit -m "Validate substrate and index preconditions in matching entry points

build_triggered_tpm and PerceptualSystem assumed binary substrates and
sorted index tuples but enforced neither: a k-ary substrate flowed
through the binary ON-probability slice and produced a well-formed but
meaningless triggered TPM, and unsorted index tuples silently misaligned
the query surface's axis bookkeeping. Both entry points now reject
non-binary substrates (an implementation limit) and index tuples that
are not strictly increasing.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe"
```

---

### Task 2: Canonicalize mechanism/state pairs in the query surface

**Files:**
- Modify: `pyphi/matching/triggered_tpm.py` (`_marginalize_system`)
- Test: `test/matching/test_triggered_tpm.py`

**Interfaces:**
- Consumes: nothing from Task 1 (independent change in the same file).
- Produces: `conditional_probability` / `marginal_probability` accept the mechanism in any order, paired positionally with `state`.

- [ ] **Step 1: Write the failing test**

Append to `test/matching/test_triggered_tpm.py` (reuses the copy-substrate construction from `test_system_axes_follow_unit_order`, where the triggered state for stimulus `(1,)` is `(unit1, unit2) = (1, 0)` — asymmetric, so an order mixup is visible):

```python
def test_mechanism_order_is_canonicalized():
    import pyphi

    sbn = np.zeros((2, 2, 2, 3))
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                sbn[a, b, c, 1] = a
    substrate = pyphi.Substrate(sbn)
    t = build_triggered_tpm(
        substrate, sensory_indices=(0,), system_indices=(1, 2), tau=1, tau_clamp=1
    )
    # (unit1, unit2) = (1, 0) with certainty for stimulus (1,).
    assert t.conditional_probability((1, 2), (1, 0), (1,)) == pytest.approx(1.0)
    # Same query with the mechanism in reversed order, state paired to match.
    assert t.conditional_probability((2, 1), (0, 1), (1,)) == pytest.approx(1.0)
    assert t.marginal_probability((2, 1), (0, 1)) == pytest.approx(
        t.marginal_probability((1, 2), (1, 0))
    )


def test_marginalization_rejects_duplicate_mechanism(ttpm):
    with pytest.raises(ValueError, match="duplicate"):
        ttpm.conditional_probability((1, 1), (0, 0), (0,))
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest test/matching/test_triggered_tpm.py -k "canonicalized or duplicate_mechanism" -v > /tmp/m2_fail.log 2>&1; tail -8 /tmp/m2_fail.log
```

Expected: `test_mechanism_order_is_canonicalized` FAILS on the reversed-order query (wrong probability, no exception); `test_marginalization_rejects_duplicate_mechanism` FAILS with `DID NOT RAISE`.

- [ ] **Step 3: Canonicalize in `_marginalize_system`**

In `pyphi/matching/triggered_tpm.py`, replace the body of `_marginalize_system` up to the `keep = ...` line:

```python
    def _marginalize_system(self, distribution, mechanism, state) -> float:
        """Return Pr(mechanism = state) from a distribution over the system axes.

        Sums out the system units not in ``mechanism``. Requires ``mechanism``
        to be a subset of ``system_indices`` (without duplicates) and ``state``
        to match its length; the (mechanism, state) pairs may be given in any
        order.
        """
        mechanism = tuple(mechanism)
        if len(set(mechanism)) != len(mechanism):
            raise ValueError(f"duplicate units in mechanism {mechanism}")
        if not set(mechanism) <= set(self.system_indices):
            raise ValueError(
                f"mechanism {mechanism} is not a subset of system_indices "
                f"{self.system_indices}"
            )
        if len(state) != len(mechanism):
            raise ValueError(f"state {state} length != mechanism {mechanism} length")
        # Canonicalize: sort the (mechanism, state) pairs together so the
        # axis bookkeeping below can assume increasing mechanism order.
        pairs = sorted(zip(mechanism, state, strict=True))
        mechanism = tuple(m for m, _ in pairs)
        state = tuple(s for _, s in pairs)
        keep = [self.system_indices.index(m) for m in mechanism]
```

Keep the remainder of the method unchanged, but replace the trailing comment

```python
        # mechanism and system_indices are both sorted, so `keep` is increasing
        # and the remaining axes are already in mechanism order.
```

with

```python
        # `mechanism` is sorted above and `system_indices` is validated sorted
        # at construction, so `keep` is increasing and the remaining axes are
        # already in mechanism order.
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest test/matching/ -q > /tmp/m2_pass.log 2>&1; tail -3 /tmp/m2_pass.log
```

Expected: entire matching suite PASS.

- [ ] **Step 5: Commit**

```bash
git add pyphi/matching/triggered_tpm.py test/matching/test_triggered_tpm.py
git commit -m "Accept any mechanism order in triggered-TPM queries

_marginalize_system assumed the mechanism tuple was sorted but did not
enforce it: a mechanism given in a different order than system_indices
indexed the reduced axes in the wrong order and returned wrong
multi-unit probabilities. The (mechanism, state) pairs are now sorted
together before lookup — the pairing is explicit in the call, so the
reorder is semantics-preserving — and duplicate mechanism units are
rejected.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe"
```

---

### Task 3: `macro_tpms` validates its own preconditions

**Files:**
- Modify: `pyphi/macro/tpm.py` (receive the three validators; call them in `macro_tpms`)
- Modify: `pyphi/macro/system.py` (remove the moved definitions; import from `macro.tpm`)
- Test: `test/macro/test_macro_tpm.py`

**Interfaces:**
- Consumes: `_system_micro_indices` (already in `macro/tpm.py`), `MacroUnit` (already imported there).
- Produces: `_validate_units(substrate, units)`, `_validate_nested_apportionment(unit)`, `_normalize_history(units, substrate, micro_history)` importable from `pyphi.macro.tpm`; `macro_tpms` raises `ValueError` on wrong-length history and non-disjoint units.

- [ ] **Step 1: Write the failing tests**

Append to `test/macro/test_macro_tpm.py` (already imports `MacroUnit`, `blackbox`, `coarse_grain`, `macro_tpms`, `pytest`; `_asymmetric_substrate` is defined in the file):

```python
class TestPreconditionValidation:
    def test_short_history_rejected(self):
        substrate = _asymmetric_substrate()
        unit = MacroUnit((0, 1), 2, blackbox(2, 2, (0,)))
        with pytest.raises(ValueError, match="micro_history"):
            macro_tpms(substrate, (unit,), ((0, 0, 0, 0),))  # needs 2 entries

    def test_correct_history_accepted(self):
        substrate = _asymmetric_substrate()
        unit = MacroUnit((0, 1), 2, blackbox(2, 2, (0,)))
        cause, effect = macro_tpms(
            substrate, (unit,), ((0, 0, 0, 0), (0, 0, 0, 0))
        )
        assert cause.n_nodes == 1 and effect.n_nodes == 1

    def test_overlapping_units_rejected(self):
        substrate = _asymmetric_substrate()
        units = (
            MacroUnit((0, 1), 1, coarse_grain(2, (1, 2))),
            MacroUnit((1, 2), 1, coarse_grain(2, (1, 2))),
        )
        with pytest.raises(ValueError, match="disjoint"):
            macro_tpms(substrate, units, ((0, 0, 0, 0),))
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest test/macro/test_macro_tpm.py::TestPreconditionValidation -v > /tmp/m3_fail.log 2>&1; tail -10 /tmp/m3_fail.log
```

Expected: `test_short_history_rejected` FAILS (`DID NOT RAISE` — the wraparound silently succeeds); `test_overlapping_units_rejected` FAILS (`DID NOT RAISE` or a different, non-matching error); `test_correct_history_accepted` PASSES already.

- [ ] **Step 3: Move the validators and wire `macro_tpms`**

Cut `_validate_units`, `_validate_nested_apportionment`, and `_normalize_history` (bodies unchanged) from `pyphi/macro/system.py` and paste them into `pyphi/macro/tpm.py` immediately before `macro_tpms`. They depend only on `_system_micro_indices` and `MacroUnit`, both already available in `tpm.py`.

In `pyphi/macro/system.py`, extend the existing `macro.tpm` import block:

```python
from pyphi.macro.tpm import _normalize_history
from pyphi.macro.tpm import _system_micro_indices
from pyphi.macro.tpm import _validate_units
from pyphi.macro.tpm import macro_tpms
```

(Remove the `_system_micro_indices` line if `system.py` no longer references it after the move — check with grep and keep imports minimal.)

In `macro_tpms`, replace the two normalization lines

```python
    units = tuple(units)
    micro_history = tuple(tuple(s) for s in micro_history)
```

with

```python
    units = tuple(units)
    _validate_units(substrate, units)
    micro_history = _normalize_history(units, substrate, micro_history)
```

Add a `Raises` section to the `macro_tpms` docstring:

```
    Raises
    ------
    ValueError
        If the substrate is not binary, the units are not pairwise
        disjoint (Eq. 18), or ``micro_history`` does not have exactly
        ``max(tau_J)`` entries.
```

- [ ] **Step 4: Run the macro suite to verify**

```bash
uv run pytest test/macro/ -q > /tmp/m3_pass.log 2>&1; tail -3 /tmp/m3_pass.log
```

Expected: entire macro suite PASS (wrappers still work; double validation is harmless).

- [ ] **Step 5: Commit**

```bash
git add pyphi/macro/tpm.py pyphi/macro/system.py test/macro/test_macro_tpm.py
git commit -m "Validate preconditions in the public macro_tpms entry point

macro_tpms validated none of its documented preconditions: a
micro_history one entry shorter than max(tau_J) silently wrapped via
negative indexing, reusing the current state as the earliest state of
the window and producing a wrong cause TPM (Eq. 34 background
conditioning); unit disjointness (Eq. 18) was likewise unchecked. The
validators move from macro.system to macro.tpm — macro_tpms now
validates its own inputs, and the wrappers import the shared helpers.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe"
```

---

### Task 4: Reject negative background-apportionment indices

**Files:**
- Modify: `pyphi/macro/units.py` (`MacroUnit.__post_init__`)
- Test: `test/macro/test_macro_units.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `MacroUnit` construction raises `ValueError` on negative apportionment indices.

- [ ] **Step 1: Write the failing test**

Append to `test/macro/test_macro_units.py` (check its imports; it should already import `MacroUnit`, `coarse_grain`, `pytest` — add any missing in isort order):

```python
def test_negative_background_apportionment_rejected():
    with pytest.raises(ValueError, match="negative"):
        MacroUnit(
            (0, 1),
            1,
            coarse_grain(2, (1, 2)),
            background_apportionment=(-1,),
        )
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest test/macro/test_macro_units.py -k negative_background -v > /tmp/m4_fail.log 2>&1; tail -5 /tmp/m4_fail.log
```

Expected: FAIL with `DID NOT RAISE`.

- [ ] **Step 3: Add the check**

In `pyphi/macro/units.py`, in `__post_init__`, immediately after
`apportionment = self.background_apportionment`:

```python
        if any(i < 0 for i in apportionment):
            raise ValueError(
                "negative background apportionment index: "
                f"{sorted(i for i in apportionment if i < 0)}"
            )
```

- [ ] **Step 4: Run the test file to verify**

```bash
uv run pytest test/macro/test_macro_units.py -q > /tmp/m4_pass.log 2>&1; tail -3 /tmp/m4_pass.log
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add pyphi/macro/units.py test/macro/test_macro_units.py
git commit -m "Reject negative background-apportionment indices in MacroUnit

Negative apportionment indices passed every validation layer, were
silently inert in the Eq. 29 discounting (axes are drawn from
range(n)), and perturbed the construction cache key, so semantically
identical units hashed differently. MacroUnit now rejects them at
construction, mirroring the existing constituent-index check.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe"
```

---

### Task 5: ROADMAP wishlist entry, changelog, full-suite gate

**Files:**
- Modify: `ROADMAP.md` (wishlist entry for k-ary matching)
- Create: `changelog.d/macro-matching-preconditions.fix.md`

**Interfaces:**
- Consumes: Tasks 1–4 committed.
- Produces: branch ready to merge; suite green.

- [ ] **Step 1: Add the ROADMAP wishlist entry**

Locate the Wishlist section of `ROADMAP.md` (the subsection that best fits feature work; "Correctness & rigor" holds the AC tie-resolver follow-up — prefer a features-oriented subsection if one exists, else add under the general Wishlist). Append:

```markdown
- **K-ary support for the matching/triggered-TPM path.** The
  clamp-then-noise construction (`pyphi/matching/triggered_tpm.py`) is
  built on the binary state-by-node representation (ON-probability
  slice, `(2,)*n` step shapes, binary state-by-state conversion,
  little-endian state decoding); non-binary substrates are rejected at
  the entry points as a current implementation limit. Generalizing
  requires reworking the construction onto explicit-alphabet per-node
  distributions and mixed-radix state indexing (the k-ary primitives
  exist in `pyphi.core.tpm` and `pyphi.utils`), plus confirming the
  matching formalism's definitions carry over unchanged.
```

- [ ] **Step 2: Write the changelog fragment**

```bash
cat > changelog.d/macro-matching-preconditions.fix.md <<'EOF'
Public macro/matching entry points now validate their preconditions
instead of returning silently wrong results: `build_triggered_tpm` and
`PerceptualSystem` reject non-binary substrates (a current
implementation limit) and unsorted index tuples; triggered-TPM queries
accept the mechanism in any order by canonicalizing the
(mechanism, state) pairs; `macro_tpms` validates history length and
unit disjointness (a one-entry-short history previously wrapped around
and produced a wrong cause TPM); and `MacroUnit` rejects negative
background-apportionment indices.
EOF
```

- [ ] **Step 3: Run the full pathless suite**

```bash
uv run pytest -q > /tmp/full_suite.log 2>&1; echo done
tail -4 /tmp/full_suite.log
```

Expected: summary line with 0 failures (≈3650+ passed). Read the summary from the file — never trust a pipeline exit code. If the perf-counter pin fails, regenerate with `uv run python scripts/gen_perf_counts.py` and inspect the diff — these guards add no repertoire calls, so any pin change is a red flag to investigate, not commit.

- [ ] **Step 4: Commit**

```bash
git add ROADMAP.md changelog.d/macro-matching-preconditions.fix.md
git commit -m "Record the k-ary matching wishlist entry and changelog fragment

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe"
```

---

### Post-merge bookkeeping (main tree, not the worktree)

`REVIEW-2026-07-13.md` is untracked and lives only in the main tree. After the
branch merges to main, update its status block: the four macro/matching
precondition findings are fixed (with the merge hash), completing Wave 1.
Update the memory file `project_whole_library_review_2026_07_13.md` likewise.
