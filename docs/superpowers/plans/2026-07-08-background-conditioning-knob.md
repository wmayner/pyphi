# IIT 3.0 Background-Conditioning Knob Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a config option `background_conditioning` selecting how substrate
units outside the candidate system enter cause repertoires:
`CAUSAL_MARGINALIZATION` (IIT 4.0 Eq. 4, the current behavior, stays the
global default) or `CONDITION_CURRENT_STATE` (the PyPhi 1.x convention —
background fixed at its observed current state). `presets.iit3` selects
`CONDITION_CURRENT_STATE`, so IIT 3.0 analyses of proper-subset systems
reproduce published 1.x results. Both conventions get discriminating
regression tests anchored to a genuine PyPhi 1.2.0 oracle.

**Architecture:** The two conventions diverge at exactly one computation
point: `System.cause_marginal` (`pyphi/system.py`), the sole producer of
cause factors — everything downstream (`nodes`, the repertoire kernel, both
formalisms) consumes its output. The knob therefore branches there, backed by
a new kernel operation `cause_conditioned` in
`pyphi/core/tpm/marginalization.py` that conditions the factored TPM's
background input axes at the observed state and returns the conditioned
forward factors directly as `CauseMarginals`. This is *exactly* what the
Eq. 4 machinery degenerates to on a background-conditioned TPM (verified:
running `_cause_marginal_factored` on the conditioned TPM produces
bit-identical factors — the background weight collapses to exactly 1.0), so
no repertoire-algebra code changes; the direct construction just skips a
no-op sum-product and a spurious reachability check. The
`CAUSAL_MARGINALIZATION` branch runs the current code unchanged, so
marginalized results are byte-identical and all existing goldens pass
untouched. The option is read at *compute* time (systems built before a
`config.override(**presets.iit3)` block still honor the override inside it),
which requires convention-aware caching: `System` caches marginals and nodes
per convention, and the kernel memo key gains the resolved convention.
Actual causation is insulated by pinning its underlying `System` to
`CAUSAL_MARGINALIZATION` via a new optional `System.background_conditioning`
field — the AC background rule is its own config axis
(`ActualCausationConfig.background_scheme`) and must not move when the IIT
knob does.

**Tech Stack:** Python 3.13, numpy, pytest. No new dependencies. One
out-of-band tool run: a genuine PyPhi 1.2.0 oracle in an isolated
Python 3.9 venv (recipe below; verified working).

## Global Constraints

- Run everything with `uv run` (e.g. `uv run pytest`, `uv run python`).
- Work in a git worktree under `.claude/worktrees/` (confirm branch name with
  the user at execution start; base on the current working branch).
- Float comparisons in tests use `pytest.approx` — never `==` on φ values.
- Every user-facing change gets a changelog fragment in `changelog.d/`
  (`<name>.<type>.md`), committed with the task.
- Docstrings describe final state only — no migration narrative, no planning
  artifacts (no task numbers, no "restores the 1.x behavior we lost", no
  design-alternative discussion).
- Do not use `git checkout -- <path>` for cleanup; other sessions may have
  unrelated working-tree changes — stage only files this plan touches.
- Never pass `--no-verify` to git. If pre-commit hooks fail, fix the failure.
- The final verification (Task 6) must run `uv run pytest` **with no path
  argument** at least once (bare paths skip the doctest sweep).

## Background for implementers (read once)

**The mathematics.** For a system S that is a proper subset of its substrate
U, the background W = U∖S enters cause repertoires differently under three
conventions: (2014 paper) W fixed at its actual *past* state — requires the
past state, which no PyPhi version has ever taken as input; never
implemented, and stays unimplemented (documented only). (PyPhi 1.x /
post-2014 literature) W fixed at its *current* state for both directions.
(IIT 4.0, Eq. 4) W causally marginalized conditional on the current state.
PyPhi 2.0's shared kernel currently applies Eq. 4 for every formalism,
including IIT 3.0 mode. The three conventions coincide when W is empty
(full-substrate systems), so the 2014 worked examples and all full-substrate
goldens are convention-invariant. The **effect side conditions W at its
current state under every convention** (`System.effect_marginal` already does
this); only the cause side branches.

**Where the code computes each side.** `System.cause_marginal`
(`pyphi/system.py:330`) calls `cause_marginal` →
`_cause_marginal_factored` (`pyphi/core/tpm/marginalization.py:122`), the
Eq. 4 sum-product. `System.effect_marginal` (`system.py:339`) conditions the
factored TPM on `external_indices` at the observed state — already the shared
convention, untouched by this plan. `System.nodes` bakes per-node slices of
`cause_marginal` into `Node` objects; the memoized kernel functions in
`pyphi/core/repertoire_algebra.py` read those slices via `cs._index2node`.
`FactoredTPM.condition` (`pyphi/core/tpm/factored.py:247`) fixes input axes
at given states and keeps them as size-1 dims (the full-ndim convention), so
a conditioned TPM is a valid `FactoredTPM` whose factors slot directly into
`CauseMarginals`.

**Verified facts this plan relies on** (each confirmed by a live prototype of
the seam against this codebase, and against a genuine PyPhi 1.2.0 install):

1. On the discriminating substrate below (S={A,B}, W={C}, state (1,0,0)),
   the mechanism-{A}/purview-{B} cause repertoire is
   `(0.40566037735849053, 0.5943396226415094)` marginalized vs `(0.1, 0.9)`
   conditioned; genuine PyPhi 1.2.0 produces `(0.1, 0.9)`. End-to-end
   `iit3.sia` under `presets.iit3`: Φ = 0.41607 marginalized vs
   **0.72 conditioned — exactly PyPhi 1.2.0's value**.
2. Running the Eq. 4 machinery on the background-conditioned TPM returns
   factors `np.array_equal` to the conditioned forward factors (the weight
   is exactly 1.0 in floating point), so the direct construction is exact.
3. `basic_substrate` subset (1,2): **every** cause repertoire coincides
   across conventions → the committed `basic_subset_iit3_emd` golden passes
   unchanged under the preset flip.
4. `basic_substrate` subset (0,2): seven cause repertoires differ; SIA φ is
   0.5 marginalized vs 1.0 conditioned — and **genuine PyPhi 1.2.0 gives
   1.0** (cut `[A] ──/ /──➤ [C]`, 2 concepts). The pins in
   `test/formalism/test_complexes.py` (`(0,2): 0.5` and the `all_sias` list)
   are marginalized-semantics values and change to the 1.x values under the
   preset flip. PyPhi 1.2.0 subset values for basic at (1,0,0):
   `(0,1,2): 2.3125, (1,2): 1.0, (0,2): 1.0, (0,1): 0.0`.
5. `validate.state_reachable` (`pyphi/validate.py:170`) already checks that
   the subsystem state is producible under *background-conditioned* forward
   dynamics, so states pathological under the conditioned convention are
   rejected at `System` construction (when validation is on); all-zero cause
   repertoires that slip through (validation off) hit the existing
   `UNREACHABLE_STATE` null-RIA path in `pyphi/formalism/queries.py:166`.
6. `null_distinction` and unconstrained cause repertoires use the
   empty-mechanism path (`cause_repertoire(cs, (), purview)`), which consults
   no cause factors — convention-independent, so `System.null_distinction`
   stays an ordinary `cached_property`.

**The discriminating substrate** (used throughout):
`p(A'=1|b,c) = 0.9 if (b or c) else 0.1` (parents B, C); `B' = copy(A)`;
`p(C'=1|·) = 0.5` (no parents). System S = {A, B}, background W = {C},
state u = (1, 0, 0). The noisy background parent C is what makes the
conventions diverge — deterministic backgrounds often mask the difference
(fact 3 vs fact 4).

**Caching landscape.** Three caches would otherwise serve stale values when
the option changes mid-process: (a) `System.cause_marginal` /
`System.nodes` `cached_property`s — replaced by per-convention dicts;
(b) the kernel `ContentCache` in `repertoire_algebra._memoize`, keyed on
`System._fingerprint` + args — gains the resolved convention in the key;
(c) the disk result cache — already safe: its key digests
`dataclasses.asdict(snapshot.formalism)` (`pyphi/cache/disk.py:88-121`),
which picks up any new `IITConfig` field automatically. The config snapshot
attached to results (`pyphi/conf/snapshot.py`) and the flat-access routing
(`pyphi/conf/_field_routing.py`) are likewise `fields()`-introspected —
adding the `IITConfig` field flows through with no registration.

**PyPhi 1.2.0 oracle environment** (verified working on this machine):

```bash
VENV=/tmp/pyphi-1x-oracle/.venv
uv venv --python 3.9 "$VENV"
VIRTUAL_ENV="$VENV" uv pip install "pyphi==1.2.0"
# run oracle scripts from a directory with no pyphi_config.yml:
"$VENV/bin/python" scripts/gen_iit3_background_oracle.py > oracle_out.json
```

---

### Task 1: The `background_conditioning` config option

Config layer only — the option is inert until Task 2 wires it into the
kernel. No cross-field (B13) constraint is added: both values are
well-defined under every IIT version (the constraints module only encodes
combinations that compute nonsense, and running IIT 4.0 with the conditioned
background is a legitimate research variant, not an invalid state).

**Files:**
- Modify: `pyphi/conf/formalism.py` (field + validation on `IITConfig`)
- Test: `test/formalism/test_formalism_config.py` (extend)
- Create: `changelog.d/background-conditioning-option.config.md`

**Interfaces:**
- Produces: `IITConfig.background_conditioning: str = "CAUSAL_MARGINALIZATION"`;
  module constant `_VALID_BACKGROUND_CONDITIONING` (imported by `System`
  validation in Task 2). Flat access (`config.background_conditioning`),
  YAML loading, `config.override(...)`, `ConfigSnapshot.diff`/`as_kwargs`,
  and the disk-cache config digest all pick the field up automatically via
  `fields()` introspection — the tests confirm rather than implement this.

- [ ] **Step 1: Write the failing tests**

Append to `test/formalism/test_formalism_config.py` (match the file's
existing import style):

```python
class TestBackgroundConditioning:
    def test_default_is_causal_marginalization(self):
        from pyphi.conf.formalism import IITConfig

        assert IITConfig().background_conditioning == "CAUSAL_MARGINALIZATION"

    def test_invalid_value_rejected(self):
        from pyphi.conf.formalism import IITConfig

        with pytest.raises(ValueError, match="background_conditioning"):
            IITConfig(background_conditioning="PAST_STATE")

    def test_flat_access_routes_to_iit_layer(self):
        import pyphi

        with pyphi.config.override(
            background_conditioning="CONDITION_CURRENT_STATE"
        ):
            assert (
                pyphi.config.formalism.iit.background_conditioning
                == "CONDITION_CURRENT_STATE"
            )
            assert (
                pyphi.config.background_conditioning
                == "CONDITION_CURRENT_STATE"
            )
        assert (
            pyphi.config.background_conditioning == "CAUSAL_MARGINALIZATION"
        )

    def test_snapshot_records_and_diffs_the_option(self):
        import pyphi

        base = pyphi.config.snapshot()
        with pyphi.config.override(
            background_conditioning="CONDITION_CURRENT_STATE"
        ):
            snap = pyphi.config.snapshot()
        diff = base.diff(snap)
        assert diff["formalism.iit.background_conditioning"] == (
            "CAUSAL_MARGINALIZATION",
            "CONDITION_CURRENT_STATE",
        )
        assert (
            snap.as_kwargs()["background_conditioning"]
            == "CONDITION_CURRENT_STATE"
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/formalism/test_formalism_config.py -v -k background`
Expected: FAIL (`IITConfig` has no field `background_conditioning`; the
invalid-value construction does not raise).

- [ ] **Step 3: Implement in `pyphi/conf/formalism.py`**

Add next to the other `_VALID_*` frozensets (~line 27):

```python
_VALID_BACKGROUND_CONDITIONING = frozenset(
    {"CAUSAL_MARGINALIZATION", "CONDITION_CURRENT_STATE"}
)
```

Add the field to `IITConfig` after `distinction_phi_normalization`
(~line 47):

```python
    background_conditioning: str = "CAUSAL_MARGINALIZATION"
```

Add to `IITConfig.__post_init__` (after the `relation_computation` check):

```python
        if self.background_conditioning not in _VALID_BACKGROUND_CONDITIONING:
            raise ValueError(
                f"background_conditioning={self.background_conditioning!r} "
                f"not in {sorted(_VALID_BACKGROUND_CONDITIONING)}"
            )
```

The full option documentation (three conventions, including the
never-implemented 2014 one) is written in Task 6 so it can describe the
preset behavior that lands in Task 4.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/formalism/test_formalism_config.py -v`
Expected: all PASS. Also confirm no collision with the AC sub-namespace
(AC's field is `background_scheme`, a different name, so
`colliding_formalism_fields()` is unaffected):
`uv run python -c "from pyphi.conf._field_routing import FIELD_TO_LAYER; print(FIELD_TO_LAYER['background_conditioning'])"`
Expected: `('formalism', 'iit')`.

- [ ] **Step 5: Changelog fragment and commit**

```bash
cat > changelog.d/background-conditioning-option.config.md <<'EOF'
Added `formalism.iit.background_conditioning`: how background units enter
cause repertoires — `"CAUSAL_MARGINALIZATION"` (IIT 4.0 Eq. 4; the default)
or `"CONDITION_CURRENT_STATE"` (background fixed at its observed state, the
PyPhi 1.x convention).
EOF
git add pyphi/conf/formalism.py test/formalism/test_formalism_config.py changelog.d/background-conditioning-option.config.md
git commit -m "Add background_conditioning config option"
```

---

### Task 2: Kernel operation and the `System.cause_marginal` dispatch

The core of the plan. Adds `cause_conditioned` to the marginalization
kernel, a `System.background_conditioning` pin field, compute-time dispatch
in `System.cause_marginal`, per-convention caching on `System`, the resolved
convention in the kernel memo key, and serialization coverage. The
`CAUSAL_MARGINALIZATION` path must call exactly the current code.

**Files:**
- Modify: `pyphi/core/tpm/marginalization.py` (add `cause_conditioned`)
- Modify: `pyphi/system.py` (field, dispatch, per-convention caches,
  `__eq__`/`__hash__`/`_fingerprint`, `apply_cut` cache-copy list)
- Modify: `pyphi/core/repertoire_algebra.py` (`_memoize` key)
- Modify: `pyphi/serialize/schema.py`, `pyphi/serialize/convert.py`
  (`SystemSchema` field)
- Modify: `test/example_substrates.py` (add `noisy_or_background_substrate`)
- Test: `test/test_background_conditioning.py` (create)
- Create: `changelog.d/system-background-conditioning-field.feature.md`

**Interfaces:**
- Consumes: `_VALID_BACKGROUND_CONDITIONING`, the config option (Task 1).
- Produces:
  - `pyphi.core.tpm.marginalization.cause_conditioned(tpm, state, node_indices, background) -> CauseMarginals`
  - `System.background_conditioning: str | None = None` (kw field; `None` =
    resolve from config at compute time; participates in `__eq__`,
    `__hash__`, `_fingerprint`, serialization)
  - `System._resolved_background_conditioning() -> str` (also the kernel
    memo-key token; Task 3 delegates it on `TransitionSystem`)
  - `test/example_substrates.py::noisy_or_background_substrate() -> Substrate`

- [ ] **Step 1: Add the shared test substrate**

Append to `test/example_substrates.py`:

```python
def noisy_or_background_substrate():
    """3-unit substrate whose cause repertoires discriminate background
    conventions for the proper-subset system {A, B}.

    ``p(A'=1 | b, c) = 0.9 if (b or c) else 0.1`` (parents B, C);
    ``B' = copy(A)``; ``p(C'=1 | ·) = 0.5`` (no parents). With system
    S = {A, B} and background W = {C}, the noisy background parent C makes
    the cause side differ between causal marginalization and current-state
    conditioning of W.
    """
    import numpy as np

    from pyphi import Substrate

    def p_a_on(b, c):
        return 0.9 if (b or c) else 0.1

    rows = []
    for c in (0, 1):
        for b in (0, 1):
            for a in (0, 1):
                rows.append((p_a_on(b, c), float(a), 0.5))
    return Substrate(
        np.array(rows),
        cm=np.array([[0, 1, 0], [1, 0, 0], [1, 0, 0]]),
        node_labels=("A", "B", "C"),
    )
```

- [ ] **Step 2: Write the failing tests**

Create `test/test_background_conditioning.py`. Check how sibling test files
import `example_substrates` (plain `from example_substrates import ...` vs
package-relative) and copy that style.

```python
"""Tests for the cause-side background-conditioning conventions."""

import numpy as np
import pytest

import pyphi
from pyphi import config
from pyphi.system import System

from example_substrates import noisy_or_background_substrate

STATE = (1, 0, 0)
SYSTEM_NODES = (0, 1)

# Manually derived on the discriminating substrate; also pinned by the
# genuine PyPhi 1.2.0 oracle (test/data/iit3-canonical/).
CAUSE_REP_MARGINALIZED = [0.40566037735849053, 0.5943396226415094]
CAUSE_REP_CONDITIONED = [0.1, 0.9]


@pytest.fixture()
def substrate():
    return noisy_or_background_substrate()


def _system(substrate, **kwargs):
    return System(substrate, STATE, node_indices=SYSTEM_NODES, **kwargs)


class TestCauseRepertoireConventions:
    def test_default_is_marginalized(self, substrate):
        rep = _system(substrate).cause_repertoire((0,), (1,)).squeeze()
        assert rep == pytest.approx(CAUSE_REP_MARGINALIZED)

    def test_conditioned_convention_via_config(self, substrate):
        with config.override(
            background_conditioning="CONDITION_CURRENT_STATE"
        ):
            rep = _system(substrate).cause_repertoire((0,), (1,)).squeeze()
        assert rep == pytest.approx(CAUSE_REP_CONDITIONED)

    def test_system_field_pins_convention_over_config(self, substrate):
        pinned = _system(
            substrate, background_conditioning="CAUSAL_MARGINALIZATION"
        )
        with config.override(
            background_conditioning="CONDITION_CURRENT_STATE"
        ):
            rep = pinned.cause_repertoire((0,), (1,)).squeeze()
        assert rep == pytest.approx(CAUSE_REP_MARGINALIZED)

    def test_invalid_field_value_rejected(self, substrate):
        with pytest.raises(ValueError, match="background_conditioning"):
            _system(substrate, background_conditioning="PAST_STATE")

    def test_effect_side_is_convention_invariant(self, substrate):
        default = _system(substrate).effect_repertoire((0,), (1,)).squeeze()
        with config.override(
            background_conditioning="CONDITION_CURRENT_STATE"
        ):
            conditioned = (
                _system(substrate).effect_repertoire((0,), (1,)).squeeze()
            )
        assert np.array_equal(default, conditioned)
        assert default == pytest.approx([0.0, 1.0])

    def test_full_substrate_system_is_convention_invariant(self, substrate):
        full = System(substrate, STATE)
        baseline = full.cause_repertoire((0,), (1,))
        with config.override(
            background_conditioning="CONDITION_CURRENT_STATE"
        ):
            conditioned = (
                System(substrate, STATE).cause_repertoire((0,), (1,))
            )
        assert np.array_equal(baseline, conditioned)


class TestCacheFreshness:
    def test_same_system_object_respects_config_flip(self, substrate):
        # The option is read at compute time: a System built (and computed
        # on) under one convention must produce the other convention's
        # values inside an override — through both the System-level caches
        # and the kernel memo cache.
        s = _system(substrate)
        before = s.cause_repertoire((0,), (1,)).squeeze()
        assert before == pytest.approx(CAUSE_REP_MARGINALIZED)
        with config.override(
            background_conditioning="CONDITION_CURRENT_STATE"
        ):
            inside = s.cause_repertoire((0,), (1,)).squeeze()
        after = s.cause_repertoire((0,), (1,)).squeeze()
        assert inside == pytest.approx(CAUSE_REP_CONDITIONED)
        assert after == pytest.approx(CAUSE_REP_MARGINALIZED)

    def test_apply_cut_shares_marginals_across_conventions(self, substrate):
        from pyphi.models.partitions import DirectedBipartition

        s = _system(substrate)
        _ = s.cause_repertoire((0,), (1,))
        cut = s.apply_cut(
            DirectedBipartition((0,), (1,), node_labels=s.node_labels)
        )
        with config.override(
            background_conditioning="CONDITION_CURRENT_STATE"
        ):
            rep = cut.cause_repertoire((0,), (1,)).squeeze()
        assert rep == pytest.approx(CAUSE_REP_CONDITIONED)


class TestKernelOperation:
    def test_conditioned_factors_match_eq4_on_conditioned_tpm(self, substrate):
        # The direct construction equals the Eq. 4 machinery run on the
        # background-conditioned TPM (the weight degenerates to exactly 1).
        from pyphi.core.tpm.marginalization import (
            _cause_marginal_factored,
            cause_conditioned,
        )

        tpm = substrate.factored_tpm
        background = {2: STATE[2]}
        direct = cause_conditioned(tpm, SYSTEM_NODES, background)
        via_eq4 = _cause_marginal_factored(
            tpm.condition(background), STATE, SYSTEM_NODES
        )
        for i in SYSTEM_NODES:
            assert np.array_equal(direct.factor(i), via_eq4.factor(i))


class TestValueSemantics:
    def test_eq_hash_fingerprint_distinguish_pinned_systems(self, substrate):
        plain = _system(substrate)
        pinned = _system(
            substrate, background_conditioning="CONDITION_CURRENT_STATE"
        )
        assert plain != pinned
        assert plain._fingerprint != pinned._fingerprint
        same = _system(substrate)
        assert plain == same
        assert hash(plain) == hash(same)

    def test_serialization_round_trip_preserves_pin(self, substrate, tmp_path):
        from pyphi import serialize

        pinned = _system(
            substrate, background_conditioning="CONDITION_CURRENT_STATE"
        )
        restored = serialize.loads(serialize.dumps(pinned))
        assert restored == pinned
        assert (
            restored.background_conditioning == "CONDITION_CURRENT_STATE"
        )
        plain = _system(substrate)
        assert serialize.loads(serialize.dumps(plain)) == plain
```

Note: check `pyphi.serialize`'s public dump/load names before running
(`dumps`/`loads` vs `save`/`load` on the `Serializable` mixin) and use
whichever the existing `System` serialization tests use.

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest test/test_background_conditioning.py -v`
Expected: FAIL — `ImportError: cannot import name 'cause_conditioned'` /
`TypeError: __init__() got an unexpected keyword argument
'background_conditioning'`.

- [ ] **Step 4: Implement the kernel operation**

In `pyphi/core/tpm/marginalization.py`, after `cause_marginal` (~line 197):

```python
def cause_conditioned(
    tpm: TPM,
    node_indices: tuple[int, ...],
    background: Mapping[int, int],
) -> CauseMarginals:
    """Cause factors with background units conditioned at their observed
    state — the PyPhi 1.x convention.

    Each output unit ``i`` in ``node_indices`` gets the forward factor
    ``P(s_i,t+1 | s_t)`` with the background input axes fixed at
    ``background`` (kept as size-1 dims), in the same
    ``(*alphabet_sizes, k_i)`` substrate-global axis convention as
    :func:`cause_marginal`. Equivalent to IIT 4.0 Eq. 4 evaluated on the
    background-conditioned TPM, where the background weight is identically
    1. Bayesian inversion and normalization happen downstream in the
    repertoire algebra, exactly as for the marginalized factors.
    """
    if isinstance(tpm, JointTPM):
        tpm = FactoredTPM.from_joint(tpm._inner)
    elif not isinstance(tpm, FactoredTPM):
        tpm = FactoredTPM.from_joint(tpm.to_array())
    conditioned = tpm.condition(dict(background))
    return CauseMarginals(
        {i: conditioned.factor(i) for i in node_indices}
    )
```

- [ ] **Step 5: Implement the `System` changes**

In `pyphi/system.py`:

**(a)** Import the kernel op and the valid-values constant:

```python
from pyphi.conf.formalism import _VALID_BACKGROUND_CONDITIONING

from .core.tpm.marginalization import cause_conditioned as _condition_cause
```

**(b)** Add the field after `external_indices` (~line 63):

```python
    background_conditioning: str | None = field(default=None)
```

and extend the class docstring's field notes with: the field pins this
System to one cause-side background convention; ``None`` (the default)
resolves ``config.formalism.iit.background_conditioning`` at compute time.

**(c)** Validate in `__post_init__` (before the `validate_system_states`
block):

```python
        if (
            self.background_conditioning is not None
            and self.background_conditioning
            not in _VALID_BACKGROUND_CONDITIONING
        ):
            raise ValueError(
                f"background_conditioning={self.background_conditioning!r} "
                f"not in {sorted(_VALID_BACKGROUND_CONDITIONING)} (or None)"
            )
```

**(d)** Add the field to `__eq__` (one more `and` clause) and to the
`__hash__` tuple.

**(e)** Extend `_fingerprint` (keeps its "serializes exactly what `__eq__`
compares" contract; note in the docstring that changing the fingerprint
orphans old disk-cache entries, which the key-addressed disk cache handles
by design):

```python
        h.update(repr(self.background_conditioning).encode())
```

**(f)** Resolution helper (near the other cached cheap properties):

```python
    def _resolved_background_conditioning(self) -> str:
        """The cause-side background convention in effect for this System:
        the instance pin when set, else the live config value."""
        if self.background_conditioning is not None:
            return self.background_conditioning
        from pyphi.conf import config as _config

        return _config.formalism.iit.background_conditioning
```

**(g)** Replace the `cause_marginal` cached property with per-convention
dispatch:

```python
    @cached_property
    def _cause_marginals(self) -> dict[str, CauseMarginals]:
        """Per-convention cause factors, computed on demand."""
        return {}

    @property
    def cause_marginal(self) -> CauseMarginals:
        """Per-system-unit cause factors under the active background
        convention: IIT 4.0 Eq. 4 marginalization, or the background
        conditioned at its observed state (``CONDITION_CURRENT_STATE``).
        """
        convention = self._resolved_background_conditioning()
        if convention not in self._cause_marginals:
            if convention == "CONDITION_CURRENT_STATE":
                external_state = utils.state_of(
                    self.external_indices, self.state
                )
                background = dict(
                    zip(self.external_indices, external_state, strict=True)
                )
                marginals = _condition_cause(
                    self._typed_tpm, self.node_indices, background
                )
            else:
                marginals = _marginalize_cause(
                    self._typed_tpm, self.state, self.node_indices
                )
            self._cause_marginals[convention] = marginals
        return self._cause_marginals[convention]
```

**(h)** Same treatment for `nodes`; `_index2node` becomes a plain property
(rebuilt per access — only touched on kernel-cache misses):

```python
    @cached_property
    def _nodes_by_convention(self) -> dict[str, Any]:
        return {}

    @property
    def nodes(self) -> Any:
        from pyphi.node import generate_nodes

        convention = self._resolved_background_conditioning()
        if convention not in self._nodes_by_convention:
            self._nodes_by_convention[convention] = generate_nodes(
                self.cause_marginal,
                self.effect_marginal,
                self.cm,
                self.state,
                self.node_indices,
                self.node_labels,
            )
        return self._nodes_by_convention[convention]

    @property
    def _index2node(self) -> dict[int, Any]:
        return {node.index: node for node in self.nodes}
```

**(i)** `proper_cause_marginal` becomes a plain `@property` (body unchanged;
it derives from the now-convention-aware `cause_marginal` and is only used
for HIGH-verbosity display). `proper_effect_marginal`, `effect_marginal`,
`null_distinction` stay cached (convention-independent; see Background
fact 6).

**(j)** Update `apply_cut`'s shared-cache name list — the marginals dict is
shared (marginals are cut-independent per the existing docstring; sharing
the dict object lets a convention entry computed on either instance serve
both):

```python
        _ = self.cause_marginal
        _ = self.effect_marginal
        new = replace(self, partition=partition)
        for name in (
            "_typed_tpm",
            "_cause_marginals",
            "effect_marginal",
            "proper_effect_marginal",
        ):
```

(`dataclasses.replace` carries the `background_conditioning` field to the
cut instance automatically. `nodes` is intentionally not shared — it embeds
the cut `cm`.)

- [ ] **Step 6: Key the kernel memo cache by the resolved convention**

In `pyphi/core/repertoire_algebra.py`, `_memoize`'s wrapper (~line 57):

```python
    @wraps(fn)
    def wrapper(cs: Any, *args: Any) -> Any:
        fp = cs._fingerprint
        cache.observe(cs, fp)
        key_args = (cs._resolved_background_conditioning(), *args)
        return cache.get_or_compute(fp, key_args, lambda: fn(cs, *args))
```

and note in `_memoize`'s docstring that keys carry the resolved background
convention so cause-side entries never cross conventions (effect-side
entries are duplicated per convention, which only costs anything when a
process actually flips the option).

If any test later fails with `AttributeError:
_resolved_background_conditioning` for a non-`System` first argument (e.g. a
macro wrapper), add the method (or a delegation entry) to that type rather
than weakening the wrapper with `getattr` defaults — silent fallback here
would reintroduce the staleness bug. Task 3 handles the one known case
(`TransitionSystem`).

- [ ] **Step 7: Serialization**

In `pyphi/serialize/schema.py`, append to `SystemSchema` (defaulted fields
must come last in a msgspec Struct, so old payloads decode with `None`):

```python
class SystemSchema(msgspec.Struct, frozen=True, tag="system"):
    substrate: SubstrateSchema
    state: tuple[int, ...]
    node_indices: tuple[int, ...]
    partition: PartitionSchema
    external_indices: tuple[int, ...]
    background_conditioning: str | None = None
```

In `pyphi/serialize/convert.py` `_register_system`, thread the field through
both the encoder and decoder lambdas.

- [ ] **Step 8: Run the new tests; then the fast regression lane**

Run: `uv run pytest test/test_background_conditioning.py -v`
Expected: all PASS.

Then confirm the marginalized path is untouched:
`uv run pytest test/integration/test_golden_regression.py test/test_system_small_phi.py test/formalism/test_complexes.py -q`
Expected: all PASS with **zero changes** — the default convention runs the
identical code, and the preset has not flipped yet. Any golden movement here
is a bug in this task.

Also run the protocol/surface and serialization suites (the `System` field
may need to be listed wherever the public surface is asserted — see
`pyphi/protocols.py` `PUBLIC_SYSTEM_ATTRS` and any `__getattr__` allow-list
tests): `uv run pytest test/test_serialization_surface.py test/models/ -q`
(adjust to the actual serialization test file names; find them with
`grep -rln "SystemSchema\|PUBLIC_SYSTEM_ATTRS" test/`).

- [ ] **Step 9: Changelog fragment and commit**

```bash
cat > changelog.d/system-background-conditioning-field.feature.md <<'EOF'
`System` accepts `background_conditioning` (default `None` = follow
`config.formalism.iit.background_conditioning` at compute time) to pin an
instance to one cause-side background convention. Cause factors, nodes, and
kernel cache entries are keyed per convention, so config overrides apply to
already-constructed systems.
EOF
git add pyphi/core/tpm/marginalization.py pyphi/system.py pyphi/core/repertoire_algebra.py pyphi/serialize/schema.py pyphi/serialize/convert.py test/example_substrates.py test/test_background_conditioning.py changelog.d/system-background-conditioning-field.feature.md
git commit -m "Add conditioned-background cause factors behind background_conditioning"
```

---

### Task 3: Insulate actual causation

`TransitionSystem` (`pyphi/actual.py:121`) builds its underlying `System`
with AC's own external-indices convention and computes cause repertoires
through the shared kernel. The AC background rule is governed by
`ActualCausationConfig.background_scheme`, not the IIT knob — and
`presets.iit3` (which flips the knob in Task 4) is also the config context
for AC 2019 analyses. Pin the underlying System to marginalization so AC
results are identical under both knob settings.

**Files:**
- Modify: `pyphi/actual.py` (`_underlying_system`, `_DELEGATED_TO_SYSTEM`)
- Test: `test/test_background_conditioning.py` (extend)

**Interfaces:**
- Consumes: `System.background_conditioning`,
  `System._resolved_background_conditioning` (Task 2).
- Produces: no new API. `TransitionSystem` delegates
  `_resolved_background_conditioning` (the kernel memo wrapper may receive a
  `TransitionSystem` directly on some AC paths).

- [ ] **Step 1: Write the failing test**

Append to `test/test_background_conditioning.py`:

```python
class TestActualCausationInsulation:
    def test_ac_account_invariant_under_the_knob(self, substrate):
        # A transition over a proper subset of the substrate: background
        # unit C is outside the transition, the exact situation where the
        # knob would otherwise leak into AC cause repertoires.
        from pyphi import actual

        def account_alphas():
            transition = actual.Transition(
                substrate,
                before_state=(1, 0, 0),
                after_state=(0, 1, 0),
                cause_indices=(0, 1),
                effect_indices=(0, 1),
            )
            account = actual.account(transition)
            return sorted(
                (link.direction, tuple(link.mechanism), float(link.alpha))
                for link in account
            )

        baseline = account_alphas()
        with config.override(
            background_conditioning="CONDITION_CURRENT_STATE"
        ):
            flipped = account_alphas()
        assert flipped == baseline
        assert len(baseline) > 0
```

Adjust the `Transition` constructor call and the `account` iteration to the
actual API in `pyphi/actual.py` (read the module's public entry points and
an existing AC test in `test/formalism/test_ac_formalism.py` first; the
essential assertions — nonempty account, α multiset invariant under the
knob — must survive any adaptation). If the chosen after-state is rejected
by the Realization check, pick any after-state with
`p(after | before) > 0` for this substrate.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest test/test_background_conditioning.py -v -k ac_account`
Expected: FAIL — the α values differ between the two settings (the knob
currently leaks into the underlying System's cause factors).

- [ ] **Step 3: Implement**

In `pyphi/actual.py` `_underlying_system` (~line 201), pass the pin:

```python
            return System(
                substrate=self.substrate,
                state=self.state,
                node_indices=self.node_indices,
                partition=self.partition,
                external_indices=external,
                background_conditioning="CAUSAL_MARGINALIZATION",
            )
```

and document in the `TransitionSystem` docstring that the AC background rule
is set by `ActualCausationConfig.background_scheme`; the underlying System
is pinned so the IIT-formalism `background_conditioning` option does not
apply to actual-causation analyses.

Add `"_resolved_background_conditioning"` to `_DELEGATED_TO_SYSTEM`
(~line 80-118) so kernel calls that receive the `TransitionSystem` itself
resolve through the pinned underlying System.

- [ ] **Step 4: Run AC suites**

Run: `uv run pytest test/test_background_conditioning.py test/formalism/test_ac_formalism.py test/test_actual*.py -q --co -q 2>/dev/null | head` to
discover the actual AC test paths, then run them.
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add pyphi/actual.py test/test_background_conditioning.py
git commit -m "Pin actual-causation systems to causal marginalization"
```

---

### Task 4: Flip `presets.iit3` and update the affected pins

The preset selects the 1.x convention. Exactly one committed test module
pins proper-subset IIT 3.0 values that move (verified inventory —
Background facts 3-4): `test/formalism/test_complexes.py`. The committed
golden `basic_subset_iit3_emd` does **not** move (fact 3). The new expected
values are genuine PyPhi 1.2.0 outputs (fact 4), so the updates restore
published-1.x behavior rather than pinning novel numbers.

**Files:**
- Modify: `pyphi/conf/presets.py` (`iit3` preset + knob-listing comment)
- Modify: `test/formalism/test_complexes.py` (two expected values + a
  companion marginalized-convention test)
- Create: `changelog.d/iit3-preset-background-conditioning.change.md`

**Interfaces:**
- Consumes: everything from Tasks 1-3.
- Produces: `presets.iit3["iit"].background_conditioning ==
  "CONDITION_CURRENT_STATE"` (inherited by `test/conftest.py::IIT_3_CONFIG`
  and `test/golden/zoo.py::IIT_3_CONFIG`, both sourced from the preset).

- [ ] **Step 1: Update the preset**

In `pyphi/conf/presets.py`, add to the `iit3` `IITConfig(...)` call:

```python
        # Background units are fixed at their observed current state on the
        # cause side (the PyPhi 1.x / post-2014-literature convention), so
        # proper-subset analyses reproduce published IIT 3.0 results. The
        # global default is IIT 4.0 Eq. 4 causal marginalization.
        background_conditioning="CONDITION_CURRENT_STATE",
```

- [ ] **Step 2: Run the complexes tests to see the expected failures**

Run: `uv run pytest test/formalism/test_complexes.py -q`
Expected: exactly `test_irreducible_sias_standard` and
`test_all_sias_standard` FAIL, with the `(0, 2)` candidate's φ now 1.0.
Any *other* failure in this file means the seam leaked somewhere
unexpected — stop and diagnose before touching expectations.

- [ ] **Step 3: Update the pins and add the companion test**

In `test/formalism/test_complexes.py`:

- `test_irreducible_sias_standard`: change `((0, 2), 0.5)` to
  `((0, 2), 1.0)` and update the docstring's value list; note in the
  docstring that these are the PyPhi 1.x values (the `iit3` preset
  conditions background units at their current state).
- `test_all_sias_standard`: change the expected list to
  `[0.0, 0.0, 1.0, 1.0, 2.3125]` (docstring likewise). The "exactly three
  irreducible" assertion is unchanged.
- Add a companion test in the same class pinning the marginalized values so
  both conventions stay covered at the complex-search level:

```python
    def test_irreducible_sias_standard_marginalized_background(self, s):
        """The same complex search under Eq. 4 causal marginalization: the
        (0, 2) candidate's φ differs from the conditioned convention (the
        (1, 2) candidate's value coincides across conventions)."""
        with config.override(
            background_conditioning="CAUSAL_MARGINALIZATION"
        ):
            sias = s.substrate.irreducible_sias(s.state)
        nodes_and_phis = {
            c.node_indices: float(c.phi) for c in sias
        }
        assert nodes_and_phis[(0, 1, 2)] == pytest.approx(2.3125)
        assert nodes_and_phis[(1, 2)] == pytest.approx(1.0)
        assert nodes_and_phis[(0, 2)] == pytest.approx(0.5)
```

(The class's autouse fixture already applies `IIT_3_CONFIG`; the nested
override flips only the knob.)

- [ ] **Step 4: Sweep the IIT 3.0 surface**

Run the golden suite and the IIT 3.0-heavy files:

```bash
uv run pytest test/integration/test_golden_regression.py -q
uv run pytest test/formalism/ test/integration/test_paper_reproduction.py test/formalism/test_iit3_divergence_audit.py -q
uv run pytest test/ -q -m "not slow"
```

Expected: all PASS. In particular `basic_subset_iit3_emd` must pass
**without regeneration** (fact 3: every repertoire on that subset coincides
across conventions). If any golden fails here, do not regenerate it —
diagnose first; the only legitimate movers are proper-subset IIT 3.0
pins, and the verified inventory says there are none besides
`test_complexes.py`.

- [ ] **Step 5: Changelog fragment and commit**

```bash
cat > changelog.d/iit3-preset-background-conditioning.change.md <<'EOF'
`presets.iit3` now sets `background_conditioning="CONDITION_CURRENT_STATE"`,
so IIT 3.0 analyses of systems that are proper subsets of their substrate
reproduce published PyPhi 1.x results (background units fixed at their
observed state on the cause side, rather than causally marginalized per
IIT 4.0 Eq. 4). Full-substrate analyses are unaffected — the conventions
coincide when there is no background.
EOF
git add pyphi/conf/presets.py test/formalism/test_complexes.py changelog.d/iit3-preset-background-conditioning.change.md
git commit -m "Select current-state background conditioning in the IIT 3.0 preset"
```

---

### Task 5: 1.x-oracle anchoring and discriminating goldens

Commit the PyPhi 1.2.0 ground truth into `test/data/iit3-canonical/` with
its reproducer script (mirroring the `gen_iit3_emd_oracle.py` convention),
an integration test asserting the library against it under **both**
conventions (repertoire level and end-to-end Φ), and two golden fixtures on
the discriminating substrate so the full golden harness (repertoires,
mechanism MIPs, SIA) covers both settings permanently.

**Files:**
- Create: `scripts/gen_iit3_background_oracle.py`
- Create: `test/data/iit3-canonical/background_conditioning_oracle.json`
  (generated by the script; expected contents below)
- Modify: `test/data/iit3-canonical/README.md` (document the new file)
- Modify: `test/golden/zoo.py` (two fixtures)
- Create: `test/data/golden/v1/noisy_or_subset_iit3_emd.{json,npz}` and
  `..._marginalized.{json,npz}` (via `--regenerate-golden`)
- Test: `test/integration/test_background_conditioning_oracle.py` (create)

**Interfaces:**
- Consumes: `noisy_or_background_substrate` (Task 2), the flipped preset
  (Task 4).
- Produces: golden fixtures `noisy_or_subset_iit3_emd`,
  `noisy_or_subset_iit3_emd_marginalized` (auto-collected by
  `test_golden_regression.py` and by `test_iit3_divergence_audit.py`'s
  `IIT3_FIXTURES` name filter).

- [ ] **Step 1: Write the oracle reproducer script**

Create `scripts/gen_iit3_background_oracle.py`. It targets the PyPhi 1.x
API and is run only in the isolated 1.2.0 venv (see Background for the
recipe); it lives under `scripts/` so the 2.0 suite never imports it.

```python
"""Generate the PyPhi 1.x reference for cause-side background conditioning.

Reproducer for
``test/data/iit3-canonical/background_conditioning_oracle.json``, consumed
by ``test/integration/test_background_conditioning_oracle.py``.

Records, from a genuine PyPhi 1.2.0 install:

1. The cause repertoire of mechanism {A} over purview {B} for the
   proper-subset system S={A,B} of the 3-unit noisy-OR substrate
   (background W={C}, state (1,0,0)) — the value discriminates
   current-state conditioning of W (1.x) from IIT 4.0 Eq. 4 causal
   marginalization, and both predictions are stored alongside the
   observation.
2. The end-to-end IIT 3.0 SIA phi for that system.
3. SIA phi for every proper subset of the ``basic`` example network in
   state (1,0,0) — independent anchors for the complex-search values.

Control: reproduces the anchored ``basic`` full-substrate value
(2.3125 = 37/16) before the oracle is trusted.

Environment setup (isolated; does not touch the project venv)::

    VENV=/tmp/pyphi-1x-oracle/.venv
    uv venv --python 3.9 "$VENV"
    VIRTUAL_ENV="$VENV" uv pip install "pyphi==1.2.0"
    "$VENV/bin/python" scripts/gen_iit3_background_oracle.py \
        > test/data/iit3-canonical/background_conditioning_oracle.json

Run from a directory with no ``pyphi_config.yml`` so 1.x uses defaults plus
the flags set below (same flags as ``gen_iit3_emd_oracle.py``).
"""
# This reproducer targets the PyPhi 1.x API (pyphi.Network / Subsystem /
# compute), which does not exist in the 2.0 package, so pyright cannot
# resolve it here.
# pyright: reportCallIssue=false, reportAttributeAccessIssue=false

import json
import sys

import numpy as np

import pyphi

pyphi.config.PARALLEL_CONCEPT_EVALUATION = False
pyphi.config.PARALLEL_CUT_EVALUATION = False
pyphi.config.PARALLEL_COMPLEX_EVALUATION = False
pyphi.config.PROGRESS_BARS = False
pyphi.config.MEASURE = "EMD"
pyphi.config.USE_SMALL_PHI_DIFFERENCE_FOR_CES_DISTANCE = False
pyphi.config.PARTITION_TYPE = "BI"
pyphi.config.PRECISION = 6
pyphi.config.CUT_ONE_APPROXIMATION = False
pyphi.config.PICK_SMALLEST_PURVIEW = False


def p_a_on(b, c):
    return 0.9 if (b or c) else 0.1


def main():
    # Control: the anchored full-substrate basic value.
    basic = pyphi.examples.basic_network()
    control = float(
        pyphi.compute.sia(
            pyphi.Subsystem(basic, (1, 0, 0), range(basic.size))
        ).phi
    )
    assert abs(control - 2.3125) < 1e-4, f"control failed: {control}"

    # Discriminating substrate: A = noisy OR of (B, C); B' = copy(A);
    # C' = 0.5 with no parents. State-by-node TPM, little-endian rows.
    rows = []
    for c in (0, 1):
        for b in (0, 1):
            for a in (0, 1):
                rows.append((p_a_on(b, c), float(a), 0.5))
    net = pyphi.Network(
        np.array(rows),
        cm=np.array([[0, 1, 0], [1, 0, 0], [1, 0, 0]]),
        node_labels=("A", "B", "C"),
    )
    sub = pyphi.Subsystem(net, (1, 0, 0), (0, 1))

    cause = np.asarray(sub.cause_repertoire((0,), (1,))).squeeze()
    # Predictions under each convention (uniform background prior for Eq. 4).
    w = np.array([0.5 * (p_a_on(0, c) + p_a_on(1, c)) for c in (0, 1)])
    w /= w.sum()
    eq4 = np.array(
        [sum(p_a_on(b, c) * w[c] for c in (0, 1)) for b in (0, 1)]
    )
    eq4 /= eq4.sum()
    legacy = np.array([p_a_on(b, 0) for b in (0, 1)])
    legacy /= legacy.sum()

    effect = np.asarray(sub.effect_repertoire((0,), (1,))).squeeze()
    sia = pyphi.compute.sia(sub)

    basic_subsets = {}
    for nodes in [(0, 1, 2), (1, 2), (0, 2), (0, 1)]:
        s = pyphi.Subsystem(basic, (1, 0, 0), nodes)
        subset_sia = pyphi.compute.sia(s)
        basic_subsets[str(nodes)] = {
            "phi": float(subset_sia.phi),
            "cut": str(subset_sia.cut),
            "n_concepts": len(subset_sia.ces),
        }

    out = {
        "oracle": f"pyphi=={pyphi.__version__}",
        "numpy": np.__version__,
        "python": sys.version,
        "config": {
            "MEASURE": "EMD",
            "PARTITION_TYPE": "BI",
            "PRECISION": 6,
            "PICK_SMALLEST_PURVIEW": False,
            "USE_SMALL_PHI_DIFFERENCE_FOR_CES_DISTANCE": False,
            "CUT_ONE_APPROXIMATION": False,
        },
        "control_basic_sia_phi": control,
        "fixture": {
            "state": [1, 0, 0],
            "system_nodes": [0, 1],
            "background_nodes": [2],
            "mechanism": [0],
            "purview": [1],
        },
        "cause_repertoire": {
            "observed": [float(x) for x in cause],
            "predicted_eq4_marginalized": [float(x) for x in eq4],
            "predicted_legacy_conditioned": [float(x) for x in legacy],
        },
        "effect_repertoire": {"observed": [float(x) for x in effect]},
        "sia": {
            "phi": float(sia.phi),
            "cut": str(sia.cut),
            "ces_size": len(sia.ces),
        },
        "basic_network_subsets": basic_subsets,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate and cross-check the oracle JSON**

Run the script in the isolated 1.2.0 venv (recipe in Background), writing
`test/data/iit3-canonical/background_conditioning_oracle.json`. Verify the
generated values against the pre-verified expectations before committing —
a mismatch means the oracle harness is mis-invoked, not that these numbers
are wrong:

- `control_basic_sia_phi`: 2.3125
- `cause_repertoire.observed`: `[0.1, 0.9]`
- `cause_repertoire.predicted_eq4_marginalized`:
  `[0.40566037735849053, 0.5943396226415094]`
- `effect_repertoire.observed`: `[0.0, 1.0]`
- `sia.phi`: 0.72, `sia.ces_size`: 2
- `basic_network_subsets`: `(0, 1, 2)` φ 2.3125 (4 concepts);
  `(1, 2)` φ 1.0; `(0, 2)` φ 1.0 (cut `[A] ──/ /──➤ [C]`, 2 concepts);
  `(0, 1)` φ 0.0

Add a short entry for the file to `test/data/iit3-canonical/README.md`
(what it anchors, which script reproduces it, which test consumes it).

- [ ] **Step 3: Write the integration test**

Create `test/integration/test_background_conditioning_oracle.py`:

```python
"""Anchor both background conventions to the PyPhi 1.2.0 oracle.

The oracle JSON is generated by ``scripts/gen_iit3_background_oracle.py``
from a genuine PyPhi 1.2.0 install; see ``test/data/iit3-canonical/``.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import pyphi
import pyphi.formalism.iit3 as iit3
from pyphi import config
from pyphi.conf import presets
from pyphi.system import System

from example_substrates import noisy_or_background_substrate

pytestmark = pytest.mark.emd

ORACLE_PATH = (
    Path(__file__).parent.parent
    / "data"
    / "iit3-canonical"
    / "background_conditioning_oracle.json"
)


@pytest.fixture(scope="module")
def oracle():
    with ORACLE_PATH.open() as f:
        return json.load(f)


@pytest.fixture()
def system():
    return System(
        noisy_or_background_substrate(), (1, 0, 0), node_indices=(0, 1)
    )


class TestRepertoireLevel:
    def test_conditioned_cause_repertoire_matches_1x(self, oracle, system):
        with config.override(**presets.iit3):
            rep = system.cause_repertoire((0,), (1,)).squeeze()
        assert rep == pytest.approx(
            oracle["cause_repertoire"]["observed"]
        )

    def test_marginalized_cause_repertoire_matches_eq4(self, oracle, system):
        with config.override(**presets.iit3):
            with config.override(
                background_conditioning="CAUSAL_MARGINALIZATION"
            ):
                rep = system.cause_repertoire((0,), (1,)).squeeze()
        assert rep == pytest.approx(
            oracle["cause_repertoire"]["predicted_eq4_marginalized"]
        )

    def test_effect_repertoire_matches_both_conventions(self, oracle, system):
        expected = oracle["effect_repertoire"]["observed"]
        for value in ("CAUSAL_MARGINALIZATION", "CONDITION_CURRENT_STATE"):
            with config.override(background_conditioning=value):
                rep = system.effect_repertoire((0,), (1,)).squeeze()
            assert rep == pytest.approx(expected)


class TestEndToEnd:
    def test_conditioned_phi_matches_1x(self, oracle, system):
        with config.override(**presets.iit3, progress_bars=False):
            phi = float(iit3.sia(system).phi)
        assert phi == pytest.approx(oracle["sia"]["phi"], abs=1e-6)

    def test_marginalized_phi_differs_and_is_pinned(self, system):
        with config.override(**presets.iit3, progress_bars=False):
            with config.override(
                background_conditioning="CAUSAL_MARGINALIZATION"
            ):
                phi = float(iit3.sia(system).phi)
        assert phi == pytest.approx(0.41607, abs=1e-6)

    def test_basic_subset_phis_match_1x(self, oracle):
        basic = pyphi.examples.basic_substrate()
        with config.override(**presets.iit3, progress_bars=False):
            for key, expected in oracle["basic_network_subsets"].items():
                nodes = tuple(
                    int(x) for x in key.strip("()").split(",") if x.strip()
                )
                system = System(basic, (1, 0, 0), node_indices=nodes)
                phi = float(iit3.sia(system).phi)
                assert phi == pytest.approx(expected["phi"], abs=1e-6), key
```

(Check whether nested `config.override` calls compose as written; if the
inner flat write is rejected while the outer holds the wholesale `iit`
sub-config, use the `dataclasses.replace(presets.iit3["iit"], ...)` pattern
from `test/golden/zoo.py` instead.)

- [ ] **Step 4: Run the integration tests**

Run: `uv run pytest test/integration/test_background_conditioning_oracle.py -v`
Expected: all PASS.

- [ ] **Step 5: Add the golden fixtures**

In `test/golden/zoo.py`, import the shared factory
(`from test.example_substrates import noisy_or_background_substrate` —
match how other `test.` imports appear in the goldens package) and append to
the "Targeted extra fixtures" section of `_make_fixtures()`:

```python
    # Proper-subset IIT 3.0 under each background convention. The noisy
    # background parent C makes the cause side diverge between conventions
    # (deterministic substrates like basic's (1,2) subset mask it), so this
    # pair pins both semantics through every layer of the harness.
    fixtures.append(
        GoldenFixture(
            name="noisy_or_subset_iit3_emd",
            description="Noisy-OR 3-unit substrate, system {A,B}, background "
            "{C}, state (1,0,0). IIT 3.0 preset semantics: background "
            "conditioned at its current state (PyPhi 1.x); SIA phi = 0.72 "
            "matches the genuine 1.2.0 oracle.",
            config_overrides=IIT_3_CONFIG,
            substrate_factory=noisy_or_background_substrate,
            state=(1, 0, 0),
            node_indices=(0, 1),
            skip_layers=SKIP_FOR_IIT_3,
        )
    )
    fixtures.append(
        GoldenFixture(
            name="noisy_or_subset_iit3_emd_marginalized",
            description="Same system under IIT 4.0 Eq. 4 causal "
            "marginalization of the background (SIA phi = 0.41607); "
            "companion to noisy_or_subset_iit3_emd.",
            config_overrides={
                **{k: v for k, v in IIT_3_CONFIG.items() if k != "iit"},
                "iit": replace(
                    IIT_3_CONFIG["iit"],
                    background_conditioning="CAUSAL_MARGINALIZATION",
                ),
            },
            substrate_factory=noisy_or_background_substrate,
            state=(1, 0, 0),
            node_indices=(0, 1),
            skip_layers=SKIP_FOR_IIT_3,
        )
    )
```

- [ ] **Step 6: Generate the goldens and verify the whole harness**

```bash
uv run pytest test/integration/test_golden_regression.py --regenerate-golden -k noisy_or
uv run pytest test/integration/test_golden_regression.py -q
uv run pytest test/formalism/test_iit3_divergence_audit.py -q
```

Expected: the two new fixture pairs are written under
`test/data/golden/v1/`; the full golden run passes; the divergence-audit
module (which auto-collects fixtures named `*iit3*`) passes with the new
fixtures included. Sanity-check the stored JSONs: `sia.phi` 0.72 and
0.41607 respectively (field name per the harness's SIA layer).

- [ ] **Step 7: Commit**

```bash
git add scripts/gen_iit3_background_oracle.py test/data/iit3-canonical/background_conditioning_oracle.json test/data/iit3-canonical/README.md test/integration/test_background_conditioning_oracle.py test/golden/zoo.py test/data/golden/v1/noisy_or_subset_iit3_emd*
git commit -m "Anchor background conventions to the PyPhi 1.2.0 oracle with goldens"
```

---

### Task 6: Documentation close-out and full verification

**Files:**
- Modify: `pyphi/conf/formalism.py` (`IITConfig` docstring)
- Modify: `CLAUDE.md` (configuration options list)
- Modify: `ROADMAP.md` (Status Dashboard row)

**Interfaces:** none (documentation only).

- [ ] **Step 1: Document the option on `IITConfig`**

Extend the `IITConfig` class docstring (`pyphi/conf/formalism.py`) with a
section — this is the canonical user-facing documentation of the option
(the Sphinx configuration page renders `pyphi.conf` docstrings):

```
    Background conditioning (``background_conditioning``)
        How substrate units outside the candidate system (the background)
        enter cause repertoires when the system is a proper subset of its
        substrate:

        - ``"CAUSAL_MARGINALIZATION"`` (default): the background past is
          causally marginalized conditional on the current state — the
          "extended background" of IIT 4.0 (Albantakis et al. 2023, Eq. 4).
          Definitional for IIT 4.0.
        - ``"CONDITION_CURRENT_STATE"``: the background is fixed at its
          observed current state — the convention of PyPhi 1.x and the
          post-2014 IIT 3.0 literature. Selected by ``presets.iit3`` so
          that IIT 3.0 analyses of proper-subset systems reproduce
          published results.

        The IIT 3.0 paper itself (Oizumi et al. 2014, Box 1) fixes the
        background at its actual *past* state on the cause side. That
        convention requires the past state as an input, which no PyPhi
        version has ever taken, and it is not implemented.

        The effect side conditions the background at its current state
        under every convention, and full-substrate systems have no
        background, so the setting affects neither. Actual-causation
        analyses are unaffected: the AC background rule is set by
        ``ActualCausationConfig.background_scheme``.
```

- [ ] **Step 2: Update the CLAUDE.md configuration list**

In `CLAUDE.md`, under "Important Configuration Options" →
"Computational Behavior (``config.formalism.iit``)", add:

```markdown
- **`background_conditioning`**: cause-side background handling —
  ``"CAUSAL_MARGINALIZATION"`` (IIT 4.0 Eq. 4; default) or
  ``"CONDITION_CURRENT_STATE"`` (PyPhi 1.x convention; set by
  ``presets.iit3``). Only affects proper-subset systems.
```

- [ ] **Step 3: Add a ROADMAP dashboard row**

In `ROADMAP.md`'s Status Dashboard, add a row following the neighboring
rows' column structure:

```markdown
| Background-conditioning knob | ✅ landed | — | `formalism.iit.background_conditioning`: cause-side background handling as an explicit axis — `CAUSAL_MARGINALIZATION` (IIT 4.0 Eq. 4, global default) vs `CONDITION_CURRENT_STATE` (PyPhi 1.x; selected by `presets.iit3` for paper-era fidelity on proper-subset systems). Anchored to a genuine PyPhi 1.2.0 oracle (`test/data/iit3-canonical/background_conditioning_oracle.json`) with discriminating goldens under both settings; AC pinned to marginalization. The 2014 paper's past-state convention was never implemented in any PyPhi version and remains documentation-only. |
```

- [ ] **Step 4: Full test suite (no path argument — includes the doctest sweep)**

Run: `uv run pytest -x -q`
Expected: all pass. Then the slow tier over the goldens:
`uv run pytest test/integration/test_golden_regression.py -q --slow`
(only if the repo's slow-marker flag is spelled that way — check
`test/conftest.py`; otherwise use the documented opt-in).

If any doctest or unrelated-looking test fails, diagnose before touching
anything — other sessions may have concurrent working-tree changes; only
fix failures traceable to this plan's commits.

- [ ] **Step 5: Pre-commit hooks over the changed files**

Run: `uv run pre-commit run --files $(git diff --name-only $(git merge-base HEAD <base-branch>) | tr '\n' ' ')` — substitute the branch this worktree was created from.
Expected: all hooks pass (ruff, pyright, file checks).

- [ ] **Step 6: Commit the documentation updates**

```bash
git add pyphi/conf/formalism.py CLAUDE.md ROADMAP.md
git commit -m "Document the background-conditioning conventions"
```

---

## Self-review notes

- **Settled-design coverage:** option + values (Task 1), preset selection
  (Task 4), discriminating goldens on both settings over the proper-subset
  fixture (Task 5), 2014-convention documented-not-built (Task 6 docstring;
  no code path accepts a past state anywhere in the plan).
- **Byte-identical marginalized path:** the `CAUSAL_MARGINALIZATION` branch
  of `System.cause_marginal` calls exactly the pre-existing
  `_marginalize_cause`; Task 2 Step 8 and Task 4 Step 4 verify the golden
  suite twice (before and after the preset flip), and fact 3 predicts
  `basic_subset_iit3_emd` passes unregenerated — treat any deviation as a
  bug, not a regeneration prompt.
- **Effect side:** pinned convention-invariant at the repertoire level in
  both the unit tests (Task 2) and the oracle-anchored integration tests
  (Task 5), matching the oracle's observed `[0.0, 1.0]`.
- **Cache correctness is the riskiest surface.** Three layers are each
  covered by a dedicated test: per-System dicts
  (`test_same_system_object_respects_config_flip`), the shared-marginal
  copy in `apply_cut` (`test_apply_cut_shares_marginals_across_conventions`),
  and the kernel memo key (same flip test exercises it — the memoized
  single-node entries would otherwise be served stale). The disk cache is
  safe by construction (config digest) and the `_fingerprint` extension
  keeps pinned/unpinned systems keyed apart there too.
- **Known blast radius was measured, not assumed:** the only committed pins
  that move are the two in `test/formalism/test_complexes.py`, and their new
  values are genuine PyPhi 1.2.0 outputs (basic `(0,2)`: 1.0). The
  `basic (1,2)` subset coincides across conventions (all repertoires
  verified equal), which is why the committed golden stands.
- **AC insulation** is tested on a proper-subset transition — the exact
  configuration where the leak would occur — and lands *before* the preset
  flip so the suite stays green at every commit boundary.
- **Names used across tasks are consistent:** `background_conditioning`
  (config field, System field, schema field), `CAUSAL_MARGINALIZATION` /
  `CONDITION_CURRENT_STATE`, `cause_conditioned`,
  `_resolved_background_conditioning`, `_cause_marginals`,
  `_nodes_by_convention`, `noisy_or_background_substrate`,
  `noisy_or_subset_iit3_emd[_marginalized]`.
- **Executor judgment points** (flagged inline): the `pyphi.serialize`
  dump/load spelling (Task 2 Step 2), the AC `Transition`/`account` API
  shape (Task 3 Step 1), nested-override composition vs the `replace`
  pattern (Task 5 Step 3), and any non-`System` object reaching the kernel
  memo wrapper (Task 2 Step 6 — fail loudly, then delegate, never default).
