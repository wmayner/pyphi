# Intrinsic-Units Macro Machinery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Marshall et al. 2024 intrinsic-units evaluation machinery — `MacroUnit` value objects, the four-step macro TPM construction (Eqs 26-40), and a `MacroSystem` that the existing IIT 4.0 pipeline consumes unchanged — replacing the legacy `pyphi/macro.py` outright.

**Architecture:** New `pyphi/macro/` package (`units.py`, `tpm.py`, `system.py`). `MacroSystem` subclasses `System` over a synthetic macro `Substrate` built from the construction's effect TPM, overriding only the cause-side TPM properties (the single point where the macro formalism cannot be derived from a forward TPM). All index math is mixed-radix keyed to per-constituent alphabet tuples; binary is enforced only by validation.

**Tech Stack:** numpy, `FactoredTPM`, `Substrate.from_factored`, the `System`/`SystemPublicInterface` seam, pytest.

**Spec:** `docs/superpowers/specs/2026-06-10-intrinsic-units-machinery-design.md`

---

## Verified groundwork (do not re-derive)

These facts were established by numerical experiment during planning
(scratch scripts `/tmp/macro_construction_check.py`,
`/tmp/macro_phi_check.py`); the test values below are copied from those
runs.

1. **Config mapping (spec obligation) — RESOLVED.** Under
   `config.override(**presets.iit4_2023)` (bare preset, all other
   formalism fields at 2.0 defaults), the 2.0 pipeline reproduces the
   authors' committed results computed with old pyphi (`941c65a`,
   `SYSTEM_PARTITION_TYPE: SET_UNI/BI`): cg micro panel and bbx micro
   panel to ≤2e-16, and both macro φ_s values **bit-for-bit** when fed
   the same macro TPMs. `DIRECTED_SET_PARTITION` ≡ legacy `SET_UNI/BI`.

2. **Direction semantics of 𝒯_c (Eqs 32-34).** Both macro TPMs are
   *forward* TPMs built from the same chained, discounted probabilities;
   they differ **only** in background weighting: effect = delta on the
   current background micro state (Eq 33); cause = Bayes posterior over
   the background state one update before the earliest state of the
   current window, uniform prior (Eq 34). With empty background,
   𝒯_c = 𝒯_e exactly. This is the exact macro generalization of 2.0's
   `_cause_tpm_factored` / `_effect_tpm_factored` (IIT 4.0 Eq 4), so
   identity macroing of a *proper subset* must reproduce
   `System.proper_cause_tpm` / `proper_effect_tpm` exactly — that is the
   anchor for the background path.

3. **Eq 28 noising is invisible at τ=1** (noised rows sum out of the
   Eq 35 preimage sums); it bites only for τ ≥ 2 via feedback into U^J.

4. **r(u^S, s) (Eqs 37-39) is a counting proportion** (uniform over
   preimage sequences, NOT probability-weighted) and **factorizes per
   unit** because the U^J are disjoint (Eq 18). Each unit's factor is
   the proportion of its g_J-preimage sequences ending in each final
   U^J state.

5. **The four-step construction is validated end-to-end:**
   - **bbx (Example 2):** construction TPM ≡ authors' computed TPM to
     1.1e-16; φ_s through the 2.0 pipeline = **1.1183776016500528**
     (bit-for-bit equal to their committed value).
   - **cg (Example 1):** construction TPM =
     `[[0.00683333…, 0.00683333…], [0.0256, 0.7855], [0.7855, 0.0256],
     [0.9216, 0.9216]]` (rows exact: 0.0615/9, 0.0256, 0.7855, 0.96²).
     The authors' committed TPM is **hand-entered** and contains a
     rounding (`0.006833`) and a hand-entry error (`0.9212` where the
     construction provably gives `0.9216`). Their committed
     φ_s = 1.0039763812908649 reproduces bit-for-bit from their literal
     TPM (config cross-check); the **exact** construction TPM gives
     φ_s = **1.0040208141253277** (our recorded construction golden).
   - **min (authors' minimal example):** their hand-derived macro entry
     `0.05*0.05 + 2*0.01*0.05/3` equals the construction's
     uniform-preimage average — confirms the r-weight semantics.

6. **Sequence-index identity.** With the pinned truth-table convention
   (within an update, first constituent varies fastest; updates oldest
   first, newest varying slowest), the per-update M-ary mixed radix over
   a sequence equals the flat binary little-endian index over all
   `m·τ` bits. The DP below exploits this: sequence-class indices are
   directly `micro_mapping` indices.

## Standing constraints

- Commits via `git -c commit.gpgsign=false commit`; targeted `git add
  <files>` only. If a commit prints hook output but no commit line, the
  hook reformatted: re-`git add` the same files and commit again.
- Ruff bans: `dict()` calls; Unicode ×/−/– in Python strings and
  docstrings (write "phi", "tau"); E402/PLC0415 (all imports at top,
  tests included); RUF005; SIM117. Imports added per-task (avoid F401
  at intermediate commits).
- `uv run` for everything. Full verification = `uv run pytest` with NO
  path argument. pyphi states are little-endian (first unit varies
  fastest).

## File structure

| File | Action | Responsibility |
|---|---|---|
| `pyphi/macro.py` | **delete** | legacy pre-2024 pipeline (disabled) |
| `pyphi/macro/__init__.py` | create | package exports |
| `pyphi/macro/units.py` | create | `MacroUnit`, mixed-radix helpers, `micro_unit`/`coarse_grain`/`blackbox` constructors |
| `pyphi/macro/tpm.py` | create | four-step construction (Eqs 26-40), `macro_tpms()` |
| `pyphi/macro/system.py` | create | `MacroSystem(System)` + `from_micro` builder |
| `pyphi/validate.py` | edit | remove legacy `time_scale`/`coarse_grain`/`blackbox`/`blackbox_and_coarse_grain` |
| `test/test_macro.py`, `test/test_macro_blackbox.py`, `test/test_macro_system.py`, `test/test_macro_disabled_during_p7_gap.py` | **delete** | legacy tests |
| `test/test_macro_units.py`, `test/test_macro_tpm.py`, `test/test_macro_system.py` | create | new test batteries |
| `test/example_substrates.py`, `test/conftest.py`, `test/test_validate.py`, `test/test_tpm_indices.py` | edit | remove legacy-macro usages |
| `pyphi/substrate.py`, `pyphi/core/tpm/joint_distribution.py` | edit | reword legacy-macro comments |
| `docs/conf.py`, `docs/index.rst`, `docs/api/macro.rst`, `docs/examples/emergence.rst`, `docs/examples/magic_cut.rst` | edit/delete | retire legacy docs |
| `changelog.d/intrinsic-units-macro.feature.md`, `ROADMAP.md` | create/edit | record the change |

Design refinement vs. the spec (functionally equivalent, surfaced for
review): the spec sketches `MacroSystem` as a standalone frozen
dataclass holding the *micro* substrate. During planning we found the
repertoire algebra reads `system.substrate.factored_tpm.alphabet_sizes`,
`system.substrate.node_indices`, and
`system.substrate.potential_purviews(...)` — so the protocol's
`substrate` member must be a **macro-level** Substrate for downstream
consumers to behave. `MacroSystem` therefore subclasses `System` with
`substrate` = synthetic macro Substrate (`Substrate.from_factored(T_e)`,
all-ones cm) and overrides only `cause_tpm`/`proper_cause_tpm`; the
micro substrate, units, and history are extra fields. Every inherited
member is then automatically correct, `apply_cut` works via
`dataclasses.replace`, and zero pipeline changes are needed. The
user-facing constructor is the classmethod
`MacroSystem.from_micro(substrate, units, micro_history, node_labels=None)`.

---

### Task 1: Delete the legacy macro implementation

**Files:**
- Delete: `pyphi/macro.py`, `test/test_macro.py`, `test/test_macro_blackbox.py`, `test/test_macro_system.py`, `test/test_macro_disabled_during_p7_gap.py`, `docs/examples/emergence.rst`, `docs/api/macro.rst`
- Modify: `pyphi/validate.py`, `test/test_validate.py`, `test/example_substrates.py`, `test/conftest.py`, `test/test_tpm_indices.py:28`, `pyphi/substrate.py:250`, `pyphi/core/tpm/joint_distribution.py:454`, `docs/conf.py`, `docs/index.rst`, `docs/examples/magic_cut.rst`, `docs/api/index.rst` (if it lists `macro`)

- [ ] **Step 1: Delete files**

```bash
git rm pyphi/macro.py test/test_macro.py test/test_macro_blackbox.py \
  test/test_macro_system.py test/test_macro_disabled_during_p7_gap.py \
  docs/examples/emergence.rst docs/api/macro.rst
```

- [ ] **Step 2: Remove legacy validate functions and their tests**

In `pyphi/validate.py`, delete `time_scale` (line ~206), `coarse_grain`
(~225), `blackbox` (~242), and `blackbox_and_coarse_grain` (~254) — they
have no callers outside `pyphi/macro.py` and `test/test_validate.py`
(verified by grep during planning). In `test/test_validate.py`, delete
the `from pyphi import macro` import and every test using
`macro.CoarseGrain`/`macro.Blackbox`/`validate.time_scale` (the block at
lines ~185-265; audit by grepping `macro\.` and the four function names).

- [ ] **Step 3: Remove `propagation_delay` example and fixture**

In `test/example_substrates.py`: delete the `propagation_delay()`
function and the `from pyphi.macro import Blackbox` / `from pyphi.macro
import MacroSystem` imports. In `test/conftest.py`: delete the
`propagation_delay` fixture (line ~367). Its only consumers were the
deleted test files.

- [ ] **Step 4: Reword comments that cite `pyphi/macro.py`**

- `pyphi/substrate.py:250` and `pyphi/core/tpm/joint_distribution.py:454`:
  drop the "the `pyphi/macro.py` legacy module" mention from each
  docstring sentence, keeping the remaining content accurate.
- `test/test_tpm_indices.py:28`: reword the docstring so it no longer
  attributes the indexing behavior to `pyphi/macro.py`.

- [ ] **Step 5: Retire legacy docs references**

- `docs/conf.py` (~lines 230, 284-288): delete the substitution
  definitions for `|macro|`, `|MacroNetwork|`, `|MacroSubsystem|`,
  `|CoarseGrain|`, `|CoarseGrains|`, and any `|Blackbox|` entry.
- `docs/index.rst:46`: remove the `examples/emergence` toctree line.
- `docs/examples/magic_cut.rst`: remove the `pyphi.macro.emergence`
  snippet (lines ~20-35) and any prose depending on it; keep the rest of
  the document intact.
- `docs/api/index.rst`: remove the `macro` entry if present.
- Grep `docs/` for remaining `pyphi.macro`/`|macro|`/`emergence`
  references and fix stragglers.

- [ ] **Step 6: Run the suite**

Run: `uv run pytest` (no path — includes the doctest sweep)
Expected: green, with ~28 fewer collected tests (22 legacy macro tests,
~6 legacy validate tests, 2 sentinel tests; skipped legacy blackbox
tests disappear).

- [ ] **Step 7: Commit**

```bash
git add -A pyphi/macro.py pyphi/validate.py pyphi/substrate.py \
  pyphi/core/tpm/joint_distribution.py test/ docs/
git -c commit.gpgsign=false commit -m "Remove legacy pre-2024 macro implementation

The CoarseGrain/Blackbox/MacroSubsystem pipeline implemented the
pre-2024 macro formalism (no sliding-window mappings, no background
apportionment, no per-unit discounting) and has been disabled since the
P7 rewrite. The Marshall et al. 2024 intrinsic-units framework replaces
it in the pyphi.macro package."
```

---

### Task 2: Mixed-radix helpers and `MacroUnit` core

**Files:**
- Create: `pyphi/macro/__init__.py`, `pyphi/macro/units.py`
- Test: `test/test_macro_units.py`

- [ ] **Step 1: Write the failing tests**

Create `test/test_macro_units.py`:

```python
"""Tests for pyphi.macro.units: MacroUnit value objects and mappings."""

import pytest

from pyphi.macro.units import MacroUnit
from pyphi.macro.units import _mixed_radix_digits
from pyphi.macro.units import _mixed_radix_index


class TestMixedRadix:
    def test_roundtrip_binary(self):
        radices = (2, 2, 2)
        for i in range(8):
            digits = _mixed_radix_digits(i, radices)
            assert _mixed_radix_index(digits, radices) == i

    def test_first_digit_varies_fastest(self):
        # little-endian: index 1 flips the FIRST digit
        assert _mixed_radix_digits(1, (2, 3, 2)) == (1, 0, 0)
        assert _mixed_radix_digits(2, (2, 3, 2)) == (0, 1, 0)

    def test_heterogeneous_radices(self):
        radices = (2, 3, 4)
        seen = set()
        for i in range(24):
            digits = _mixed_radix_digits(i, radices)
            assert _mixed_radix_index(digits, radices) == i
            seen.add(digits)
        assert len(seen) == 24

    def test_index_rejects_out_of_range_digit(self):
        with pytest.raises(ValueError):
            _mixed_radix_index((2, 0), (2, 2))


class TestMacroUnitConstruction:
    def test_minimal_identity_unit(self):
        unit = MacroUnit(constituents=(0,), update_grain=1, mapping=(0, 1))
        assert unit.micro_constituents == (0,)
        assert unit.micro_grain == 1
        assert unit.alphabet_size == 2

    def test_micro_constituents_sorted_union(self):
        unit = MacroUnit(constituents=(3, 1), update_grain=1,
                         mapping=(0, 0, 0, 1))
        assert unit.micro_constituents == (1, 3)

    def test_micro_grain_is_product_down_hierarchy(self):
        inner = MacroUnit(constituents=(0, 1), update_grain=2,
                          mapping=(0,) * 15 + (1,))
        outer = MacroUnit(constituents=(inner,), update_grain=3,
                          mapping=(0, 1, 1, 0, 0, 1, 1, 0))
        assert inner.micro_grain == 2
        assert outer.micro_grain == 6
        assert outer.micro_constituents == (0, 1)

    def test_empty_constituents_rejected(self):
        with pytest.raises(ValueError, match="constituent"):
            MacroUnit(constituents=(), update_grain=1, mapping=(0, 1))

    def test_update_grain_below_one_rejected(self):
        with pytest.raises(ValueError, match="grain"):
            MacroUnit(constituents=(0,), update_grain=0, mapping=(0, 1))

    def test_wrong_mapping_length_rejected(self):
        with pytest.raises(ValueError, match="mapping"):
            MacroUnit(constituents=(0, 1), update_grain=1, mapping=(0, 1))

    def test_nonbinary_mapping_entry_rejected(self):
        with pytest.raises(ValueError, match="mapping"):
            MacroUnit(constituents=(0,), update_grain=1, mapping=(0, 2))

    def test_nonsurjective_mapping_rejected(self):
        with pytest.raises(ValueError, match="both"):
            MacroUnit(constituents=(0,), update_grain=1, mapping=(0, 0))

    def test_overlapping_constituents_rejected(self):
        inner = MacroUnit(constituents=(0,), update_grain=1, mapping=(0, 1))
        with pytest.raises(ValueError, match="overlap"):
            MacroUnit(constituents=(inner, 0), update_grain=1,
                      mapping=(0, 0, 0, 1))

    def test_mismatched_constituent_grains_rejected(self):
        deep = MacroUnit(constituents=(0,), update_grain=2,
                         mapping=(0, 0, 0, 1))
        with pytest.raises(ValueError, match="grain"):
            MacroUnit(constituents=(deep, 2), update_grain=1,
                      mapping=(0, 0, 0, 1))

    def test_apportionment_overlapping_constituents_rejected(self):
        with pytest.raises(ValueError, match="apportionment"):
            MacroUnit(constituents=(0, 1), update_grain=1,
                      mapping=(0, 0, 0, 1), background_apportionment=(1,))

    def test_frozen(self):
        unit = MacroUnit(constituents=(0,), update_grain=1, mapping=(0, 1))
        with pytest.raises(AttributeError):
            unit.update_grain = 2
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest test/test_macro_units.py -x -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyphi.macro'`

- [ ] **Step 3: Implement**

Create `pyphi/macro/units.py`:

```python
"""Macro unit value objects (Marshall et al. 2024, Eq. 11).

A macro unit ``J = (U^J, V^J, tau'_J, g'_J, W^J)`` is specified by its
direct constituents ``V^J`` (micro unit indices or meso ``MacroUnit``
objects), an update grain ``tau'_J`` counted in constituent updates, a
state mapping ``g'_J``, and a background apportionment ``W^J``. The
micro constituents ``U^J`` are derived recursively.

Truth-table indexing convention: the mapping is a flat tuple over the
joint sequence-states of the direct constituents. Within an update the
first constituent varies fastest (little-endian, matching pyphi's state
convention); updates are ordered oldest first, with newer updates
varying slower.

All index arithmetic is mixed-radix, keyed to per-constituent alphabet
tuples. Binary alphabets are enforced by validation at both the micro
and macro level.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property


def _mixed_radix_index(digits, radices):
    """Index of mixed-radix ``digits``; the first digit varies fastest."""
    index = 0
    for digit, radix in zip(reversed(digits), reversed(radices), strict=True):
        if not 0 <= digit < radix:
            raise ValueError(f"digit {digit} out of range for radix {radix}")
        index = index * radix + digit
    return index


def _mixed_radix_digits(index, radices):
    """Digits of ``index`` in mixed radix; the first digit varies fastest."""
    digits = []
    for radix in radices:
        digits.append(index % radix)
        index //= radix
    return tuple(digits)


@dataclass(frozen=True)
class MacroUnit:
    """A macro unit ``J = (U^J, V^J, tau'_J, g'_J, W^J)`` (Eq. 11).

    Args:
        constituents: Direct constituents ``V^J`` — micro unit indices
            or meso ``MacroUnit`` objects. Order fixes the truth-table
            digit order.
        update_grain: ``tau'_J`` — constituent updates per unit update.
        mapping: ``g'_J`` as a flat truth table of 0/1 entries over the
            ``prod(alphabets) ** update_grain`` joint sequence-states of
            the constituents (see module docstring for digit order).
        background_apportionment: ``W^J`` — universe indices apportioned
            to this unit.
    """

    constituents: tuple[MacroUnit | int, ...]
    update_grain: int
    mapping: tuple[int, ...]
    background_apportionment: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "constituents", tuple(self.constituents))
        object.__setattr__(self, "mapping", tuple(self.mapping))
        object.__setattr__(
            self,
            "background_apportionment",
            tuple(self.background_apportionment),
        )
        if not self.constituents:
            raise ValueError("a macro unit requires at least one constituent")
        if self.update_grain < 1:
            raise ValueError(
                f"update grain must be >= 1; got {self.update_grain}"
            )
        micro_sets = []
        grains = set()
        for c in self.constituents:
            if isinstance(c, MacroUnit):
                micro_sets.append(set(c.micro_constituents))
                grains.add(c.micro_grain)
            elif isinstance(c, int) and not isinstance(c, bool):
                if c < 0:
                    raise ValueError(f"negative micro unit index: {c}")
                micro_sets.append({c})
                grains.add(1)
            else:
                raise TypeError(
                    f"constituents must be ints or MacroUnits; got {c!r}"
                )
        union: set[int] = set()
        for s in micro_sets:
            if union & s:
                raise ValueError(
                    "constituents overlap in their micro constituents: "
                    f"{sorted(union & s)}"
                )
            union |= s
        if len(grains) > 1:
            raise ValueError(
                "constituents must share a single micro grain; got "
                f"{sorted(grains)}"
            )
        expected = 1
        for size in self.constituent_alphabet_sizes:
            expected *= size
        expected **= self.update_grain
        if len(self.mapping) != expected:
            raise ValueError(
                f"mapping must have {expected} entries for "
                f"{len(self.constituents)} constituents at update grain "
                f"{self.update_grain}; got {len(self.mapping)}"
            )
        if not set(self.mapping) <= {0, 1}:
            raise ValueError("mapping entries must be 0 or 1")
        if 0 not in self.mapping or 1 not in self.mapping:
            raise ValueError("mapping must produce both macro states")
        apportionment = self.background_apportionment
        if len(set(apportionment)) != len(apportionment):
            raise ValueError(
                f"duplicate background apportionment: {apportionment}"
            )
        if set(apportionment) & union:
            raise ValueError(
                "background apportionment overlaps the unit's micro "
                f"constituents: {sorted(set(apportionment) & union)}"
            )

    @property
    def alphabet_size(self) -> int:
        """Number of unit states (binary)."""
        return 2

    @cached_property
    def constituent_alphabet_sizes(self) -> tuple[int, ...]:
        """Alphabet size of each direct constituent."""
        return tuple(
            c.alphabet_size if isinstance(c, MacroUnit) else 2
            for c in self.constituents
        )

    @cached_property
    def micro_constituents(self) -> tuple[int, ...]:
        """``U^J``: the sorted union of micro constituents."""
        out: set[int] = set()
        for c in self.constituents:
            if isinstance(c, MacroUnit):
                out |= set(c.micro_constituents)
            else:
                out.add(c)
        return tuple(sorted(out))

    @cached_property
    def constituent_micro_grain(self) -> int:
        """The common micro grain of the direct constituents."""
        for c in self.constituents:
            return c.micro_grain if isinstance(c, MacroUnit) else 1
        raise AssertionError("unreachable: constituents validated nonempty")

    @cached_property
    def micro_grain(self) -> int:
        """``tau_J``: micro updates spanned by one update of this unit."""
        return self.update_grain * self.constituent_micro_grain
```

Create `pyphi/macro/__init__.py`:

```python
"""Intrinsic-units macro framework (Marshall et al. 2024).

Macro units are defined by sliding-window state mappings over their
micro constituents; macro cause and effect TPMs are built by the
four-step construction (Eqs. 26-40) and analyzed by the IIT 4.0
pipeline exactly as micro systems are.
"""

from pyphi.macro.units import MacroUnit

__all__ = ["MacroUnit"]
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest test/test_macro_units.py -x -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyphi/macro/__init__.py pyphi/macro/units.py test/test_macro_units.py
git -c commit.gpgsign=false commit -m "Add MacroUnit value object (Marshall 2024 Eq 11)"
```

---

### Task 3: `state_from` and the composed micro mapping

**Files:**
- Modify: `pyphi/macro/units.py`
- Test: `test/test_macro_units.py`

- [ ] **Step 1: Write the failing tests**

Append to `test/test_macro_units.py`:

```python
class TestStateFrom:
    def test_identity_unit(self):
        unit = MacroUnit(constituents=(0,), update_grain=1, mapping=(0, 1))
        assert unit.state_from(((0,),)) == 0
        assert unit.state_from(((1,),)) == 1

    def test_history_length_validated(self):
        unit = MacroUnit(constituents=(0,), update_grain=2,
                         mapping=(0, 0, 0, 1))
        with pytest.raises(ValueError, match="history"):
            unit.state_from(((0,),))

    def test_entry_shape_validated(self):
        unit = MacroUnit(constituents=(0, 1), update_grain=1,
                         mapping=(0, 0, 0, 1))
        with pytest.raises(ValueError, match="state"):
            unit.state_from(((0,),))

    def test_constituent_order_pins_truth_table_digits(self):
        # Mapping is over constituents in GIVEN order (3 first), while
        # state_from input columns follow sorted U^J = (1, 3).
        # mapping index = digits (state(3), state(1)) little-endian:
        # ON iff constituent 3 is ON and constituent 1 is OFF -> index 1.
        unit = MacroUnit(constituents=(3, 1), update_grain=1,
                         mapping=(0, 1, 0, 0))
        # columns ordered by micro index: (state(1), state(3))
        assert unit.state_from(((0, 1),)) == 1
        assert unit.state_from(((1, 0),)) == 0
        assert unit.state_from(((1, 1),)) == 0

    def test_updates_oldest_first_newest_slowest(self):
        # tau = 2 over one constituent; digits = (oldest, newest).
        # mapping index 1 = (1, 0): ON in the OLD update only.
        unit = MacroUnit(constituents=(5,), update_grain=2,
                         mapping=(0, 1, 0, 0))
        assert unit.state_from(((1,), (0,))) == 1
        assert unit.state_from(((0,), (1,))) == 0

    def test_meso_composition_hand_checked(self):
        # inner: over micro (0, 1), grain 1, ON iff both ON
        inner = MacroUnit(constituents=(0, 1), update_grain=1,
                          mapping=(0, 0, 0, 1))
        # outer: over (inner,), grain 2, ON iff inner ON at the NEWER
        # of its two updates: digits (old, new) -> index 2 = (0, 1)
        outer = MacroUnit(constituents=(inner,), update_grain=2,
                          mapping=(0, 0, 1, 1))
        # micro history: two updates of (u0, u1), oldest first
        assert outer.state_from(((0, 0), (1, 1))) == 1
        assert outer.state_from(((1, 1), (0, 1))) == 0
        assert outer.state_from(((1, 1), (1, 1))) == 1

    def test_micro_mapping_identity(self):
        unit = MacroUnit(constituents=(0,), update_grain=1, mapping=(0, 1))
        assert unit.micro_mapping == (0, 1)

    def test_micro_mapping_matches_state_from(self):
        from pyphi.macro.units import _mixed_radix_digits

        inner = MacroUnit(constituents=(2, 0), update_grain=1,
                          mapping=(0, 1, 1, 0))
        outer = MacroUnit(constituents=(inner,), update_grain=2,
                          mapping=(0, 1, 1, 0))
        n = len(outer.micro_constituents)
        tau = outer.micro_grain
        for index in range(2 ** (n * tau)):
            digits = _mixed_radix_digits(index, (2,) * (n * tau))
            history = tuple(
                digits[t * n:(t + 1) * n] for t in range(tau)
            )
            assert outer.micro_mapping[index] == outer.state_from(history)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest test/test_macro_units.py -x -q`
Expected: FAIL with `AttributeError: ... 'state_from'`

- [ ] **Step 3: Implement**

Append to the `MacroUnit` class in `pyphi/macro/units.py`:

```python
    def state_from(self, history) -> int:
        """The unit's state given a micro-state window of ``U^J`` (Eq. 22).

        Args:
            history: Sequence of length :attr:`micro_grain` of micro
                states of ``U^J`` (each a tuple of 0/1 values ordered by
                ascending micro index), oldest first.

        Returns:
            int: The macro state ``j = g_J`` applied to the window.
        """
        history = tuple(tuple(s) for s in history)
        if len(history) != self.micro_grain:
            raise ValueError(
                f"history must have {self.micro_grain} entries; "
                f"got {len(history)}"
            )
        n = len(self.micro_constituents)
        for s in history:
            if len(s) != n or not set(s) <= {0, 1}:
                raise ValueError(
                    f"each history state must be a binary tuple of "
                    f"length {n}; got {s}"
                )
        position = {u: i for i, u in enumerate(self.micro_constituents)}
        child_grain = self.constituent_micro_grain
        digits = []
        for k in range(self.update_grain):
            window = history[k * child_grain:(k + 1) * child_grain]
            for c in self.constituents:
                if isinstance(c, MacroUnit):
                    sub = tuple(
                        tuple(s[position[u]] for u in c.micro_constituents)
                        for s in window
                    )
                    digits.append(c.state_from(sub))
                else:
                    # micro constituents imply child_grain == 1
                    digits.append(window[0][position[c]])
        radices = self.constituent_alphabet_sizes * self.update_grain
        return self.mapping[_mixed_radix_index(tuple(digits), radices)]

    @cached_property
    def micro_mapping(self) -> tuple[int, ...]:
        """``g_J``: the composed truth table over micro windows (Eq. 14).

        Indexed with the same convention as :attr:`mapping`, with the
        micro constituents of ``U^J`` in ascending order as the
        within-update digits.
        """
        n = len(self.micro_constituents)
        tau = self.micro_grain
        radices = (2,) * (n * tau)
        table = []
        for index in range(2 ** (n * tau)):
            digits = _mixed_radix_digits(index, radices)
            history = tuple(
                digits[t * n:(t + 1) * n] for t in range(tau)
            )
            table.append(self.state_from(history))
        return tuple(table)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest test/test_macro_units.py -x -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyphi/macro/units.py test/test_macro_units.py
git -c commit.gpgsign=false commit -m "Add MacroUnit state mapping composition (Eqs 14, 22)"
```

---

### Task 4: `micro_unit`, `coarse_grain`, and `blackbox` constructors

**Files:**
- Modify: `pyphi/macro/units.py`, `pyphi/macro/__init__.py`
- Test: `test/test_macro_units.py`

- [ ] **Step 1: Write the failing tests**

Append to `test/test_macro_units.py` (add the imports at the top of the
file):

```python
from pyphi.macro.units import blackbox
from pyphi.macro.units import coarse_grain
from pyphi.macro.units import micro_unit


class TestMappingConstructors:
    def test_micro_unit_is_identity(self):
        unit = micro_unit(3)
        assert unit.micro_constituents == (3,)
        assert unit.micro_grain == 1
        assert unit.micro_mapping == (0, 1)

    def test_coarse_grain_both_on(self):
        # Example 1's mapping: ON iff both constituents ON
        assert coarse_grain(2, on_counts={2}) == (0, 0, 0, 1)

    def test_coarse_grain_at_least_one(self):
        assert coarse_grain(2, on_counts={1, 2}) == (0, 1, 1, 1)

    def test_coarse_grain_invalid_counts_rejected(self):
        with pytest.raises(ValueError, match="count"):
            coarse_grain(2, on_counts={3})

    def test_coarse_grain_degenerate_rejected(self):
        with pytest.raises(ValueError, match="both"):
            MacroUnit(constituents=(0, 1), update_grain=1,
                      mapping=coarse_grain(2, on_counts=set()))

    def test_blackbox_single_output_final_update(self):
        # Example 2's mapping: 4 constituents, tau = 2, output = local
        # index 2 (C); state = C at the final update.
        table = blackbox(4, update_grain=2, output_constituents=(2,))
        assert len(table) == 2 ** 8
        from pyphi.macro.units import _mixed_radix_digits
        for index in (0, 5, 64, 127, 200, 255):
            digits = _mixed_radix_digits(index, (2,) * 8)
            final = digits[4:]
            assert table[index] == final[2]

    def test_blackbox_multiple_outputs_all_on(self):
        table = blackbox(2, update_grain=1, output_constituents=(0, 1))
        assert table == (0, 0, 0, 1)

    def test_blackbox_output_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="output"):
            blackbox(2, update_grain=1, output_constituents=(2,))

    def test_blackbox_no_outputs_rejected(self):
        with pytest.raises(ValueError, match="output"):
            blackbox(2, update_grain=1, output_constituents=())
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest test/test_macro_units.py -x -q`
Expected: FAIL with ImportError on `blackbox`

- [ ] **Step 3: Implement**

Append module functions to `pyphi/macro/units.py`:

```python
def micro_unit(index: int, background_apportionment=()) -> MacroUnit:
    """An identity macro unit over a single micro unit."""
    return MacroUnit(
        constituents=(index,),
        update_grain=1,
        mapping=(0, 1),
        background_apportionment=background_apportionment,
    )


def coarse_grain(num_constituents: int, on_counts) -> tuple[int, ...]:
    """A coarse-graining truth table (update grain 1).

    The macro state is 1 when the number of ON constituents is in
    ``on_counts``.
    """
    on_counts = frozenset(on_counts)
    if not on_counts <= set(range(num_constituents + 1)):
        raise ValueError(
            f"on_counts must be counts in 0..{num_constituents}; "
            f"got {sorted(on_counts)}"
        )
    radices = (2,) * num_constituents
    return tuple(
        1 if sum(_mixed_radix_digits(i, radices)) in on_counts else 0
        for i in range(2 ** num_constituents)
    )


def blackbox(
    num_constituents: int, update_grain: int, output_constituents
) -> tuple[int, ...]:
    """A black-boxing truth table.

    The macro state is 1 when every designated output constituent is ON
    at the final update of the window; all other constituents and
    updates are ignored.
    """
    outputs = tuple(output_constituents)
    if not outputs or len(set(outputs)) != len(outputs):
        raise ValueError(
            f"output_constituents must be nonempty and unique; "
            f"got {outputs}"
        )
    if not set(outputs) <= set(range(num_constituents)):
        raise ValueError(
            f"output_constituents must be local indices in "
            f"0..{num_constituents - 1}; got {outputs}"
        )
    radices = (2,) * (num_constituents * update_grain)
    table = []
    for i in range(2 ** (num_constituents * update_grain)):
        digits = _mixed_radix_digits(i, radices)
        final = digits[(update_grain - 1) * num_constituents:]
        table.append(1 if all(final[o] for o in outputs) else 0)
    return tuple(table)
```

Update `pyphi/macro/__init__.py` exports:

```python
from pyphi.macro.units import MacroUnit
from pyphi.macro.units import blackbox
from pyphi.macro.units import coarse_grain
from pyphi.macro.units import micro_unit

__all__ = ["MacroUnit", "blackbox", "coarse_grain", "micro_unit"]
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest test/test_macro_units.py -x -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyphi/macro/units.py pyphi/macro/__init__.py test/test_macro_units.py
git -c commit.gpgsign=false commit -m "Add coarse_grain and blackbox mapping constructors"
```

---

### Task 5: Step 1 — discounted unit probabilities (Eqs 26-30)

**Files:**
- Create: `pyphi/macro/tpm.py`
- Test: `test/test_macro_tpm.py`

- [ ] **Step 1: Write the failing tests**

Create `test/test_macro_tpm.py`:

```python
"""Tests for pyphi.macro.tpm: the four-step macro TPM construction."""

import numpy as np
import pytest

from pyphi.macro.tpm import _discounted_on_probabilities
from pyphi.macro.units import MacroUnit
from pyphi.macro.units import coarse_grain
from pyphi.macro.units import micro_unit
from pyphi.substrate import Substrate

# Asymmetric 4-unit substrate: every unit has a distinct rule, so axis
# or endianness errors cannot cancel (TriggeredTPM lesson).
def _asymmetric_substrate():
    n = 4
    states = [tuple((i >> k) & 1 for k in range(n)) for i in range(2 ** n)]
    tpm = np.zeros((2 ** n, n))
    for r, s in enumerate(states):
        tpm[r, 0] = 0.1 + 0.5 * s[1] + 0.3 * s[3]
        tpm[r, 1] = 0.2 + 0.7 * s[0]
        tpm[r, 2] = 0.05 + 0.6 * s[0] * s[1] + 0.3 * s[3]
        tpm[r, 3] = 0.9 - 0.8 * s[2]
    return Substrate(tpm, node_labels=("A", "B", "C", "D"))


def _flat_on_probabilities(substrate):
    """(2**n, n) ON probabilities, little-endian rows, from the factors."""
    factored = substrate.factored_tpm
    n = factored.n_nodes
    return np.stack(
        [factored.factor(i)[..., 1].reshape(-1, order="F") for i in range(n)],
        axis=1,
    )


class TestDiscounting:
    def test_constituent_rows_untouched(self):
        substrate = _asymmetric_substrate()
        units = (
            MacroUnit((0, 1), 1, coarse_grain(2, on_counts={2})),
            MacroUnit((2, 3), 1, coarse_grain(2, on_counts={2})),
        )
        original = _flat_on_probabilities(substrate)
        discounted = _discounted_on_probabilities(substrate.factored_tpm,
                                                  units, 0)
        # Eq 27: units 0 and 1 keep all connections
        assert np.array_equal(discounted[:, 0], original[:, 0])
        assert np.array_equal(discounted[:, 1], original[:, 1])

    def test_other_system_units_fully_noised(self):
        substrate = _asymmetric_substrate()
        units = (
            MacroUnit((0, 1), 1, coarse_grain(2, on_counts={2})),
            MacroUnit((2, 3), 1, coarse_grain(2, on_counts={2})),
        )
        original = _flat_on_probabilities(substrate)
        discounted = _discounted_on_probabilities(substrate.factored_tpm,
                                                  units, 0)
        # Eq 28: units 2 and 3 become the uniform-average marginal
        for i in (2, 3):
            assert np.allclose(discounted[:, i], original[:, i].mean())

    def test_unapportioned_background_fully_noised(self):
        substrate = _asymmetric_substrate()
        units = (MacroUnit((0, 1), 1, coarse_grain(2, on_counts={2})),)
        original = _flat_on_probabilities(substrate)
        discounted = _discounted_on_probabilities(substrate.factored_tpm,
                                                  units, 0)
        for i in (2, 3):
            assert np.allclose(discounted[:, i], original[:, i].mean())

    def test_apportioned_background_keeps_patron_inputs_only(self):
        substrate = _asymmetric_substrate()
        # Unit over (0, 1) with background unit 2 apportioned to it;
        # unit 3 is unapportioned background. Unit 2's original rule is
        # 0.05 + 0.6*s0*s1 + 0.3*s3: after Eq 29 noising over unit 3
        # (outside U union W = {0, 1, 2}) the row keeps its dependence
        # on the patron constituents 0, 1 but averages s3 to 0.5.
        units = (
            MacroUnit((0, 1), 1, coarse_grain(2, on_counts={2}),
                      background_apportionment=(2,)),
        )
        discounted = _discounted_on_probabilities(substrate.factored_tpm,
                                                  units, 0)
        n = 4
        idx = np.arange(2 ** n)
        s0 = (idx >> 0) & 1
        s1 = (idx >> 1) & 1
        expected = 0.05 + 0.6 * s0 * s1 + 0.3 * 0.5
        assert np.allclose(discounted[:, 2], expected, atol=1e-15)
        # Unit 3 is unapportioned: fully noised (Eq 28)
        original = _flat_on_probabilities(substrate)
        assert np.allclose(discounted[:, 3], original[:, 3].mean())

    def test_little_endian_row_order_pinned(self):
        substrate = _asymmetric_substrate()
        units = (MacroUnit((0, 1, 2, 3), 1,
                           coarse_grain(4, on_counts={4})),)
        discounted = _discounted_on_probabilities(substrate.factored_tpm,
                                                  units, 0)
        # row index 1 = state (1,0,0,0): unit 1's rule = 0.2 + 0.7*s[0]
        assert discounted[1, 1] == pytest.approx(0.9)
        # row index 2 = state (0,1,0,0): unit 0's rule = 0.1 + 0.5*s[1]
        assert discounted[2, 0] == pytest.approx(0.6)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest test/test_macro_tpm.py -x -q`
Expected: FAIL with `ModuleNotFoundError` on `pyphi.macro.tpm`

- [ ] **Step 3: Implement**

Create `pyphi/macro/tpm.py`:

```python
"""The four-step macro TPM construction (Marshall et al. 2024, Eqs. 26-40).

Step 1 discounts micro connections extrinsic to the macro unit being
updated (Eqs. 26-30). Step 2 chains the modified probabilities into
micro-update sequences (Eq. 31). Step 3 causally marginalizes the
background, conditioning on the current micro state for effects (Eq. 33)
and Bayesian-weighting the pre-window background state for causes
(Eq. 34). Step 4 compresses sequences into macro states via the mapping
preimages and the sequence-proportion weights ``r(u^S, s)``
(Eqs. 35-40). Steps 2 and 4 are fused: sequence probabilities are
accumulated per chaining step into per-update state classes of the
unit's micro constituents, never materializing the full sequence tensor.
"""

from __future__ import annotations

import numpy as np

from pyphi import exceptions
from pyphi.core.tpm.factored import FactoredTPM

from pyphi.macro.units import MacroUnit
from pyphi.macro.units import _mixed_radix_digits


def _system_micro_indices(units) -> tuple[int, ...]:
    """``U^S``: the sorted union of the units' micro constituents."""
    out: set[int] = set()
    for unit in units:
        out |= set(unit.micro_constituents)
    return tuple(sorted(out))


def _patron_units(units) -> dict[int, int]:
    """Map each apportioned background index to its patron unit's index."""
    out: dict[int, int] = {}
    for k, unit in enumerate(units):
        for w in unit.background_apportionment:
            out[w] = k
    return out


def _discounted_on_probabilities(
    factored: FactoredTPM, units: tuple[MacroUnit, ...], j: int
) -> np.ndarray:
    """Step 1 (Eqs. 26-30): modified ON probabilities for updating unit ``j``.

    Returns:
        np.ndarray: ``(2**n, n)`` — for each universe state (little-endian
        row index) and micro unit, the modified probability that the unit
        is ON at the next micro update.
    """
    n = factored.n_nodes
    constituents = set(units[j].micro_constituents)
    patron = _patron_units(units)
    columns = []
    for i in range(n):
        p_on = factored.factor(i)[..., 1]
        if i in constituents:
            out = p_on  # Eq. 27: connections among U^J kept intact
        elif i in patron:
            k = patron[i]
            keep = set(units[k].micro_constituents) | set(
                units[k].background_apportionment
            )
            axes = tuple(a for a in range(n) if a not in keep)
            if axes:
                out = np.broadcast_to(
                    p_on.mean(axis=axes, keepdims=True), p_on.shape
                )
            else:
                out = p_on  # Eq. 29 with nothing to noise
        else:
            # Eq. 28: other system units and unapportioned background
            out = np.full(p_on.shape, p_on.mean())
        columns.append(np.asarray(out).reshape(-1, order="F"))
    return np.stack(columns, axis=1)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest test/test_macro_tpm.py -x -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyphi/macro/tpm.py test/test_macro_tpm.py
git -c commit.gpgsign=false commit -m "Add Step 1 discounting of the macro TPM construction (Eqs 26-30)"
```

---

### Task 6: Steps 2+4 fused — sequence chaining and compression (empty background)

**Files:**
- Modify: `pyphi/macro/tpm.py`
- Test: `test/test_macro_tpm.py`

- [ ] **Step 1: Write the failing tests**

Append to `test/test_macro_tpm.py` (add imports at top:
`from pyphi.macro.tpm import _full_transition_matrix`,
`_unit_macro_probabilities`, `_unit_final_state_proportions`,
`from pyphi.macro.units import blackbox`):

```python
MIN_TPM = np.array([
    [0.05, 0.05],
    [0.05, 0.06],
    [0.06, 0.05],
    [0.95, 0.95],
])

CG_TPM = np.array([
    [0.05, 0.05, 0.05, 0.05],
    [0.06, 0.15, 0.05, 0.05],
    [0.15, 0.06, 0.05, 0.05],
    [0.16, 0.16, 0.85, 0.85],
    [0.05, 0.05, 0.06, 0.15],
    [0.06, 0.15, 0.06, 0.15],
    [0.15, 0.06, 0.06, 0.15],
    [0.16, 0.16, 0.86, 0.95],
    [0.05, 0.05, 0.15, 0.06],
    [0.06, 0.15, 0.15, 0.06],
    [0.15, 0.06, 0.15, 0.06],
    [0.16, 0.16, 0.95, 0.86],
    [0.85, 0.85, 0.16, 0.16],
    [0.86, 0.95, 0.16, 0.16],
    [0.95, 0.86, 0.16, 0.16],
    [0.96, 0.96, 0.96, 0.96],
])


def _bbx_micro_tpm():
    n = 8
    states = [tuple((i >> k) & 1 for k in range(n)) for i in range(2 ** n)]
    tpm = np.zeros((2 ** n, n))
    for r, cs in enumerate(states):
        p = tpm[r]
        p[0] = 0.01 + 0.01 * cs[0] + 0.1 * cs[3] + 0.8 * cs[6] + 0.05 * cs[1]
        p[1] = 0.01 + 0.01 * cs[1] + 0.1 * cs[3] + 0.8 * cs[6] + 0.05 * cs[0]
        p[2] = (0.01 + 0.01 * cs[2] + 0.85 * int(cs[0] + cs[1] > 0)
                + 0.1 * int(cs[0] + cs[1] == 2))
        p[3] = 0.01 + 0.01 * cs[3] + 0.85 * cs[2] + 0.05 * (cs[0] + cs[1])
        p[4] = 0.01 + 0.01 * cs[4] + 0.1 * cs[7] + 0.8 * cs[2] + 0.05 * cs[5]
        p[5] = 0.01 + 0.01 * cs[5] + 0.1 * cs[7] + 0.8 * cs[2] + 0.05 * cs[4]
        p[6] = (0.01 + 0.01 * cs[6] + 0.85 * int(cs[4] + cs[5] > 0)
                + 0.1 * int(cs[4] + cs[5] == 2))
        p[7] = 0.01 + 0.01 * cs[7] + 0.85 * cs[6] + 0.05 * (cs[4] + cs[5])
    return tpm


class TestTransitionMatrix:
    def test_rows_stochastic(self):
        substrate = _asymmetric_substrate()
        on = _flat_on_probabilities(substrate)
        P = _full_transition_matrix(on)
        assert np.allclose(P.sum(axis=1), 1.0)

    def test_hand_checked_entry(self):
        substrate = _asymmetric_substrate()
        on = _flat_on_probabilities(substrate)
        P = _full_transition_matrix(on)
        # From state (1,0,0,0) (row 1) to state (1,1,0,0) (column 3):
        # p = pA(1)*pB(1)*(1-pC)*(1-pD) at s=(1,0,0,0):
        # pA = 0.1, pB = 0.9, pC = 0.05, pD = 0.9
        expected = 0.1 * 0.9 * 0.95 * 0.1
        assert P[1, 3] == pytest.approx(expected, abs=1e-15)


class TestFinalStateProportions:
    def test_tau1_uniform_over_preimage(self):
        unit = MacroUnit((0, 1), 1, coarse_grain(2, on_counts={2}))
        r0 = _unit_final_state_proportions(unit, 0)
        r1 = _unit_final_state_proportions(unit, 1)
        assert np.allclose(r0, [1 / 3, 1 / 3, 1 / 3, 0.0])
        assert np.allclose(r1, [0.0, 0.0, 0.0, 1.0])

    def test_blackbox_counts_prefix_multiplicity(self):
        # tau = 2 over 1 constituent, state = final update: preimages
        # {(a, j) : a free} -> uniform over the final state only.
        unit = MacroUnit((0,), 2, blackbox(1, 2, (0,)))
        assert np.allclose(_unit_final_state_proportions(unit, 1),
                           [0.0, 1.0])
        assert np.allclose(_unit_final_state_proportions(unit, 0),
                           [1.0, 0.0])

    def test_sums_to_one(self):
        unit = MacroUnit((0, 1), 2, blackbox(2, 2, (1,)))
        for j in (0, 1):
            assert _unit_final_state_proportions(unit, j).sum() == (
                pytest.approx(1.0)
            )


class TestHandComputedTinyCase:
    """Authors' 'minimal' example: 2 micro units, 1 macro unit, tau=1.

    alpha = coarse-grain of (A, B), ON iff both ON. Empty background.
    Hand computation (matches the authors' hand-derived min_macro TPM):
      T(alpha'=1 | alpha=0)
        = mean over preimage {00, 10, 01} of p(A'=1)p(B'=1)
        = (0.05*0.05 + 0.05*0.06 + 0.06*0.05) / 3
        = 0.05*0.05 + 2*0.01*0.05/3
      T(alpha'=1 | alpha=1) = 0.95*0.95
    """

    def test_min_macro_tpm(self):
        from pyphi.macro.tpm import macro_tpms

        substrate = Substrate(MIN_TPM, node_labels=("A", "B"))
        units = (MacroUnit((0, 1), 1, coarse_grain(2, on_counts={2})),)
        cause, effect = macro_tpms(substrate, units, ((0, 0),))
        expected_off = 0.05 * 0.05 + 2 * 0.01 * 0.05 / 3
        expected_on = 0.95 * 0.95
        for tpm in (cause, effect):
            factor = tpm.factor(0)
            assert factor.shape == (2, 2)
            assert factor[0, 1] == pytest.approx(expected_off, abs=1e-15)
            assert factor[1, 1] == pytest.approx(expected_on, abs=1e-15)
            assert np.allclose(factor.sum(axis=-1), 1.0)


class TestPaperExampleTPMs:
    def test_cg_construction_exact(self):
        """Example 1. The construction values are derived in closed form.

        The authors' committed macro TPM (their repo, results from a
        hand-entered matrix) contains a rounding (0.006833 for 0.0615/9)
        and a hand-entry error (0.9212 where the construction gives
        0.96**2 = 0.9216); rows (1,0)/(0,1) match exactly. See the plan
        document for the derivation and the numerical validation against
        their computed bbx TPM, which has no such discrepancy.
        """
        from pyphi.macro.tpm import macro_tpms

        substrate = Substrate(CG_TPM, node_labels=("A", "B", "C", "D"))
        units = (
            MacroUnit((0, 1), 1, coarse_grain(2, on_counts={2})),
            MacroUnit((2, 3), 1, coarse_grain(2, on_counts={2})),
        )
        cause, effect = macro_tpms(substrate, units, ((0, 0, 0, 0),))
        expected = np.array([
            [0.0615 / 9, 0.0615 / 9],
            [0.0256, 0.7855],
            [0.7855, 0.0256],
            [0.9216, 0.9216],
        ])
        for tpm in (cause, effect):
            built = np.stack(
                [tpm.factor(i)[..., 1].reshape(-1, order="F")
                 for i in range(2)],
                axis=1,
            )
            assert np.allclose(built, expected, atol=1e-14)

    def test_bbx_construction_matches_authors_computation(self):
        """Example 2: must equal the authors' computed TPM to ~1e-15.

        Their computation (repo, `_get_blackbox_example_macro_tpm`):
        square the state-by-state micro TPM, condition on the (C, G)
        preimage of the current macro state, and read out C / G at the
        final update. For this wiring that shortcut coincides with the
        full construction because external influence enters each half
        only via the conditioned units at the window start.
        """
        from pyphi.convert import sbn2sbs
        from pyphi.macro.tpm import macro_tpms

        micro = _bbx_micro_tpm()
        substrate = Substrate(micro, node_labels=tuple("ABCDEFGH"))
        units = (
            MacroUnit((0, 1, 2, 3), 2, blackbox(4, 2, (2,))),
            MacroUnit((4, 5, 6, 7), 2, blackbox(4, 2, (2,))),
        )
        ones = (1,) * 8
        cause, effect = macro_tpms(substrate, units, (ones, ones))

        sbs = sbn2sbs(micro)
        tpm2 = sbs @ sbs
        idx = np.arange(2 ** 8)
        c_bit = (idx >> 2) & 1
        g_bit = (idx >> 6) & 1
        expected = np.zeros((4, 2))
        for si, (a, b) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
            rows = np.where((c_bit == a) & (g_bit == b))[0]
            expected[si, 0] = tpm2[rows][:, c_bit == 1].sum(axis=1).mean()
            expected[si, 1] = tpm2[rows][:, g_bit == 1].sum(axis=1).mean()

        for tpm in (cause, effect):
            built = np.stack(
                [tpm.factor(i)[..., 1].reshape(-1, order="F")
                 for i in range(2)],
                axis=1,
            )
            assert np.allclose(built, expected, atol=1e-13)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest test/test_macro_tpm.py -x -q`
Expected: FAIL with ImportError on `_full_transition_matrix`

- [ ] **Step 3: Implement**

Append to `pyphi/macro/tpm.py`:

```python
def _full_transition_matrix(on_probabilities: np.ndarray) -> np.ndarray:
    """Row-stochastic ``(2**n, 2**n)`` matrix from ON probabilities (Eq. 30).

    Rows and columns are little-endian universe state indices.
    """
    num_states, n = on_probabilities.shape
    transition = np.ones((num_states, num_states))
    column_bits = np.arange(num_states)
    for i in range(n):
        bit = (column_bits >> i) & 1
        p = on_probabilities[:, i][:, np.newaxis]
        transition *= np.where(bit[np.newaxis, :] == 1, p, 1.0 - p)
    return transition


def _unit_sequence_distributions(
    transition: np.ndarray, unit: MacroUnit
) -> np.ndarray:
    """Steps 2+4a fused (Eqs. 31, 35-36).

    Chains ``tau_J`` micro updates of the discounted transition matrix,
    accumulating probability into per-update state classes of ``U^J``.
    With the pinned digit convention, a sequence-class index is directly
    an index into ``unit.micro_mapping``.

    Returns:
        np.ndarray: ``(2**n, 2**(m * tau_J))`` — for each starting
        universe state, the probability of each ``U^J`` state-sequence.
    """
    num_states = transition.shape[0]
    m = len(unit.micro_constituents)
    num_classes = 2 ** m
    idx = np.arange(num_states)
    state_class = np.zeros(num_states, dtype=np.int64)
    for k, u in enumerate(unit.micro_constituents):
        state_class |= ((idx >> u) & 1) << k
    tau = unit.micro_grain
    dist = np.eye(num_states)[:, np.newaxis, :]  # (start, seq, current)
    place = 1
    for step in range(tau):
        seq_dim = dist.shape[1]
        advanced = np.einsum("xsu,uv->xsv", dist, transition)
        if step == tau - 1:
            out = np.zeros((num_states, seq_dim * num_classes))
            for a in range(num_classes):
                selected = state_class == a
                block = advanced[:, :, selected].sum(axis=2)
                out[:, a * place:a * place + seq_dim] += block
            return out
        out = np.zeros((num_states, seq_dim * num_classes, num_states))
        for a in range(num_classes):
            selected = state_class == a
            out[:, a * place:a * place + seq_dim, selected] = (
                advanced[:, :, selected]
            )
        dist = out
        place *= num_classes
    # tau >= 1 always returns inside the loop
    raise AssertionError("unreachable")


def _unit_macro_probabilities(
    transition: np.ndarray, unit: MacroUnit
) -> np.ndarray:
    """Eq. 35: probability of each macro state of ``J`` per starting state.

    Returns:
        np.ndarray: ``(2**n, 2)``.
    """
    sequence_dist = _unit_sequence_distributions(transition, unit)
    table = np.asarray(unit.micro_mapping)
    return np.stack(
        [
            sequence_dist[:, table == 0].sum(axis=1),
            sequence_dist[:, table == 1].sum(axis=1),
        ],
        axis=1,
    )


def _unit_final_state_proportions(unit: MacroUnit, j: int) -> np.ndarray:
    """Per-unit factor of ``r(u^S, s)`` (Eqs. 37-39).

    The proportion of ``g_J``-preimage sequences for macro state ``j``
    that end in each final ``U^J`` state. Counting is uniform over
    sequences (Eq. 38), not probability-weighted.
    """
    m = len(unit.micro_constituents)
    tau = unit.micro_grain
    table = np.asarray(unit.micro_mapping)
    idx = np.arange(len(table))
    final_state = idx >> (m * (tau - 1))
    counts = np.array(
        [np.sum((table == j) & (final_state == f)) for f in range(2 ** m)],
        dtype=np.float64,
    )
    return counts / counts.sum()
```

Also append the empty-background `macro_tpms` (Step 3 weighting is added
in Task 7; for this task background must be empty):

```python
def _state_weights(units, system_indices, macro_state) -> np.ndarray:
    """``r(u^S, s)`` over system micro states (Eqs. 37-39).

    Factorizes as the product of per-unit final-state proportions
    because the ``U^J`` are disjoint (Eq. 18) and exactly cover
    ``U^S`` (Eq. 23).
    """
    num_system_states = 2 ** len(system_indices)
    position = {u: k for k, u in enumerate(system_indices)}
    idx = np.arange(num_system_states)
    weights = np.ones(num_system_states)
    for unit, j in zip(units, macro_state, strict=True):
        local = np.zeros(num_system_states, dtype=np.int64)
        for b, u in enumerate(unit.micro_constituents):
            local |= ((idx >> position[u]) & 1) << b
        weights *= _unit_final_state_proportions(unit, j)[local]
    return weights
```

(The public `macro_tpms` entry point is implemented in Task 7 together
with Step 3; to make this task's `macro_tpms`-using tests pass, Task 7's
implementation below may be written in the same sitting — if so, run
both tasks' tests and fold the work into this commit followed
immediately by Task 7's. Otherwise implement a temporary
empty-background-only `macro_tpms` and replace it in Task 7.)

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest test/test_macro_tpm.py -x -q`
Expected: PASS (after Task 7's `macro_tpms` lands, for the tiny-case and
paper-TPM tests)

- [ ] **Step 5: Commit**

```bash
git add pyphi/macro/tpm.py test/test_macro_tpm.py
git -c commit.gpgsign=false commit -m "Add fused sequence chaining and compression (Eqs 31, 35-39)"
```

---

### Task 7: Step 3 — background weighting and the public `macro_tpms`

**Files:**
- Modify: `pyphi/macro/tpm.py`, `pyphi/macro/__init__.py`
- Test: `test/test_macro_tpm.py`

- [ ] **Step 1: Write the failing tests**

Append to `test/test_macro_tpm.py` (import `macro_tpms`,
`_background_weights_cause`, `micro_unit`, `System`):

```python
from pyphi.macro.tpm import _background_weights_cause
from pyphi.macro.tpm import macro_tpms
from pyphi.system import System


class TestBackgroundWeights:
    def test_cause_weights_hand_bayes(self):
        """q_c (Eq 34) on the asymmetric substrate, system = {0}.

        q_c(w) = sum_{uS} P(earliest | (w, uS)) / sum_u P(earliest | u),
        with earliest = the current state for tau = 1.
        """
        substrate = _asymmetric_substrate()
        earliest = (1, 0, 1, 0)
        q = _background_weights_cause(
            substrate.factored_tpm, system_indices=(0,), earliest=earliest
        )
        on = _flat_on_probabilities(substrate)
        # P(earliest | u) for all 16 prior states u
        likelihood = np.ones(16)
        for i in range(4):
            p = on[:, i]
            likelihood *= p if earliest[i] == 1 else 1 - p
        # background = units 1,2,3; sum over the system bit (unit 0)
        idx = np.arange(16)
        w_index = ((idx >> 1) & 1) | (((idx >> 2) & 1) << 1) | (
            ((idx >> 3) & 1) << 2
        )
        expected = np.zeros(8)
        np.add.at(expected, w_index, likelihood)
        expected /= likelihood.sum()
        assert np.allclose(q, expected, atol=1e-15)
        assert q.sum() == pytest.approx(1.0)

    def test_unreachable_earliest_state_raises(self):
        # A deterministic substrate where state (0,) cannot be reached:
        tpm = np.array([[1.0], [1.0]])  # always ON
        substrate = Substrate(tpm, node_labels=("A",))
        from pyphi import exceptions

        with pytest.raises(exceptions.StateUnreachableBackwardsError):
            _background_weights_cause(
                substrate.factored_tpm, system_indices=(), earliest=(0,)
            )


class TestMicroReductionWithBackground:
    """Identity macroing of a proper subset must reproduce System's
    background-conditioned TPMs exactly (Eqs 33-34 reduce to IIT 4.0
    Eq 4 at tau = 1)."""

    @pytest.mark.parametrize("subset", [(0,), (0, 1), (1, 3)])
    def test_identity_subset_equals_proper_tpms(self, subset):
        substrate = _asymmetric_substrate()
        state = (1, 0, 1, 0)
        units = tuple(micro_unit(i) for i in subset)
        cause, effect = macro_tpms(substrate, units, (state,))
        system = System(substrate, state, subset)
        for built, reference in (
            (cause, system.proper_cause_tpm),
            (effect, system.proper_effect_tpm),
        ):
            for k in range(len(subset)):
                assert np.allclose(
                    built.factor(k), reference.factor(k), atol=1e-15
                )

    def test_cause_and_effect_differ_with_background(self):
        substrate = _asymmetric_substrate()
        state = (1, 0, 1, 0)
        units = (micro_unit(0), micro_unit(1))
        cause, effect = macro_tpms(substrate, units, (state,))
        assert not all(
            np.allclose(cause.factor(k), effect.factor(k))
            for k in range(2)
        )


class TestApportionedBackgroundPath:
    """Eq 29 path: no published anchor (both paper examples have empty
    background) — unit-level checks only until sub-project 3."""

    def test_apportionment_changes_the_tpm(self):
        substrate = _asymmetric_substrate()
        state = (1, 0, 1, 0)
        plain = (MacroUnit((0, 1), 1, coarse_grain(2, on_counts={2})),)
        apportioned = (
            MacroUnit((0, 1), 1, coarse_grain(2, on_counts={2}),
                      background_apportionment=(3,)),
        )
        # tau = 1: Step 1 noising is invisible, so the TPMs agree...
        _, effect_plain = macro_tpms(substrate, plain, (state,))
        _, effect_app = macro_tpms(substrate, apportioned, (state,))
        assert np.allclose(effect_plain.factor(0), effect_app.factor(0))

    def test_apportionment_bites_at_tau_2(self):
        substrate = _asymmetric_substrate()
        history = ((1, 0, 1, 0), (1, 1, 0, 0))
        plain = (MacroUnit((0, 1), 2, blackbox(2, 2, (0,))),)
        apportioned = (
            MacroUnit((0, 1), 2, blackbox(2, 2, (0,)),
                      background_apportionment=(3,)),
        )
        _, effect_plain = macro_tpms(substrate, plain, history)
        _, effect_app = macro_tpms(substrate, apportioned, history)
        assert not np.allclose(effect_plain.factor(0),
                               effect_app.factor(0))

    def test_outputs_are_stochastic(self):
        substrate = _asymmetric_substrate()
        history = ((1, 0, 1, 0), (1, 1, 0, 0))
        units = (
            MacroUnit((0, 1), 2, blackbox(2, 2, (0,)),
                      background_apportionment=(3,)),
            MacroUnit((2,), 2, blackbox(1, 2, (0,))),
        )
        cause, effect = macro_tpms(substrate, units, history)
        for tpm in (cause, effect):
            for k in range(2):
                assert np.allclose(tpm.factor(k).sum(axis=-1), 1.0)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest test/test_macro_tpm.py -x -q`
Expected: FAIL with ImportError on `_background_weights_cause`

- [ ] **Step 3: Implement**

Append to `pyphi/macro/tpm.py`:

```python
def _background_weights_cause(
    factored: FactoredTPM, system_indices, earliest
) -> np.ndarray:
    """``q_c`` (Eq. 34): Bayes posterior over the pre-window background.

    The posterior over the background state one micro update before the
    earliest state of the current window, given that earliest universe
    state, with a uniform prior over the full prior state. Computed from
    the ORIGINAL (undiscounted) TPM.

    Returns:
        np.ndarray: ``(2**|W|,)`` over little-endian background states.
    """
    n = factored.n_nodes
    likelihood = np.ones((2,) * n)
    for i in range(n):
        likelihood = likelihood * factored.factor(i)[..., earliest[i]]
    total = likelihood.sum()
    if total <= 0.0:
        raise exceptions.StateUnreachableBackwardsError(tuple(earliest))
    system_axes = tuple(sorted(system_indices))
    if system_axes:
        posterior = likelihood.sum(axis=system_axes)
    else:
        posterior = likelihood
    return (posterior / total).reshape(-1, order="F")


def _background_weights_effect(background_indices, current_state) -> np.ndarray:
    """``q_e`` (Eq. 33): delta on the current background micro state."""
    weights = np.zeros(2 ** len(background_indices))
    index = 0
    for k, i in enumerate(background_indices):
        index |= current_state[i] << k
    weights[index] = 1.0
    return weights


def _initial_distributions(
    n: int, system_indices, background_weights: np.ndarray
) -> np.ndarray:
    """Initial universe-state distribution per system micro state.

    Returns:
        np.ndarray: ``(2**|U^S|, 2**n)`` — row ``u^S`` is the
        distribution with the system part pinned to ``u^S`` and the
        background part distributed per ``background_weights``.
    """
    background_indices = tuple(
        i for i in range(n) if i not in set(system_indices)
    )
    idx = np.arange(2 ** n)
    system_part = np.zeros(2 ** n, dtype=np.int64)
    for k, i in enumerate(system_indices):
        system_part |= ((idx >> i) & 1) << k
    background_part = np.zeros(2 ** n, dtype=np.int64)
    for k, i in enumerate(background_indices):
        background_part |= ((idx >> i) & 1) << k
    init = np.zeros((2 ** len(system_indices), 2 ** n))
    init[system_part, idx] = background_weights[background_part]
    return init


def macro_tpms(substrate, units, micro_history):
    """The macro cause and effect TPMs ``(T_c, T_e)`` (Eqs. 26-42).

    Args:
        substrate: A binary :class:`~pyphi.substrate.Substrate` for the
            micro universe.
        units: The system's macro units. Their ``U^J ∪ W^J`` must be
            pairwise disjoint (Eq. 18).
        micro_history: Universe micro states, oldest first, of length
            ``max(tau_J)``; the last entry is the current state.

    Returns:
        tuple[FactoredTPM, FactoredTPM]: ``(T_c, T_e)`` with one factor
        per macro unit over the macro system's states.
    """
    factored = substrate.factored_tpm
    n = factored.n_nodes
    units = tuple(units)
    micro_history = tuple(tuple(s) for s in micro_history)
    system_indices = _system_micro_indices(units)
    background_indices = tuple(
        i for i in range(n) if i not in set(system_indices)
    )
    current_state = micro_history[-1]
    num_macro = len(units)
    macro_shape = (2,) * num_macro
    factors_cause = []
    factors_effect = []
    effect_weights = _background_weights_effect(
        background_indices, current_state
    )
    for j, unit in enumerate(units):
        on_probabilities = _discounted_on_probabilities(factored, units, j)
        transition = _full_transition_matrix(on_probabilities)
        macro_prob_full = _unit_macro_probabilities(transition, unit)
        earliest = micro_history[len(micro_history) - unit.micro_grain]
        cause_weights = _background_weights_cause(
            factored, system_indices, earliest
        )
        unit_factors = []
        for background_weights in (cause_weights, effect_weights):
            init = _initial_distributions(
                n, system_indices, background_weights
            )
            prob_given_system_state = init @ macro_prob_full  # (2**|S|, 2)
            factor = np.zeros(macro_shape + (2,))
            for s_index in range(2 ** num_macro):
                macro_state = _mixed_radix_digits(s_index, macro_shape)
                weights = _state_weights(units, system_indices, macro_state)
                factor[macro_state] = weights @ prob_given_system_state
            unit_factors.append(factor)
        factors_cause.append(unit_factors[0])
        factors_effect.append(unit_factors[1])
    return (
        FactoredTPM(factors=factors_cause),
        FactoredTPM(factors=factors_effect),
    )
```

Export from `pyphi/macro/__init__.py`:

```python
from pyphi.macro.tpm import macro_tpms
```

(and add `"macro_tpms"` to `__all__`).

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest test/test_macro_tpm.py test/test_macro_units.py -x -q`
Expected: PASS (including Task 6's `macro_tpms`-dependent tests)

- [ ] **Step 5: Commit**

```bash
git add pyphi/macro/tpm.py pyphi/macro/__init__.py test/test_macro_tpm.py
git -c commit.gpgsign=false commit -m "Add background marginalization and macro_tpms entry point (Eqs 32-40)"
```

---

### Task 8: `MacroSystem`

**Files:**
- Create: `pyphi/macro/system.py`
- Modify: `pyphi/macro/__init__.py`
- Test: `test/test_macro_system.py`

- [ ] **Step 1: Write the failing tests**

Create `test/test_macro_system.py`:

```python
"""Tests for pyphi.macro.system: the MacroSystem protocol implementer."""

import numpy as np
import pytest

from pyphi import config
from pyphi.conf import presets
from pyphi.macro import MacroSystem
from pyphi.macro.units import MacroUnit
from pyphi.macro.units import blackbox
from pyphi.macro.units import coarse_grain
from pyphi.macro.units import micro_unit
from pyphi.substrate import Substrate
from pyphi.system import System

from test.test_macro_tpm import CG_TPM
from test.test_macro_tpm import _asymmetric_substrate


def _cg_macro_system():
    substrate = Substrate(CG_TPM, node_labels=("A", "B", "C", "D"))
    units = (
        MacroUnit((0, 1), 1, coarse_grain(2, on_counts={2})),
        MacroUnit((2, 3), 1, coarse_grain(2, on_counts={2})),
    )
    return MacroSystem.from_micro(substrate, units, ((0, 0, 0, 0),))


class TestConstruction:
    def test_bare_state_wrapped_when_all_grains_one(self):
        substrate = Substrate(CG_TPM)
        units = (
            MacroUnit((0, 1), 1, coarse_grain(2, on_counts={2})),
            MacroUnit((2, 3), 1, coarse_grain(2, on_counts={2})),
        )
        system = MacroSystem.from_micro(substrate, units, (0, 0, 0, 0))
        assert system.micro_history == ((0, 0, 0, 0),)

    def test_bare_state_rejected_with_higher_grain(self):
        substrate = _asymmetric_substrate()
        units = (MacroUnit((0, 1), 2, blackbox(2, 2, (0,))),)
        with pytest.raises(ValueError, match="history"):
            MacroSystem.from_micro(substrate, units, (1, 0, 1, 0))

    def test_history_length_must_be_max_grain(self):
        substrate = _asymmetric_substrate()
        units = (MacroUnit((0, 1), 2, blackbox(2, 2, (0,))),)
        with pytest.raises(ValueError, match="history"):
            MacroSystem.from_micro(
                substrate, units,
                ((1, 0, 1, 0), (1, 0, 1, 0), (1, 0, 1, 0)),
            )

    def test_macro_state_from_history(self):
        system = _cg_macro_system()
        assert system.state == (0, 0)

    def test_eq18_overlap_rejected(self):
        substrate = Substrate(CG_TPM)
        units = (
            MacroUnit((0, 1), 1, coarse_grain(2, on_counts={2})),
            MacroUnit((1, 2), 1, coarse_grain(2, on_counts={2})),
        )
        with pytest.raises(ValueError, match="disjoint"):
            MacroSystem.from_micro(substrate, units, (0, 0, 0, 0))

    def test_apportionment_inside_system_rejected(self):
        substrate = _asymmetric_substrate()
        units = (
            MacroUnit((0, 1), 1, coarse_grain(2, on_counts={2}),
                      background_apportionment=(2,)),
            MacroUnit((2,), 1, (0, 1)),
        )
        with pytest.raises(ValueError, match="background"):
            MacroSystem.from_micro(substrate, units, (1, 0, 1, 0))

    def test_eq12_nested_apportionment_rejected(self):
        substrate = _asymmetric_substrate()
        inner = MacroUnit((0,), 1, (0, 1), background_apportionment=(3,))
        outer = MacroUnit((inner, 1), 1, coarse_grain(2, on_counts={2}))
        with pytest.raises(ValueError, match="Eq. 12"):
            MacroSystem.from_micro(substrate, (outer,), (1, 0, 1, 0))

    def test_nonbinary_substrate_rejected(self):
        from pyphi.core.tpm.factored import FactoredTPM

        # A 1-node ternary substrate, built from a uniform factor
        factor = np.full((3, 3), 1 / 3)
        substrate = Substrate.from_factored(
            FactoredTPM(factors=[factor])
        )
        with pytest.raises(ValueError, match="binary"):
            MacroSystem.from_micro(
                substrate, (micro_unit(0),), ((0,),)
            )

    def test_constituent_outside_substrate_rejected(self):
        substrate = _asymmetric_substrate()
        units = (micro_unit(7),)
        with pytest.raises(ValueError, match="substrate"):
            MacroSystem.from_micro(substrate, units, ((1, 0, 1, 0),))

    def test_from_substrate_directs_to_from_micro(self):
        with pytest.raises(TypeError, match="from_micro"):
            MacroSystem.from_substrate(None, None)


class TestProtocolSurface:
    def test_tpms_and_shape(self):
        system = _cg_macro_system()
        assert system.size == 2
        assert system.node_indices == (0, 1)
        assert np.array_equal(system.cm, np.ones((2, 2)))
        for tpm in (system.cause_tpm, system.effect_tpm):
            for k in range(2):
                assert tpm.factor(k).shape == (2, 2, 2)

    def test_nodes_use_macro_tpms(self):
        system = _cg_macro_system()
        assert len(system.nodes) == 2

    def test_apply_cut_preserves_type_and_fields(self):
        from pyphi.partition import system_partitions

        system = _cg_macro_system()
        partition = next(
            iter(system_partitions(system.node_indices,
                                   system.node_labels))
        )
        cut = system.apply_cut(partition)
        assert isinstance(cut, MacroSystem)
        assert cut.units == system.units
        assert cut.micro_history == system.micro_history
        assert cut.is_partitioned

    def test_equality_and_hash(self):
        a = _cg_macro_system()
        b = _cg_macro_system()
        assert a == b
        assert hash(a) == hash(b)
        substrate = Substrate(CG_TPM, node_labels=("A", "B", "C", "D"))
        c = MacroSystem.from_micro(
            substrate,
            (
                MacroUnit((0, 1), 1, coarse_grain(2, on_counts={1, 2})),
                MacroUnit((2, 3), 1, coarse_grain(2, on_counts={2})),
            ),
            ((0, 0, 0, 0),),
        )
        assert a != c
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest test/test_macro_system.py -x -q`
Expected: FAIL with ImportError on `MacroSystem`

- [ ] **Step 3: Implement**

Create `pyphi/macro/system.py`:

```python
"""MacroSystem: a system of macro units analyzed by the IIT pipeline.

``MacroSystem`` subclasses :class:`~pyphi.system.System` over a
synthetic macro-level :class:`~pyphi.substrate.Substrate` built from the
construction's effect TPM (all-ones connectivity, one binary node per
macro unit). The cause-side TPM properties are overridden with the
construction's cause TPM: the two directions differ in their treatment
of micro background units (Eqs. 33-34) and the cause TPM is therefore
not derivable from the synthetic substrate. Everything else — nodes,
repertoires, partitions, ``sia``/``ces`` — is inherited unchanged, so
the pipeline consumes a ``MacroSystem`` exactly like a ``System``.

Once the macro TPMs are built there is no further reference to the
background units, the units' grains, or their micro constituents; macro
units are perturbed uniformly over their two states like any units.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

import numpy as np

from pyphi.core.tpm.factored import FactoredTPM
from pyphi.substrate import Substrate
from pyphi.system import System

from pyphi.macro.tpm import _system_micro_indices
from pyphi.macro.tpm import macro_tpms
from pyphi.macro.units import MacroUnit


def _validate_units(substrate: Substrate, units: tuple[MacroUnit, ...]) -> None:
    if not units:
        raise ValueError("at least one macro unit is required")
    sizes = substrate.factored_tpm.alphabet_sizes
    if any(size != 2 for size in sizes):
        raise ValueError(
            f"the substrate must be binary; got alphabet sizes {sizes}"
        )
    n = substrate.size
    claimed: set[int] = set()
    for unit in units:
        footprint = set(unit.micro_constituents) | set(
            unit.background_apportionment
        )
        if max(footprint) >= n:
            raise ValueError(
                f"unit references indices outside the substrate "
                f"(size {n}): {sorted(i for i in footprint if i >= n)}"
            )
        if claimed & footprint:
            raise ValueError(
                "units' micro constituents and apportionments must be "
                f"pairwise disjoint (Eq. 18); overlap: "
                f"{sorted(claimed & footprint)}"
            )
        claimed |= footprint
    system = set(_system_micro_indices(units))
    for unit in units:
        if set(unit.background_apportionment) & system:
            raise ValueError(
                "background apportionment must lie outside the system's "
                "micro constituents: "
                f"{sorted(set(unit.background_apportionment) & system)}"
            )
        _validate_nested_apportionment(unit)


def _validate_nested_apportionment(unit: MacroUnit) -> None:
    """Eq. 12: constituents' apportionments nest within their parent's."""
    parent = set(unit.background_apportionment)
    for c in unit.constituents:
        if isinstance(c, MacroUnit):
            if not set(c.background_apportionment) <= parent:
                raise ValueError(
                    "a constituent's background apportionment must be a "
                    "subset of its parent's (Eq. 12); offending indices: "
                    f"{sorted(set(c.background_apportionment) - parent)}"
                )
            _validate_nested_apportionment(c)


def _normalize_history(units, substrate, micro_history):
    max_grain = max(unit.micro_grain for unit in units)
    history = tuple(micro_history)
    if history and not isinstance(history[0], (tuple, list)):
        if max_grain == 1:
            history = (history,)
        else:
            raise ValueError(
                "micro_history must be a sequence of states (oldest "
                f"first) of length {max_grain}; got a bare state"
            )
    history = tuple(tuple(s) for s in history)
    if len(history) != max_grain:
        raise ValueError(
            f"micro_history must have {max_grain} entries (the maximum "
            f"micro grain); got {len(history)}"
        )
    n = substrate.size
    for s in history:
        if len(s) != n or not all(v in (0, 1) for v in s):
            raise ValueError(
                f"each history entry must be a binary universe state of "
                f"length {n}; got {s}"
            )
    return history


def _macro_state(units, history):
    state = []
    for unit in units:
        window = tuple(
            tuple(s[u] for u in unit.micro_constituents)
            for s in history[len(history) - unit.micro_grain:]
        )
        state.append(unit.state_from(window))
    return tuple(state)


@dataclass(frozen=True, eq=False)
class MacroSystem(System):
    """A system of macro units, consumed by the pipeline like a System.

    Construct with :meth:`from_micro`. The inherited ``substrate`` field
    holds the synthetic macro substrate; the micro universe lives in
    ``micro_substrate``.
    """

    units: tuple[MacroUnit, ...] = ()
    micro_substrate: Substrate | None = None
    micro_history: tuple[tuple[int, ...], ...] = ()
    macro_cause_tpm: FactoredTPM | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.micro_substrate is None or not self.units:
            raise TypeError(
                "MacroSystem must be constructed via MacroSystem.from_micro"
            )
        super().__post_init__()

    @classmethod
    def from_micro(
        cls,
        substrate: Substrate,
        units,
        micro_history,
        node_labels=None,
    ) -> MacroSystem:
        """Build a MacroSystem from a micro substrate and macro units.

        Args:
            substrate: The binary micro universe.
            units: The system's macro units (Eq. 18 must hold).
            micro_history: Universe micro states, oldest first, of
                length ``max(tau_J)``. A bare state is accepted when
                every unit has micro grain 1.
            node_labels: Labels for the macro units.
        """
        units = tuple(units)
        _validate_units(substrate, units)
        history = _normalize_history(units, substrate, micro_history)
        cause_tpm, effect_tpm = macro_tpms(substrate, units, history)
        macro_substrate = Substrate.from_factored(
            effect_tpm, node_labels=node_labels
        )
        return cls(
            substrate=macro_substrate,
            state=_macro_state(units, history),
            units=units,
            micro_substrate=substrate,
            micro_history=history,
            macro_cause_tpm=cause_tpm,
        )

    @classmethod
    def from_substrate(cls, *args: Any, **kwargs: Any) -> MacroSystem:
        raise TypeError(
            "MacroSystem cannot be built from a substrate alone; use "
            "MacroSystem.from_micro(substrate, units, micro_history)"
        )

    @property
    def cause_tpm(self) -> FactoredTPM:  # type: ignore[override]
        """The construction's cause TPM (Eqs. 26-40, cause weighting)."""
        assert self.macro_cause_tpm is not None
        return self.macro_cause_tpm

    @property
    def proper_cause_tpm(self) -> FactoredTPM:  # type: ignore[override]
        """Identical to :attr:`cause_tpm`: there is no macro background."""
        return self.cause_tpm

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MacroSystem):
            return NotImplemented
        return (
            self.micro_substrate == other.micro_substrate
            and self.units == other.units
            and self.micro_history == other.micro_history
            and self.partition == other.partition
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.micro_substrate,
                self.units,
                self.micro_history,
                self.partition,
            )
        )
```

Implementation notes for the engineer:

- `System.cause_tpm`/`proper_cause_tpm` are `cached_property` on the
  parent; overriding with plain `property` is correct (the macro TPM is
  already computed). If pyright complains about incompatible override
  types, match the parent's return annotation.
- `dataclasses.replace` (used by the inherited `apply_cut`) reconstructs
  the subclass with all fields, including `macro_cause_tpm`, so
  partitioned copies do not recompute the construction.
- If `validate.state_reachable` (run by `System.__post_init__` under
  `config.infrastructure.validate_system_states`) fails for a macro
  state, that is a real property of the constructed TPMs — do NOT
  suppress it.
- The unused-`np`/`field` imports above are needed (np only if used —
  remove if not; keep imports minimal to satisfy ruff F401).

Export from `pyphi/macro/__init__.py`:

```python
from pyphi.macro.system import MacroSystem
```

(and add `"MacroSystem"` to `__all__`).

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest test/test_macro_system.py test/test_macro_tpm.py -x -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyphi/macro/system.py pyphi/macro/__init__.py test/test_macro_system.py
git -c commit.gpgsign=false commit -m "Add MacroSystem protocol implementer over the macro TPMs"
```

---

### Task 9: Micro-reduction regression (the seam check)

**Files:**
- Test: `test/test_macro_system.py`

- [ ] **Step 1: Write the failing tests**

Append to `test/test_macro_system.py`:

```python
from pyphi.examples import EXAMPLES


def _identity_macro(substrate, state, subset, node_labels):
    units = tuple(micro_unit(i) for i in subset)
    return MacroSystem.from_micro(
        substrate, units, (tuple(state),), node_labels=node_labels
    )


GRID3_TPM_SUBSTRATE = None  # built in the fixture below


def _examples_for_reduction():
    basic = EXAMPLES["substrate"]["basic"]()
    basic_state = (1, 0, 0)
    xor = EXAMPLES["substrate"]["xor"]()
    xor_state = (0, 0, 0)
    # grid3 ships an explicit cm; the macro formalism defines no macro
    # connectivity (all-ones), so compare against an all-ones-cm twin.
    # Substrate.from_factored defaults cm to all-ones.
    grid3 = EXAMPLES["substrate"]["grid3"]()
    grid3_allones = Substrate.from_factored(
        grid3.factored_tpm, node_labels=["A", "B", "C"]
    )
    return [
        ("basic", basic, basic_state),
        ("xor", xor, xor_state),
        ("grid3", grid3_allones, (0, 0, 0)),
    ]


class TestMicroReduction:
    """Identity macroing == System, exactly (paper p. 10)."""

    @pytest.mark.parametrize(
        "name,substrate,state",
        _examples_for_reduction(),
        ids=lambda v: v if isinstance(v, str) else "",
    )
    def test_full_system_sia_matches(self, name, substrate, state):
        with config.override(**presets.iit4_2023):
            subset = tuple(range(substrate.size))
            labels = [str(l) for l in substrate.node_labels]
            macro = _identity_macro(substrate, state, subset, labels)
            micro = System(substrate, state)
            macro_sia = macro.sia()
            micro_sia = micro.sia()
            assert macro_sia.phi == micro_sia.phi
            assert macro_sia.partition == micro_sia.partition

    @pytest.mark.parametrize(
        "name,substrate,state",
        _examples_for_reduction(),
        ids=lambda v: v if isinstance(v, str) else "",
    )
    def test_full_system_ces_matches(self, name, substrate, state):
        with config.override(**presets.iit4_2023):
            subset = tuple(range(substrate.size))
            labels = [str(l) for l in substrate.node_labels]
            macro = _identity_macro(substrate, state, subset, labels)
            micro = System(substrate, state)
            macro_ces = macro.ces()
            micro_ces = micro.ces()
            assert len(macro_ces) == len(micro_ces)
            for m_dist, u_dist in zip(macro_ces, micro_ces, strict=True):
                assert m_dist.mechanism == u_dist.mechanism
                assert m_dist.phi == u_dist.phi

    def test_subset_system_sia_matches(self):
        """Background path: identity macroing of a proper subset."""
        with config.override(**presets.iit4_2023):
            substrate = EXAMPLES["substrate"]["basic"]()
            state = (1, 0, 0)
            subset = (0, 1)
            labels = ["A", "B"]
            macro = _identity_macro(substrate, state, subset, labels)
            micro = System(substrate, state, subset)
            assert macro.sia().phi == micro.sia().phi
```

Adjust attribute names (`mechanism`, `phi`, `partition`) to the actual
`Distinction`/SIA surface if they differ — check
`pyphi/models/mechanism.py` and `pyphi/models/sia.py` before writing;
the assertion intent is: same phi, same partition, same distinction
mechanisms and phis.

- [ ] **Step 2: Run tests to verify behavior**

Run: `uv run pytest test/test_macro_system.py -x -q`
Expected: PASS if the construction is exact. **If any equality fails,
STOP and investigate** — this is the central correctness regression;
diagnose before touching tolerances (exact float equality is expected
because identity macroing must produce bit-identical TPMs; if it
produces only 1e-16-close TPMs, find out why — e.g. a `mean()` over a
single element — and fix the construction, not the test).

- [ ] **Step 3: Commit**

```bash
git add test/test_macro_system.py
git -c commit.gpgsign=false commit -m "Add micro-reduction regression for identity macroing"
```

---

### Task 10: Paper acceptance tests

**Files:**
- Test: `test/test_macro_system.py`

- [ ] **Step 1: Write the tests**

Append to `test/test_macro_system.py` (import `_bbx_micro_tpm` from
`test.test_macro_tpm`):

```python
from test.test_macro_tpm import _bbx_micro_tpm


class TestPaperExample1:
    """Marshall et al. 2024, Example 1 (coarse-graining, Fig. 4)."""

    def test_micro_panel(self):
        with config.override(**presets.iit4_2023):
            substrate = Substrate(CG_TPM, node_labels=("A", "B", "C", "D"))
            state = (0, 0, 0, 0)
            panel = {
                (0,): 0.003976279885291341,
                (0, 1): 0.044088890564147803,
                (0, 1, 2, 3): 0.02015654077792439,
            }
            for nodes, expected in panel.items():
                phi = System(substrate, state, nodes).sia().phi
                assert phi == pytest.approx(expected, abs=1e-13)

    def test_macro_phi_s(self):
        """phi_s of the exact construction TPM.

        The authors' committed value (1.0039763812908649) was computed
        from their hand-entered macro TPM, which contains a rounding
        (0.006833 for 0.0615/9) and a hand-entry error (0.9212 for
        0.9216 = 0.96**2); see test_cg_construction_exact. The value
        below is the 2.0 pipeline's result for the exact construction
        TPM, recorded as this project's golden during planning.
        """
        with config.override(**presets.iit4_2023):
            system = _cg_macro_system()
            assert system.sia().phi == pytest.approx(
                1.0040208141253277, abs=1e-13
            )

    def test_authors_committed_tpm_reproduces_their_phi_s(self):
        """Config-mapping cross-check against the authors' literal TPM."""
        with config.override(**presets.iit4_2023):
            authors_tpm = np.array([
                [0.006833, 0.006833],
                [0.0256, 0.7855],
                [0.7855, 0.0256],
                [0.9212, 0.9212],
            ])
            substrate = Substrate(authors_tpm, node_labels=("a", "b"))
            phi = System(substrate, (0, 0)).sia().phi
            assert phi == pytest.approx(1.0039763812908649, abs=1e-15)

    def test_macro_beats_micro(self):
        with config.override(**presets.iit4_2023):
            system = _cg_macro_system()
            assert system.sia().phi > 0.044088890564147803


class TestPaperExample2:
    """Marshall et al. 2024, Example 2 (black-boxing, Fig. 5)."""

    def _macro_system(self):
        substrate = Substrate(_bbx_micro_tpm(),
                              node_labels=tuple("ABCDEFGH"))
        units = (
            MacroUnit((0, 1, 2, 3), 2, blackbox(4, 2, (2,))),
            MacroUnit((4, 5, 6, 7), 2, blackbox(4, 2, (2,))),
        )
        ones = (1,) * 8
        return MacroSystem.from_micro(substrate, units, (ones, ones))

    def test_macro_phi_s(self):
        """The strong anchor: the authors computed this TPM (not
        hand-entered), the construction matches it to 1e-16, and the
        committed phi_s reproduces bit-for-bit under the mapped config.
        """
        with config.override(**presets.iit4_2023):
            assert self._macro_system().sia().phi == pytest.approx(
                1.1183776016500528, abs=1e-13
            )

    @pytest.mark.slow
    def test_micro_panel(self):
        with config.override(**presets.iit4_2023):
            substrate = Substrate(_bbx_micro_tpm(),
                                  node_labels=tuple("ABCDEFGH"))
            ones = (1,) * 8
            panel = {
                (0, 2, 4, 6): 0.135185781056239,
                (0, 1, 2, 3): 0.02998866492258486,
            }
            for nodes, expected in panel.items():
                phi = System(substrate, ones, nodes).sia().phi
                assert phi == pytest.approx(expected, abs=1e-13)

    def test_macro_beats_micro(self):
        with config.override(**presets.iit4_2023):
            assert self._macro_system().sia().phi > 0.135185781056239
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest test/test_macro_system.py -x -q` then
`uv run pytest test/test_macro_system.py -m slow -x -q`
Expected: PASS. The planning experiments already confirmed every number;
a failure indicates an implementation deviation, not a wrong anchor.

- [ ] **Step 3: Commit**

```bash
git add test/test_macro_system.py
git -c commit.gpgsign=false commit -m "Add paper acceptance tests for Examples 1 and 2"
```

---

### Task 11: Changelog, ROADMAP, API docs

**Files:**
- Create: `changelog.d/intrinsic-units-macro.feature.md`, `docs/api/macro.rst`
- Modify: `ROADMAP.md` (the "Macro framework — Marshall 2024 intrinsic units" entry, ~line 1615), `docs/api/index.rst` (if it has a toctree)

- [ ] **Step 1: Changelog fragment**

```bash
cat > changelog.d/intrinsic-units-macro.feature.md << 'EOF'
Added `pyphi.macro`: the intrinsic-units macro framework of Marshall,
Findlay, Albantakis & Tononi (2024). `MacroUnit` defines a macro unit by
its constituents (micro or meso), update grain, sliding-window state
mapping, and background apportionment; `coarse_grain()` and `blackbox()`
build the paper's two mapping classes; `macro_tpms()` implements the
four-step macro TPM construction (Eqs. 26-40); and `MacroSystem` exposes
the result to the standard IIT 4.0 pipeline (`sia()`, `ces()`,
relations) exactly like a micro `System`. Identity macroing reproduces
micro results exactly, and both paper examples are reproduced at the
authors' published precision. The legacy pre-2024 `pyphi.macro` module
(`CoarseGrain`/`Blackbox`/`MacroSubsystem`) is removed.
EOF
```

- [ ] **Step 2: ROADMAP update**

Rewrite the deferred macro entry (item 10, ~line 1615): now in scope for
2.0 ahead of the P15 surface freeze; sub-project structure (SP1
machinery — this work; SP2 intrinsic-unit criteria Eqs 15-16/19 and
search; SP3 reference goldens from the authors' result sets); note the
config mapping (`DIRECTED_SET_PARTITION` ≡ legacy `SET_UNI/BI`,
confirmed bit-for-bit) and the documented discrepancy in the authors'
hand-entered Example 1 TPM. No machine-specific paths.

- [ ] **Step 3: API docs page**

Create `docs/api/macro.rst` mirroring a sibling page's structure
(e.g. `docs/api/` for another package module), with `automodule`
entries for `pyphi.macro.units`, `pyphi.macro.tpm`,
`pyphi.macro.system`. Add it back to any API toctree it was removed
from in Task 1.

- [ ] **Step 4: Commit**

```bash
git add changelog.d/intrinsic-units-macro.feature.md ROADMAP.md docs/
git -c commit.gpgsign=false commit -m "Document the intrinsic-units macro framework"
```

---

### Task 12: Full verification

- [ ] **Step 1: Full suite (no path argument — includes doctest sweep)**

Run in background (`run_in_background=true`):
`uv run pytest`
Expected: baseline 1908 passed / 41 skipped / 3 xfailed, minus the
deleted legacy tests, plus the new macro tests; zero failures.

- [ ] **Step 2: Fast checks while the suite runs**

```bash
uv run ruff check pyphi/macro test/test_macro_units.py test/test_macro_tpm.py test/test_macro_system.py
uv run pyright pyphi/macro
```

(Note: an unstaged `typeCheckingMode = "off"` workaround may be present
in `pyproject.toml`; the pre-commit hook is the authoritative pyright
gate.)

- [ ] **Step 3: Confirm no stray staged files; report results to the user**

---

## Self-review notes

- **Spec coverage:** MacroUnit five-tuple with derived U^J (Task 2-3);
  truth-table convention + mixed-radix hedge (Tasks 2-4); constructors
  (Task 4); four-step construction with fused Steps 2+4 (Tasks 5-7);
  Eq 29 unit-test-only coverage flagged (Task 7); MacroSystem protocol
  seam, cm all-ones, bare-state wrapping, Eq 12/18 + surjectivity +
  history validation (Task 8); micro-equivalence battery (Task 9); paper
  acceptance at authors' precision (Task 10); legacy deletion (Task 1);
  changelog + ROADMAP (Task 11). Six test batteries of the spec all
  present.
- **Known deviations from spec, for user sign-off:** (1) `MacroSystem`
  subclasses `System` instead of standalone-implementing the protocol —
  rationale in File structure section. (2) The spec's acceptance anchor
  φ_s({α,β}) = 1.0039763812908649 for Example 1 is reproducible only
  from the authors' hand-entered TPM (kept as a config cross-check);
  the construction golden is 1.0040208141253277 (authors' TPM has a
  documented hand-entry error). (3) `blackbox()` with multiple outputs
  is defined as ALL-outputs-ON (the paper only exercises a single
  output; SP2's enumeration may generalize).
- **Type consistency:** `macro_tpms(substrate, units, micro_history)`
  returns `(T_c, T_e)` — order pinned everywhere (cause first, matching
  the spec). `MacroUnit.micro_grain`/`micro_constituents`/`state_from`
  names consistent across tasks. Test helpers `_asymmetric_substrate`,
  `_flat_on_probabilities`, `CG_TPM`, `_bbx_micro_tpm` shared via
  imports from `test_macro_tpm`.
