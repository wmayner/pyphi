# 2.0.0 Release Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the scientific-fidelity gaps found in the final paper-to-code read before the 2.0.0 tag: name the 2026 paper's quantities, add an `explain()` finding for the intrinsic-information requirement, add `Substrate.inactivate`, pin the published worked examples of four more papers, and finish the "requirement" terminology sweep.

**Architecture:** Three small additions to the frozen IIT 4.0 result surface (a property alias, a display label, an `Explanation` finding), one substrate method, then tests-only work: new example fixtures in `pyphi/examples.py` and paper-sourced pins in `test/integration/`. The surface tasks (2–6) run sequentially and first, because the reproduction tests consume the new accessors; tasks 7–17 are independent of one another and can run in parallel on disjoint files.

**Tech Stack:** Python 3.13, numpy, pytest (`uv run pytest`), the existing `pyphi.conf.presets` pinning convention, `pyphi.substrate_generator.build_substrate` with `ising.probability` for logistic units.

**Spec:** `docs/superpowers/specs/2026-09-11-release-pass-design.md`

## Global Constraints

- Every φ-asserting test pins its formalism with a whole preset from `pyphi.conf.presets` (`iit4_2026`, `iit4_2023`, `iit3`), never a hand-listed subset of `iit.*` fields (see `CLAUDE.md` §"Formalism pinning").
- Every pinned number is quoted from the paper with its figure/equation number in the test docstring; PyPhi is never the source of a pin.
- Every pin is proven to fail when its expected value is perturbed before it is committed (change the expected value by more than the tolerance, run, see FAIL, restore); the commit message says so ("perturbation-verified").
- Where a reconstructed fixture cannot match a published value, pin what matches and document the deviation in the docstring (spec §5). Never force a value.
- Tests slower than ~10 s carry `@pytest.mark.slow`; run the slow lane with `uv run pytest -m slow --slow`.
- Never pipe pytest through `tail`/`grep`: `uv run pytest … -q > <scratch>/log 2>&1; echo $?` then read the summary line.
- Docstrings: NumPy style, final-state voice, Unicode symbols (`pyphi/CLAUDE.md`).
- Every user-facing change gets a `changelog.d/<name>.<type>.md` fragment.
- `uvx ruff format` + `uvx ruff check` before every commit; pre-commit hooks run ruff and pyright; never `--no-verify`.
- Terminology: "intrinsic-information requirement", "with/without the requirement". Never "cap", "capped", "uncapped" in prose.
- Environment variables for any pyphi script: `PYPHI_WELCOME_OFF=1 PYPHI_AGENT_NOTE_OFF=1`.
- Commit messages end with the attribution trailer from the session (`Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` + `Claude-Session:` line).

---

### Task 1: Worktree and branch

**Files:** none (git only)

- [ ] **Step 1: Create the worktree**

```bash
cd /Users/will/projects/pyphi
git worktree add .claude/worktrees/release-pass -b release-pass main
cd .claude/worktrees/release-pass
uv sync --all-extras > /dev/null 2>&1; echo "sync exit: $?"
```

- [ ] **Step 2: Verify the baseline is green on one fast file**

```bash
uv run pytest test/models/test_explanation.py -q > /tmp/baseline.log 2>&1; echo $?; tail -1 /tmp/baseline.log
```
Expected: exit 0, "N passed".

Note for every later task: the worktree's venv is used by `uv run` from inside the worktree; `uv pip install` from inside a worktree targets the *main* venv (see memory) — do not install anything, only `uv run`.

---

### Task 2: Rename `applies_ii_cap` → `applies_intrinsic_information_requirement`

**Files:**
- Modify: `pyphi/measures/protocols.py:70-81`
- Modify: `pyphi/measures/distribution.py:395-440, 1363-1395`
- Modify: `pyphi/formalism/iit4/__init__.py:760, 1056, 1324`
- Modify: `pyphi/formalism/base.py:76-84` (rename `requires_ii_cap` → `applies_intrinsic_information_requirement` on the formalism Protocol too — same rationale as D4; one name for both Protocols)
- Modify: `pyphi/formalism/iit4/formalism.py` (the `requires_ii_cap` class attribute on `IIT4_2026Formalism` / `IIT4_2023Formalism`)
- Modify: `pyphi/macro/search.py:370`, `pyphi/conf/constraints.py:405-415`
- Test: `test/measures/test_measure_protocols.py:143-160`

**Interfaces:**
- Produces: `CompositeMeasure.applies_intrinsic_information_requirement: bool`; `PhiFormalism.applies_intrinsic_information_requirement: bool`. Task 5 reads the measure attribute.

- [ ] **Step 1: Find every site**

```bash
grep -rn "applies_ii_cap\|requires_ii_cap" pyphi test docs --include='*.py' --include='*.md' | grep -v "_build/"
```
Expected: the sites listed above plus any docs mention; note them all.

- [ ] **Step 2: Update the Protocol test first**

In `test/measures/test_measure_protocols.py`, replace `applies_ii_cap` with `applies_intrinsic_information_requirement` in the parametrized attribute test (lines ~143–160), including the comment.

- [ ] **Step 3: Run it to verify it fails**

```bash
uv run pytest test/measures/test_measure_protocols.py -q 2>&1 | tail -3
```
Expected: FAIL with `AttributeError`/assertion on the missing attribute.

- [ ] **Step 4: Rename in the source**

Mechanical rename of the identifier at every site from Step 1 (`applies_ii_cap` and `requires_ii_cap` both become `applies_intrinsic_information_requirement`). Update the surrounding docstring sentences to read "applies the intrinsic-information requirement (Mayner et al. 2026, Eq. 23)". In `pyphi/measures/distribution.py` the `register` decorator's keyword becomes `applies_intrinsic_information_requirement: bool = False`, and the `INTRINSIC_INFORMATION` registration passes `applies_intrinsic_information_requirement=True`.

- [ ] **Step 5: Run the affected suites**

```bash
uv run pytest test/measures test/formalism/test_formalism.py test/formalism/test_formalism_config.py test/conf test/macro -q > /tmp/t2.log 2>&1; echo $?; tail -1 /tmp/t2.log
grep -rn "applies_ii_cap\|requires_ii_cap" pyphi test docs | grep -v _build/
```
Expected: exit 0; the grep prints nothing.

- [ ] **Step 6: Changelog fragment and commit**

```bash
cat > changelog.d/applies-ii-requirement-rename.change.md <<'EOF'
Renamed the measure and formalism Protocol attribute `applies_ii_cap` (and the formalism's `requires_ii_cap`) to `applies_intrinsic_information_requirement`, matching the project's terminology for Mayner et al. (2026) Eq. 23.
EOF
uvx ruff format pyphi test && uvx ruff check pyphi test
git add -A pyphi test changelog.d
git commit -m "Rename applies_ii_cap to applies_intrinsic_information_requirement

One name on both the measure and formalism Protocols for the attribute
that says whether Mayner et al. (2026) Eq. 23 is applied."
```

---

### Task 3: `intrinsic_specification` accessors

**Files:**
- Modify: `pyphi/models/state_specification.py:79-135` (add property)
- Modify: `pyphi/formalism/iit4/__init__.py:292-326, 405-420` (add property; pandas columns)
- Test: `test/formalism/test_sia_accessors.py` (append)

**Interfaces:**
- Produces: `StateSpecification.intrinsic_specification -> float | DistanceResult` (same value as `intrinsic_information`); `SystemIrreducibilityAnalysis.intrinsic_specification -> dict[Direction, float | None]`. Tasks 5, 7, 10, 11 consume these.

- [ ] **Step 1: Write the failing tests**

Append to `test/formalism/test_sia_accessors.py`:

```python
def test_state_specification_intrinsic_specification_is_the_2023_intrinsic_information(s):
    """Mayner et al. (2026) rename the 2023 per-state intrinsic information
    (Albantakis et al. 2023 Eqs. 5/7) to intrinsic specification (2026 Eqs. 7/9)."""
    from pyphi.formalism import FORMALISM_REGISTRY

    sia = FORMALISM_REGISTRY["IIT_4_0_2023"].evaluate_system(s)
    for direction in Direction.both():
        spec = sia.system_state[direction]
        assert float(spec.intrinsic_specification) == float(spec.intrinsic_information)


def test_sia_intrinsic_specification_is_per_direction(s):
    from pyphi.formalism import FORMALISM_REGISTRY

    sia = FORMALISM_REGISTRY["IIT_4_0_2023"].evaluate_system(s)
    spec = sia.intrinsic_specification
    assert set(spec) == set(Direction.both())
    for direction in Direction.both():
        assert spec[direction] == float(sia.system_state[direction].intrinsic_information)
    # Shape-parallel to the differentiation dict, and ii(s) is their joint minimum.
    assert set(sia.intrinsic_differentiation) == set(spec)
    terms = [max(0.0, v) for v in spec.values()] + [
        max(0.0, float(v)) for v in sia.intrinsic_differentiation.values()
    ]
    assert sia.intrinsic_information == pytest.approx(min(terms))


def test_sia_intrinsic_specification_none_on_null_sia(s_empty):
    from pyphi.formalism import FORMALISM_REGISTRY

    sia = FORMALISM_REGISTRY["IIT_4_0_2023"].evaluate_system(s_empty)
    assert sia.intrinsic_specification == {
        Direction.CAUSE: None,
        Direction.EFFECT: None,
    }


def test_sia_pandas_has_per_direction_specification_and_differentiation(s):
    from pyphi.formalism import FORMALISM_REGISTRY

    sia = FORMALISM_REGISTRY["IIT_4_0_2023"].evaluate_system(s)
    record = sia.to_pandas()
    for column in (
        "cause_intrinsic_specification",
        "effect_intrinsic_specification",
        "cause_intrinsic_differentiation",
        "effect_intrinsic_differentiation",
    ):
        assert column in record.index
    assert record["cause_intrinsic_specification"] == pytest.approx(
        sia.intrinsic_specification[Direction.CAUSE]
    )
```

Add `import pytest` and `from pyphi.direction import Direction` at the top if the file lacks them (check the existing imports first). `s` and `s_empty` are root-conftest fixtures (`test/conftest.py:166-175`); `to_pandas()` on a scalar record returns a `Series` indexed by column name (see `_pandas_record`).

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest test/formalism/test_sia_accessors.py -q -k intrinsic_specification 2>&1 | tail -3
```
Expected: FAIL with `AttributeError: ... has no attribute 'intrinsic_specification'`.

- [ ] **Step 3: Implement**

In `pyphi/models/state_specification.py`, after `state_margin`:

```python
    @property
    def intrinsic_specification(self) -> float | DistanceResult:
        """The intrinsic specification of the specified state.

        The same value as :attr:`intrinsic_information`: the product of
        selectivity and informativeness for the specified state. Albantakis
        et al. (2023, Eqs. 5 and 7) call this quantity intrinsic
        information; Mayner et al. (2026, Eqs. 7 and 9) rename it intrinsic
        specification and reserve *intrinsic information* for its minimum
        with the intrinsic differentiation (2026, Eq. 13).
        """
        return self.intrinsic_information
```

Also add the attribute line to the class docstring's Attributes list. In `pyphi/formalism/iit4/__init__.py`, after the `intrinsic_information` property on `SystemIrreducibilityAnalysis`:

```python
    @property
    def intrinsic_specification(self) -> dict[Direction, float | None]:
        """Per-direction intrinsic specification of the specified system
        state (Mayner et al. 2026, Eqs. 7 and 9).

        Parallel in shape to :attr:`intrinsic_differentiation`; together
        the two give ``ii(s) = min over directions of
        min(i_spec, i_diff)`` (2026, Eq. 13), exposed as
        :attr:`intrinsic_information`. An entry is ``None`` when that
        direction's state was not specified (null analyses).
        """
        out: dict[Direction, float | None] = {}
        for direction in Direction.both():
            spec = (
                self.system_state[direction]
                if self.system_state is not None
                else None
            )
            out[direction] = (
                float(spec.intrinsic_specification) if spec is not None else None
            )
        return out
```

In `_pandas_record`, after `"integrated_fraction"`:

```python
            "cause_intrinsic_specification": self.intrinsic_specification[Direction.CAUSE],
            "effect_intrinsic_specification": self.intrinsic_specification[Direction.EFFECT],
            "cause_intrinsic_differentiation": _optional_float(
                (self.intrinsic_differentiation or {}).get(Direction.CAUSE)
            ),
            "effect_intrinsic_differentiation": _optional_float(
                (self.intrinsic_differentiation or {}).get(Direction.EFFECT)
            ),
```

- [ ] **Step 4: Run**

```bash
uv run pytest test/formalism/test_sia_accessors.py test/models test/test_display.py -q > /tmp/t3.log 2>&1; echo $?; tail -1 /tmp/t3.log
```
Expected: exit 0. If `test/test_display.py` or a pandas coverage test pins the SIA column set, update that pin (the four new columns are intended).

- [ ] **Step 5: Commit**

```bash
cat > changelog.d/intrinsic-specification.feature.md <<'EOF'
Added `intrinsic_specification` on `StateSpecification` and, per direction, on the IIT 4.0 `SystemIrreducibilityAnalysis` — the name Mayner et al. (2026, Eqs. 7 and 9) give the quantity Albantakis et al. (2023) call intrinsic information — alongside the existing `intrinsic_differentiation`, so both terms of the intrinsic-information requirement (2026, Eq. 13) are readable by their paper names. `to_pandas()` on the SIA gains the four per-direction columns.
EOF
uvx ruff format pyphi test && uvx ruff check pyphi test
git add pyphi/models/state_specification.py pyphi/formalism/iit4/__init__.py test/formalism/test_sia_accessors.py changelog.d/intrinsic-specification.feature.md
git commit -m "Expose intrinsic specification by its 2026 name on state specifications and the SIA"
```

---

### Task 4: Formalism-aware display label for the specification row

**Files:**
- Modify: `pyphi/display/description.py:13-40` (add `intrinsic_specification_label`)
- Modify: `pyphi/formalism/iit4/__init__.py:436-464` (`_describe` rows)
- Test: `test/test_display.py` (append)

- [ ] **Step 1: Read `system_phi_label` in full**

```bash
sed -n '13,45p' pyphi/display/description.py
```
Note how it reads the version from the snapshot (`config.formalism.iit.version`) and its `None` fallback; mirror that exactly.

- [ ] **Step 2: Write the failing test**

Append to `test/test_display.py`:

```python
def test_specification_row_label_follows_formalism(s):
    """The per-direction row is "Intrinsic specification" under IIT 4.0 (2026)
    and "Intrinsic information" under IIT 4.0 (2023), the papers' own names."""
    from pyphi.conf import config, presets
    from pyphi.formalism import FORMALISM_REGISTRY

    with config.override(**presets.iit4_2026):
        card_2026 = str(FORMALISM_REGISTRY["IIT_4_0_2026"].evaluate_system(s))
    with config.override(**presets.iit4_2023):
        card_2023 = str(FORMALISM_REGISTRY["IIT_4_0_2023"].evaluate_system(s))
    assert "Intrinsic specification" in card_2026
    assert "Intrinsic specification" not in card_2023
    assert "Intrinsic information" in card_2023
```

- [ ] **Step 3: Run to verify failure**

```bash
uv run pytest test/test_display.py -q -k specification_row_label 2>&1 | tail -3
```
Expected: FAIL (`"Intrinsic specification" in card_2026` is False).

- [ ] **Step 4: Implement**

In `pyphi/display/description.py`, after `system_phi_label`:

```python
def intrinsic_specification_label(config: Any) -> str:
    """The label for a specified state's selectivity-times-informativeness
    value under ``config``'s formalism.

    Mayner et al. (2026, Eqs. 7 and 9) call it intrinsic specification;
    Albantakis et al. (2023, Eqs. 5 and 7) call the same quantity intrinsic
    information. ``None`` falls back to the 2026 label, the library default.
    """
    version = None
    if config is not None:
        try:
            version = config.formalism.iit.version
        except AttributeError:
            version = None
    if version == "IIT_4_0_2023":
        return "Intrinsic information"
    return "Intrinsic specification"
```

(Adjust the attribute path to whatever `system_phi_label` uses.) In `_describe` of the SIA, replace both `Row("Intrinsic information", ...)` labels with `Row(intrinsic_specification_label(self.config), ...)`, importing the helper next to `system_phi_label`'s import.

- [ ] **Step 5: Run the display suite**

```bash
uv run pytest test/test_display.py test/models -q > /tmp/t4.log 2>&1; echo $?; tail -1 /tmp/t4.log
```
Expected: exit 0. If the SIA card golden in `test/test_display.py` was rendered under the 2026 default, its expected text changes from "Intrinsic information" to "Intrinsic specification" — update the golden text (that is the intended change) and say so in the commit.

- [ ] **Step 6: Commit**

```bash
uvx ruff format pyphi test && uvx ruff check pyphi test
git add pyphi/display/description.py pyphi/formalism/iit4/__init__.py test/test_display.py
git commit -m "Label the specified-state row by formalism: specification (2026) or information (2023)"
```

---

### Task 5: `explain()` finding when the requirement binds

**Files:**
- Modify: `pyphi/models/explanation.py:177-197` (add `requirement_binding_finding`)
- Modify: `pyphi/formalism/iit4/__init__.py:493-558` (`_findings`)
- Test: `test/models/test_explanation.py` (append)

**Interfaces:**
- Consumes: `applies_intrinsic_information_requirement` (Task 2), `intrinsic_specification` (Task 3).
- Produces: a `Finding` with `kind="requirement_binding"`, `value` in `{"differentiation", "specification"}`, `detail=(("direction", "CAUSE"|"EFFECT"), ("ii", float), ("φ_s", float))`, `tone` = the direction. Task 8 asserts on `value`.

- [ ] **Step 1: Write the failing tests**

Append to `test/models/test_explanation.py`:

```python
class TestRequirementBindingFinding:
    """Under IIT 4.0 (2026), φₛ = min{φ_c, φ_e, ii(s)}; when ii(s) is the
    minimum, explain() names the direction and term that set it."""

    def _sia(self, preset_name):
        from pyphi.conf import config, presets
        from pyphi.formalism import FORMALISM_REGISTRY
        from pyphi import examples

        # The Fig 1A logistic network: the requirement lowers aB's φ_s from
        # the published 0.17 to about 0.04 (docs/theory/intrinsic-information.md).
        system = examples.iit4_2023_fig1a_system()
        with config.override(**presets.by_name[preset_name], validate_system_states=False):
            return FORMALISM_REGISTRY[preset_name].evaluate_system(system)

    def test_fires_when_ii_is_the_minimum_under_2026(self):
        import pytest
        from pyphi import numerics

        sia = self._sia("IIT_4_0_2026")
        assert numerics.eq(float(sia.phi), sia.intrinsic_information)
        finding = next(f for f in sia.explain().findings if f.kind == "requirement_binding")
        assert finding.value in {"differentiation", "specification"}
        detail = dict(finding.detail)
        assert detail["direction"] in {"CAUSE", "EFFECT"}
        assert detail["ii"] == pytest.approx(sia.intrinsic_information)
        # The named term is the one whose value equals ii(s).
        from pyphi.direction import Direction

        direction = Direction[detail["direction"]]
        term = (
            sia.intrinsic_differentiation[direction]
            if finding.value == "differentiation"
            else sia.intrinsic_specification[direction]
        )
        assert max(0.0, float(term)) == pytest.approx(sia.intrinsic_information)

    def test_never_fires_under_2023(self):
        sia = self._sia("IIT_4_0_2023")
        assert not any(f.kind == "requirement_binding" for f in sia.explain().findings)

    def test_absent_when_integration_binds(self):
        """A system whose φ_s is strictly below ii(s) carries no finding."""
        from pyphi.conf import config, presets
        from pyphi.formalism import FORMALISM_REGISTRY
        from pyphi import examples, numerics

        with config.override(**presets.iit4_2026, validate_system_states=False):
            sia = FORMALISM_REGISTRY["IIT_4_0_2026"].evaluate_system(
                examples.grid3_system()
            )
        if numerics.eq(float(sia.phi), sia.intrinsic_information):
            import pytest

            pytest.skip("grid3 happens to bind on ii; pick another fixture")
        assert not any(f.kind == "requirement_binding" for f in sia.explain().findings)
```

(`grid3_system` and `iit4_2023_fig1a_system` are registered examples; if `iit4_2023_fig1a_system` does not accept the default `subset`, use `System(examples.iit4_2023_fig1a_substrate(), (0, 1, 1), node_indices=(0, 1))` — the aB system of the paper.) If `grid3` skips, replace it with `examples.basic_noisy_selfloop_system()` and re-run; leave the skip guard so the test never silently passes.

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest test/models/test_explanation.py -q -k RequirementBinding 2>&1 | tail -3
```
Expected: `test_fires_when_ii_is_the_minimum_under_2026` FAILS with `StopIteration` (no such finding); the other two pass vacuously.

- [ ] **Step 3: Implement the finding constructor**

In `pyphi/models/explanation.py` after `binding_direction_finding`:

```python
def requirement_binding_finding(
    phi: Any,
    intrinsic_information: Any,
    specification: Mapping[Any, Any],
    differentiation: Mapping[Any, Any],
) -> Finding | None:
    """The Finding naming which term of the intrinsic-information requirement
    set φₛ, or ``None`` when the requirement does not bind.

    Under Mayner et al. (2026, Eq. 23), φₛ = min{φ_c, φ_e, ii(s)} with
    ii(s) = min over directions of min(i_spec, i_diff) (Eq. 13). When φₛ
    equals ii(s) up to ``config.numerics.precision``, the finding reports
    the direction and the term (``"specification"`` or
    ``"differentiation"``) whose rectified value equals ii(s).
    """
    if intrinsic_information is None or not numerics.eq(
        float(phi), float(intrinsic_information)
    ):
        return None
    for direction in specification:
        for term_name, values in (
            ("differentiation", differentiation),
            ("specification", specification),
        ):
            value = values.get(direction)
            if value is None:
                continue
            if numerics.eq(
                max(0.0, float(value)), float(intrinsic_information)
            ):
                tone = "cause" if direction.name == "CAUSE" else "effect"
                return Finding(
                    kind="requirement_binding",
                    label="Intrinsic-information requirement binds",
                    value=term_name,
                    detail=(
                        ("direction", direction.name),
                        ("ii", float(intrinsic_information)),
                        ("φ_s", float(phi)),
                    ),
                    tone=tone,
                )
    return None
```

Add `from collections.abc import Mapping` to the imports. Then in `SystemIrreducibilityAnalysis._findings` (iit4/`__init__.py`), after the `binding_direction_finding` append:

```python
        if self._applies_requirement():
            finding = requirement_binding_finding(
                self.phi,
                self.intrinsic_information,
                self.intrinsic_specification,
                self.intrinsic_differentiation or {},
            )
            if finding is not None:
                findings.append(finding)
```

and the helper on the class:

```python
    def _applies_requirement(self) -> bool:
        """Whether the result's own configuration applies Eq. 23."""
        if self.config is None:
            return False
        from pyphi.measures.distribution import resolve_system_measure

        try:
            measure = resolve_system_measure(self.config.formalism.iit.system_phi_measure)
        except (AttributeError, KeyError):
            return False
        return bool(getattr(measure, "applies_intrinsic_information_requirement", False))
```

Import `requirement_binding_finding` next to `binding_direction_finding`.

- [ ] **Step 4: Run**

```bash
uv run pytest test/models/test_explanation.py -q > /tmp/t5.log 2>&1; echo $?; tail -1 /tmp/t5.log
```
Expected: exit 0. `test_explain_is_total` enumerates finding kinds — if it pins the allowed set, add `"requirement_binding"`.

- [ ] **Step 5: Commit**

```bash
cat > changelog.d/explain-requirement-binding.feature.md <<'EOF'
`explain()` on an IIT 4.0 system analysis now reports when the intrinsic-information requirement (Mayner et al. 2026, Eq. 23) set φₛ, naming the direction and the term — intrinsic differentiation or intrinsic specification — whose value is the minimum. The finding never fires under formalisms without the requirement.
EOF
uvx ruff format pyphi test && uvx ruff check pyphi test
git add pyphi/models/explanation.py pyphi/formalism/iit4/__init__.py test/models/test_explanation.py changelog.d/explain-requirement-binding.feature.md
git commit -m "Report in explain() when the intrinsic-information requirement binds"
```

---

### Task 6: `Substrate.inactivate`

**Files:**
- Modify: `pyphi/substrate.py:605-665` (add method near the export helpers)
- Modify: `pyphi/examples.py:1805-1820` (rewrite `iit4_2023_fig7_inactivated_substrate`)
- Test: `test/test_substrate.py` (append; create the file if absent — check `ls test/test_substrate*.py`)

**Interfaces:**
- Produces: `Substrate.inactivate(fixed: Mapping[int | str, int]) -> Substrate`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from pyphi import examples
from pyphi.substrate import Substrate


class TestInactivate:
    def test_matches_manual_conditioning(self):
        substrate = examples.iit4_2023_fig7_substrate()
        manual = Substrate.from_factored(
            substrate.factored_tpm.condition({4: 0}),
            node_labels=("A", "B", "C", "D", "E"),
        )
        assert substrate.inactivate({4: 0}) == manual
        assert substrate.inactivate({"E": 0}) == manual

    def test_labels_preserved(self):
        substrate = examples.iit4_2023_fig7_substrate()
        assert tuple(substrate.inactivate({"E": 0}).node_labels) == tuple(
            substrate.node_labels
        )

    def test_frozen_unit_no_longer_inputs_to_others(self):
        substrate = examples.iit4_2023_fig7_substrate()
        cm = np.asarray(substrate.inactivate({"E": 0}).cm)
        assert not cm[4, :4].any()

    def test_unknown_unit_and_bad_state_raise(self):
        substrate = examples.iit4_2023_fig7_substrate()
        with pytest.raises((ValueError, KeyError)):
            substrate.inactivate({"Z": 0})
        with pytest.raises(ValueError):
            substrate.inactivate({"E": 2})

    def test_fig7c_reproduces_through_the_method(self):
        """Albantakis et al. (2023) Fig 7C: with E inactivated the complex
        shrinks to {A, B, C, D} (see test_paper_reproduction.py)."""
        assert examples.iit4_2023_fig7_inactivated_substrate() == (
            examples.iit4_2023_fig7_substrate().inactivate({"E": 0})
        )
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest test/test_substrate.py -q -k Inactivate 2>&1 | tail -3
```
Expected: FAIL with `AttributeError: 'Substrate' object has no attribute 'inactivate'`.

- [ ] **Step 3: Implement**

In `pyphi/substrate.py`, in the `Substrate` class after `to_dbn_dict`:

```python
    def inactivate(self, fixed: Mapping[int | str, int]) -> Substrate:
        """Return a copy with the given units frozen in a state.

        Each unit in ``fixed`` (by index or label) is conditioned into every
        other unit's transition factor at the given state, so it has no
        counterfactual states and cannot be intervened upon. Node labels
        are preserved. Inputs from a frozen unit become fixed biases and
        disappear from the inferred connectivity.

        Albantakis et al. (2023, Fig 7) distinguish an *inactive* unit —
        in its OFF state, still contributing distinctions and relations —
        from an *inactivated* one, whose cause–effect power is abolished
        (Fig 7C): the complex that contained it shrinks. Inactivation is
        also distinct from holding a unit as a background condition of a
        candidate system: a background unit keeps its counterfactual states
        and is causally marginalized (2023, Eqs. 3–4); an inactivated unit
        has none.

        Parameters
        ----------
        fixed : Mapping[int or str, int]
            Units (indices or labels) mapped to the state index each is
            frozen in.

        Returns
        -------
        Substrate

        Raises
        ------
        ValueError
            If a unit is unknown or a state is outside the unit's alphabet.

        Examples
        --------
        >>> from pyphi import examples
        >>> lesioned = examples.iit4_2023_fig7_substrate().inactivate({"E": 0})
        >>> lesioned.size
        5
        """
        units = tuple(fixed)
        indices = self.node_labels.coerce_to_indices(units)
        frozen: dict[int, int] = {}
        for index, state in zip(indices, fixed.values(), strict=True):
            alphabet = len(self.state_space[index])
            if not isinstance(state, (int, np.integer)) or not 0 <= state < alphabet:
                raise ValueError(
                    f"state {state!r} for unit {index} is not a valid index for "
                    f"alphabet size {alphabet}"
                )
            frozen[int(index)] = int(state)
        return type(self).from_factored(
            self.factored_tpm.condition(frozen), node_labels=self.node_labels
        )
```

Check `NodeLabels.coerce_to_indices` (`pyphi/labels.py:126`) for the error it raises on an unknown label and match the test's `(ValueError, KeyError)` accordingly (narrow the test if it is one specific type). Add `from collections.abc import Mapping` if not imported.

Then in `pyphi/examples.py`:

```python
    return iit4_2023_fig7_substrate().inactivate({"E": 0})
```

replacing the `Substrate.from_factored(...)` body (keep the docstring; add "Built with :meth:`Substrate.inactivate`.").

- [ ] **Step 4: Run**

```bash
uv run pytest test/test_substrate.py test/test_examples.py -q > /tmp/t6.log 2>&1; echo $?; tail -1 /tmp/t6.log
uv run pytest test/integration/test_paper_reproduction.py -q -k fig7C > /tmp/t6b.log 2>&1; echo $?; tail -1 /tmp/t6b.log
uv run pytest pyphi/substrate.py -q 2>&1 | tail -1
```
Expected: all exit 0 (the last runs the doctest).

- [ ] **Step 5: Commit**

```bash
cat > changelog.d/substrate-inactivate.feature.md <<'EOF'
Added `Substrate.inactivate(fixed)`: returns a copy with the given units (by index or label) frozen in a state and conditioned into every other unit's dynamics — the lesion Albantakis et al. (2023, Fig 7C) call inactivation, distinct from an inactive unit and from a background condition. The Fig 7C example substrate is built with it.
EOF
uvx ruff format pyphi test && uvx ruff check pyphi test
git add pyphi/substrate.py pyphi/examples.py test/test_substrate.py changelog.d/substrate-inactivate.feature.md
git commit -m "Add Substrate.inactivate for the inactivated-unit lesion of Fig 7C"
```

---

### Task 7: 2026 paper — the monad (Fig 2)

**Files:**
- Modify: `pyphi/examples.py` (add `mayner_2026_monad_substrate(p)`)
- Test: `test/integration/test_paper_reproduction.py` (append a new section + `_iit4_2026` fixture)

**Interfaces:**
- Produces: `examples.mayner_2026_monad_substrate(p: float) -> Substrate` (one binary unit that keeps its state with probability ``p``); the `_iit4_2026` fixture used by Tasks 8–9.

- [ ] **Step 1: Add the fixture and test**

In `pyphi/examples.py` (under the IIT 4.0 (2023) block), register:

```python
@register_example
def mayner_2026_monad_substrate(p=0.744):
    """A single binary unit that keeps its state with probability ``p``
    (Mayner, Marshall & Tononi 2026, Fig 2A–B): an imperfect COPY for
    ``p > 0.5``. Its φₛ is ``min{p·log₂(2p), −log₂ p}`` (2026, Eq. 27),
    maximal at ``p ≈ 0.744`` where φₛ ≈ 0.427 (Fig 2C).
    """
    # Rows are the current state (0, 1); the column is P(unit is ON next).
    return Substrate(np.array([[1 - p], [p]]), node_labels=("M",))
```

In the test module, after the `_iit4_2023` fixture:

```python
@pytest.fixture
def _iit4_2026():
    with config.override(
        **presets.iit4_2026, validate_system_states=False, progress_bars=False
    ):
        yield


# --------------------------------------------------------------------------- #
# IIT 4.0 (2026) -- Mayner, Marshall & Tononi, Entropy 28(4):410, Fig 2
# --------------------------------------------------------------------------- #
# Fig 2C: the monad's phi_s = min{p log2(2p), -log2 p} (Eq. 27) peaks at
# p = 0.744 with phi_s = 0.427.


def _monad_sia(p):
    return System(examples.mayner_2026_monad_substrate(p), (1,), node_indices=(0,)).sia()


def test_mayner_2026_fig2_monad_peak(_iit4_2026):
    """Fig 2C / Eq. 27: phi_s(0.744) = 0.427, and the peak is interior."""
    peak = _monad_sia(0.744)
    assert float(peak.phi) == pytest.approx(0.427, abs=0.001)
    assert float(_monad_sia(0.70).phi) < float(peak.phi)
    assert float(_monad_sia(0.80).phi) < float(peak.phi)


@pytest.mark.parametrize("p", [0.6, 0.744, 0.9])
def test_mayner_2026_fig2_monad_terms(_iit4_2026, p):
    """Eqs. 24-25 at the ON state: specification p log2(2p) and
    differentiation -log2 p, identical on the cause and effect sides."""
    sia = _monad_sia(p)
    for direction in Direction.both():
        assert sia.intrinsic_specification[direction] == pytest.approx(
            p * np.log2(2 * p), abs=1e-9
        )
        assert float(sia.intrinsic_differentiation[direction]) == pytest.approx(
            -np.log2(p), abs=1e-9
        )
    assert float(sia.phi) == pytest.approx(min(p * np.log2(2 * p), -np.log2(p)), abs=1e-9)
```

- [ ] **Step 2: Run, then perturbation-verify**

```bash
uv run pytest test/integration/test_paper_reproduction.py -q -k mayner_2026_fig2 > /tmp/t7.log 2>&1; echo $?; tail -1 /tmp/t7.log
```
Expected: exit 0. Then temporarily change `0.427` to `0.437`, re-run, confirm FAIL, restore. If the peak test fails for real, print `float(_monad_sia(0.744).phi)` and the two terms; a monad's φₛ under the 2026 preset must equal ii(s) (the only partition severs the self-connection, so φ_c = φ_e = i_spec) — a mismatch is a finding to report, not to paper over.

- [ ] **Step 3: Commit**

```bash
uvx ruff format pyphi test && uvx ruff check pyphi test
git add pyphi/examples.py test/integration/test_paper_reproduction.py
git commit -m "Pin Mayner et al. (2026) Fig 2: the monad's phi_s peak at p = 0.744 (perturbation-verified)"
```

---

### Task 8: 2026 paper — the temperature sweep (Fig 3D–G)

**Files:**
- Modify: `pyphi/examples.py:1699-1722` (`iit4_2023_fig6d_substrate(k=4.0)`)
- Test: `test/integration/test_paper_reproduction.py` (append)

**Interfaces:**
- Consumes: `requirement_binding` finding (Task 5), `_iit4_2026` (Task 7).

- [ ] **Step 1: Give the Fig 6D substrate a `k` parameter**

Change the signature to `def iit4_2023_fig6d_substrate(k=4.0):` and the last line to `temperature=1 / k`; add to the docstring: "``k`` is the logistic slope (the paper's determinism parameter, K = 4 in Fig 6D); Mayner et al. (2026, Fig 3D–G) vary it on this same network."

- [ ] **Step 2: Write the tests**

```python
# --------------------------------------------------------------------------- #
# IIT 4.0 (2026), Fig 3D-G -- the Fig 6D lattice under a determinism sweep
# --------------------------------------------------------------------------- #
# Section 3.2: "the full 6-unit system being a complex for K >~ 0.775 and the
# system breaking down into two-unit complexes for K <~ 0.775"; "intrinsic
# differentiation only affects phi_s when K >~ 2.839 (Figure 3F)".
_FIG3_STATE = (1, 0, 0, 0, 0, 0)


@pytest.mark.slow
@pytest.mark.parametrize(("k", "expected_size"), [(0.85, 6), (0.70, 2)])
def test_mayner_2026_fig3_complex_size_crossover(_iit4_2026, k, expected_size):
    """Fig 3D-G / Section 3.2: the largest complex is the whole 6-unit
    system just above K = 0.775 and a 2-unit system just below it."""
    substrate = examples.iit4_2023_fig6d_substrate(k=k)
    largest = substrate.maximal_complex(_FIG3_STATE)
    assert len(largest.node_indices) == expected_size


@pytest.mark.parametrize(("k", "binds_on_differentiation"), [(3.0, True), (2.7, False)])
def test_mayner_2026_fig3_differentiation_binds_above_k_2_839(
    _iit4_2026, k, binds_on_differentiation
):
    """Fig 3F: intrinsic differentiation sets phi_s of the full system only
    for K >~ 2.839."""
    sia = System(
        examples.iit4_2023_fig6d_substrate(k=k), _FIG3_STATE, node_indices=tuple(range(6))
    ).sia()
    finding = next(
        (f for f in sia.explain().findings if f.kind == "requirement_binding"), None
    )
    binds = finding is not None and finding.value == "differentiation"
    assert binds == binds_on_differentiation
```

Check `Substrate.maximal_complex`'s signature (`pyphi/substrate.py:541`) and adapt the call (it may take `state` positionally or return a null object with `node_indices=()`).

- [ ] **Step 3: Run and perturbation-verify**

```bash
uv run pytest test/integration/test_paper_reproduction.py -q -k "fig3_differentiation" > /tmp/t8a.log 2>&1; echo $?; tail -1 /tmp/t8a.log
uv run pytest test/integration/test_paper_reproduction.py -m slow --slow -q -k "fig3_complex_size" > /tmp/t8b.log 2>&1; echo $?; tail -1 /tmp/t8b.log
```
Expected: both exit 0. Perturbation: swap the two `expected_size` values → FAIL; flip one boolean → FAIL; restore. If the complex-size test disagrees with the paper at these K values, bracket more finely (0.75 / 0.80) and, if the crossover sits elsewhere, pin the observed bracket and document the deviation in the docstring with the numbers; do not move the paper's quoted 0.775.

- [ ] **Step 4: Commit**

```bash
uvx ruff format pyphi test && uvx ruff check pyphi test
git add pyphi/examples.py test/integration/test_paper_reproduction.py
git commit -m "Pin Mayner et al. (2026) Fig 3: complex-size and differentiation crossovers on the Fig 6D lattice (perturbation-verified)"
```

---

### Task 9: 2026 paper — intrinsic units crossover (Fig 4C)

**Files:**
- Test: `test/integration/test_paper_reproduction.py` (append)

- [ ] **Step 1: Write the test**

```python
# --------------------------------------------------------------------------- #
# IIT 4.0 (2026), Fig 4 -- intrinsic units: macro monad vs micro pair
# --------------------------------------------------------------------------- #
# Section 3.3: two imperfect AND gates with parameter p (epsilon = 0.01) and
# the macro unit alpha = 1 iff both are ON. "For p < 0.096 we find
# phi_s({a,b}) > phi_s(alpha), while for p > 0.096, phi_s(alpha) > phi_s({a,b})."
# The macro TPM is the paper's Fig 4B mapping, materialized by the fixture.


def _fig4_micro_phi(p):
    substrate = Substrate(examples.differentiation_micro_tpm(p, 0.01))
    return float(System(substrate, (0, 0), node_indices=(0, 1)).sia().phi)


def _fig4_macro_phi(p):
    substrate = Substrate(examples.differentiation_macro_tpm(p, 0.01))
    return float(System(substrate, (0,), node_indices=(0,)).sia().phi)


@pytest.mark.parametrize(("p", "macro_wins"), [(0.10, True), (0.09, False)])
def test_mayner_2026_fig4_macro_micro_crossover(_iit4_2026, p, macro_wins):
    """Fig 4C: the macro monad has higher phi_s than the micro pair for
    p > 0.096 and lower for p < 0.096."""
    assert (_fig4_macro_phi(p) > _fig4_micro_phi(p)) == macro_wins


@pytest.mark.parametrize("p", [0.09, 0.10])
def test_mayner_2026_fig4_micro_pair_is_maximally_irreducible_within(_iit4_2026, p):
    """Section 3.3: {a, b} satisfies the maximally-irreducible-within
    criterion for all p in (0, 0.5) -- its phi_s exceeds each unit alone."""
    substrate = Substrate(examples.differentiation_micro_tpm(p, 0.01))
    pair = float(System(substrate, (0, 0), node_indices=(0, 1)).sia().phi)
    for unit in (0, 1):
        alone = float(System(substrate, (0, 0), node_indices=(unit,)).sia().phi)
        assert pair > alone
```

- [ ] **Step 2: Run and perturbation-verify**

```bash
uv run pytest test/integration/test_paper_reproduction.py -q -k fig4 > /tmp/t9.log 2>&1; echo $?; tail -1 /tmp/t9.log
```
Expected: exit 0. Perturb: swap the two `macro_wins` booleans → FAIL; restore. If the crossover is not between 0.09 and 0.10, print both φₛ at p ∈ {0.05, 0.08, 0.09, 0.10, 0.12, 0.2} and either the fixture's macro mapping (`differentiation_macro_tpm`) disagrees with the paper's Fig 4B mapping or the state (0, 0) is wrong — investigate before documenting a deviation.

- [ ] **Step 3: Commit**

```bash
uvx ruff format test && uvx ruff check test
git add test/integration/test_paper_reproduction.py
git commit -m "Pin Mayner et al. (2026) Fig 4: the macro/micro phi_s crossover at p = 0.096 (perturbation-verified)"
```

---

### Task 10: Marshall et al. (2023) Fig 1 — determinism and degeneracy

**Files:**
- Modify: `pyphi/examples.py` (add `marshall_2023_fig1_substrate(variant)`)
- Test: `test/integration/test_paper_reproduction.py` (append)

**Interfaces:**
- Consumes: `intrinsic_specification` (Task 3).
- Produces: `examples.marshall_2023_fig1_substrate(variant="deterministic"|"noisy"|"degenerate")`.

Background: Fig 1 uses the whole 4-unit universe as the system, so background conditioning does not enter. The paper gives the unit functions only as a figure table; any deterministic *bijective* TPM with four distinct unit functions reproduces panel B exactly (ii_c = ii_e = log₂ 16 = 4), and panels C and D follow from it as the paper describes: noise on D (0.6/0.4) gives ii = 0.6·log₂(0.6·16) = 1.958 on both sides; copying A's function into D makes the map 2-to-1, giving ii_e = log₂ 8 = 3 and ii_c = 0.5·3 = 1.5.

- [ ] **Step 1: Add the fixture**

```python
@register_example
def marshall_2023_fig1_substrate(variant="deterministic"):
    """The four-unit systems of Marshall et al. (2023), Fig 1.

    Four all-to-all units with distinct deterministic functions
    (``A' = B``, ``B' = C``, ``C' = D``, ``D' = A xor B``), a bijection on
    the state space, so the system in any state specifies a unique cause
    and effect with ``ii_c = ii_e = 4`` (Fig 1B). ``variant="noisy"``
    makes ``D`` go to its specified state with probability 0.6 (Fig 1C;
    ``ii_c = ii_e = 1.95``); ``variant="degenerate"`` gives ``D`` the same
    function as ``A`` (Fig 1D; ``ii_c = 1.5``, ``ii_e = 3.0``). The paper
    states the functions only as a figure table; this fixture reproduces
    the panel values, not the table.
    """
    if variant not in ("deterministic", "noisy", "degenerate"):
        raise ValueError(f"unknown variant {variant!r}")
    n = 4

    def d_target(a, b, c, d):
        if variant == "degenerate":
            return b  # A's function
        return a ^ b

    functions = (
        lambda a, b, c, d: b,
        lambda a, b, c, d: c,
        lambda a, b, c, d: d,
        d_target,
    )
    marginals = []
    for i, f in enumerate(functions):
        factor = np.zeros((2,) * n + (2,))
        for state in np.ndindex(*(2,) * n):
            target = f(*state)
            if i == 3 and variant == "noisy":
                factor[(*state, target)] = 0.6
                factor[(*state, 1 - target)] = 0.4
            else:
                factor[(*state, target)] = 1.0
        marginals.append(factor)
    return Substrate(
        marginals=marginals, state_space=((0, 1),) * n, node_labels=("A", "B", "C", "D")
    )
```

- [ ] **Step 2: Write the test**

```python
# --------------------------------------------------------------------------- #
# Marshall et al. (2023), System Integrated Information, Entropy 25:334, Fig 1
# --------------------------------------------------------------------------- #
# Fig 1B-D (state ABcD = (1, 1, 0, 1)): ii_c = ii_e = 4 (deterministic,
# non-degenerate); 1.95 / 1.95 with unit D noisy at 0.6; 1.5 / 3.0 with D's
# function identical to A's. The whole universe is the system, so the
# background-conditioning convention is immaterial here.
_MARSHALL_FIG1_STATE = (1, 1, 0, 1)
_MARSHALL_FIG1 = {
    "deterministic": (4.0, 4.0),
    "noisy": (1.95, 1.95),
    "degenerate": (1.5, 3.0),
}


@pytest.fixture
def _iit4_2023_current_state_background():
    """IIT 4.0 (2023) with background units conditioned on their current
    state, the convention of Marshall et al. (2023) Section 2.1."""
    with config.override(
        **{k: v for k, v in presets.iit4_2023.items() if k != "iit"},
        iit=replace(presets.iit4_2023["iit"], background_conditioning="CONDITION_CURRENT_STATE"),
        validate_system_states=False,
        progress_bars=False,
    ):
        yield


@pytest.mark.parametrize(("variant", "expected"), list(_MARSHALL_FIG1.items()))
def test_marshall_2023_fig1_information(_iit4_2023_current_state_background, variant, expected):
    """Fig 1B/C/D: cause and effect intrinsic information of the four-unit
    system (the paper's ii_c / ii_e; intrinsic specification in 2026 terms)."""
    ii_c, ii_e = expected
    sia = System(
        examples.marshall_2023_fig1_substrate(variant),
        _MARSHALL_FIG1_STATE,
        node_indices=(0, 1, 2, 3),
    ).sia()
    spec = sia.intrinsic_specification
    assert spec[Direction.CAUSE] == pytest.approx(ii_c, abs=0.01)
    assert spec[Direction.EFFECT] == pytest.approx(ii_e, abs=0.01)
```

- [ ] **Step 3: Run and perturbation-verify**

```bash
uv run pytest test/integration/test_paper_reproduction.py -q -k marshall_2023_fig1 > /tmp/t10.log 2>&1; echo $?; tail -1 /tmp/t10.log
```
Expected: exit 0. Perturb `(4.0, 4.0)` → `(3.0, 4.0)`: FAIL; restore. (1.958 vs the paper's 1.95 is within the 0.01 tolerance.)

- [ ] **Step 4: Commit**

```bash
uvx ruff format pyphi test && uvx ruff check pyphi test
git add pyphi/examples.py test/integration/test_paper_reproduction.py
git commit -m "Pin Marshall et al. (2023) Fig 1: determinism and degeneracy set intrinsic information (perturbation-verified)"
```

---

### Task 11: Marshall et al. (2023) Fig 2 — fault lines

**Files:**
- Modify: `pyphi/examples.py` (add `marshall_2023_fig2_substrate(panel)`)
- Test: `test/integration/test_paper_reproduction.py` (append)

- [ ] **Step 1: Add the fixture**

Weights from §3.2 (rows are sources, columns targets; `Σ_j w_ji = 1` for every target `i`, Eq. 2). k = 3 → `temperature = 1/3`; l = 1 is the logistic's own constant. Units A, B, C, D = indices 0–3.

```python
@register_example
def marshall_2023_fig2_substrate(panel="A"):
    """The four-unit sigmoid systems of Marshall et al. (2023), Fig 2.

    Units follow Eq. 2 (logistic of the weighted ±1 inputs, ``k = 3``,
    ``l = 1``) in the all-OFF state. ``panel="A"``: a symmetric cycle of
    strong (0.4) forward and weaker (0.3) reverse connections, self 0.2,
    non-neighbour 0.1 — no fault line. ``"B"``: {A, B, C} strongly
    interconnected (0.3), unit D weakly attached (0.2). ``"C"``: two
    strongly coupled pairs {A, B}, {C, D} (0.4 within, 0.15 between,
    self 0.3). Every column sums to 1.
    """
    if panel == "A":
        # forward cycle A->B->C->D->A at 0.4, reverse at 0.3, self 0.2,
        # opposite unit 0.1
        w = np.array([
            [0.2, 0.4, 0.1, 0.3],
            [0.3, 0.2, 0.4, 0.1],
            [0.1, 0.3, 0.2, 0.4],
            [0.4, 0.1, 0.3, 0.2],
        ])
    elif panel == "B":
        w = np.array([
            [0.2, 0.3, 0.3, 0.2],
            [0.3, 0.2, 0.3, 0.2],
            [0.3, 0.3, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.4],
        ])
    elif panel == "C":
        w = np.array([
            [0.3, 0.4, 0.15, 0.15],
            [0.4, 0.3, 0.15, 0.15],
            [0.15, 0.15, 0.3, 0.4],
            [0.15, 0.15, 0.4, 0.3],
        ])
    else:
        raise ValueError(f"unknown panel {panel!r}")
    assert np.allclose(w.sum(axis=0), 1.0)
    return build_substrate([ising.probability] * 4, w, temperature=1 / 3)
```

Verify the convention `weights[i, j]` = weight of the edge from unit `i` to unit `j` (the Fig 6 comment at `examples.py:~1550` says so), so `w.sum(axis=0)` is the per-target sum.

- [ ] **Step 2: Write the test**

```python
# --------------------------------------------------------------------------- #
# Marshall et al. (2023), Fig 2 -- fault lines reduce integration
# --------------------------------------------------------------------------- #
# Section 3.2 (k = 3, l = 1, all units OFF): phi_s = 0.3393 (A, two equivalent
# MIPs {AB}|{CD} and {AD}|{BC}), 0.0628 (B, MIP {ABC}<- | {D}->), 0.1477 (C,
# MIP {AB}<->{CD}); the integrated cause and effect information are 48.1 %,
# 10.0 %, 21.2 % of the intrinsic information.
_MARSHALL_FIG2 = {
    "A": (0.3393, 0.481),
    "B": (0.0628, 0.100),
    "C": (0.1477, 0.212),
}


@pytest.mark.parametrize(("panel", "expected"), list(_MARSHALL_FIG2.items()))
def test_marshall_2023_fig2_fault_lines(_iit4_2023_current_state_background, panel, expected):
    """Fig 2A-C: phi_s and the integrated fraction phi_s / ii under fault lines."""
    phi_s, fraction = expected
    sia = System(
        examples.marshall_2023_fig2_substrate(panel), (0, 0, 0, 0), node_indices=(0, 1, 2, 3)
    ).sia()
    assert float(sia.phi) == pytest.approx(phi_s, abs=0.0005)
    assert sia.integrated_fraction == pytest.approx(fraction, abs=0.001)


def test_marshall_2023_fig2a_two_equivalent_mips(_iit4_2023_current_state_background):
    """Fig 2A: the symmetric system has two equivalent minimum partitions."""
    sia = System(
        examples.marshall_2023_fig2_substrate("A"), (0, 0, 0, 0), node_indices=(0, 1, 2, 3)
    ).sia()
    assert len(sia.ties) == 2
```

`integrated_fraction` is φₛ / ii(s) where ii(s) = min over directions of min(i_spec, i_diff); the paper's percentage is φ_c / ii_c (= φ_e / ii_e here, "the same because the current state equals the selected cause and effect states"). If the two differ because the differentiation term is the smaller one for this system, compute the paper's ratio directly instead: `float(sia.cause.phi) / sia.intrinsic_specification[Direction.CAUSE]`, and say so in the docstring.

- [ ] **Step 3: Run and perturbation-verify**

```bash
uv run pytest test/integration/test_paper_reproduction.py -q -k marshall_2023_fig2 > /tmp/t11.log 2>&1; echo $?; tail -1 /tmp/t11.log
```
Expected: exit 0. Perturb `0.3393` → `0.3493`: FAIL; restore. If a panel misses: (1) confirm the weight orientation by checking the Fig 6 fixtures' convention; (2) re-run under plain `_iit4_2023` (causal marginalization) — for the whole-universe system the two conventions coincide, so a mismatch is not background-related; (3) print φ_c, φ_e, ii_c, ii_e and compare to the paper's 48.1 % logic. Document any residual deviation per the spec.

- [ ] **Step 4: Commit**

```bash
uvx ruff format pyphi test && uvx ruff check pyphi test
git add pyphi/examples.py test/integration/test_paper_reproduction.py
git commit -m "Pin Marshall et al. (2023) Fig 2: fault lines and integrated fractions (perturbation-verified)"
```

---

### Task 12: Marshall et al. (2023) Fig 3 — the 8-unit universe condenses

**Files:**
- Modify: `pyphi/examples.py` (add `marshall_2023_fig3_substrate()`)
- Test: `test/integration/test_paper_reproduction.py` (append, slow)

- [ ] **Step 1: Add the fixture**

§3.3: units A–E (indices 0–4) each get self 0.025, strong 0.45 from the previous unit in the loop A→B→C→D→E→A, moderate 0.225 from one unit, weak 0.1 from the other two cluster units, and 0.033 from each of F, G, H. F (5): self 0.769, 0.033 from each of G, H and each of A–E. G, H (6, 7): 0.769 from each other, self 0.033, 0.033 from F and from each of A–E. k = 2 for A–F, 0.2 for G–H (per-unit temperature via `functools.partial`). The paper does not say which cluster unit supplies the moderate 0.225; the symmetric candidates are "two steps back" (A←D... i.e. target i from i−2) and "one step forward" (from i+1). Implement `moderate="back"` first, `"forward"` as the fallback.

```python
@register_example
def marshall_2023_fig3_substrate(moderate="back"):
    """The eight-unit universe of Marshall et al. (2023), Fig 3.

    A five-unit cluster {A, B, C, D, E} whose strong (0.45) connections
    form a loop, with a moderate (0.225) input from the unit two steps
    back in the loop, weak (0.1) inputs from the other two cluster units,
    a weak self-connection (0.025), and weak (0.033) inputs from the three
    units outside; unit F with a strong self-connection (0.769); units G
    and H strongly (0.769) coupled to each other. Sigmoid units per Eq. 2
    with ``k = 2`` (A–F) and ``k = 0.2`` (G, H), ``l = 1``. Every column
    sums to 1. Condenses into the complexes {F}, {A, B, C, D, E}, {G, H}
    (Fig 3C). The paper leaves the source of the moderate input implicit;
    ``moderate="forward"`` takes it from the next unit in the loop instead.
    """
    import functools

    n = 8
    w = np.zeros((n, n))
    cluster = range(5)
    for i in cluster:
        w[i, i] = 0.025
        w[(i - 1) % 5, i] = 0.45
        mod = (i - 2) % 5 if moderate == "back" else (i + 1) % 5
        w[mod, i] = 0.225
        for j in cluster:
            if j not in (i, (i - 1) % 5, mod):
                w[j, i] = 0.1
        for j in (5, 6, 7):
            w[j, i] = 0.033
    # F
    w[5, 5] = 0.769
    for j in (6, 7):
        w[j, 5] = 0.033
    for j in cluster:
        w[j, 5] = 0.033
    # G, H
    for i, other in ((6, 7), (7, 6)):
        w[other, i] = 0.769
        w[i, i] = 0.033
        w[5, i] = 0.033
        for j in cluster:
            w[j, i] = 0.033
    assert np.allclose(w.sum(axis=0), 1.0, atol=0.002)
    units = [functools.partial(ising.probability, temperature=1 / 2)] * 6 + [
        functools.partial(ising.probability, temperature=1 / 0.2)
    ] * 2
    return build_substrate(units, w, node_labels=NodeLabels("ABCDEFGH", range(n)))
```

Check that `build_tpm` calls each unit function as `f(element, weights, state, **kwargs)` so a `partial` with `temperature` bound works (it does per the `build_substrate` docstring); import `NodeLabels` if not already imported in `examples.py`.

- [ ] **Step 2: Write the tests**

```python
# --------------------------------------------------------------------------- #
# Marshall et al. (2023), Fig 3 -- exclusion condenses a universe
# --------------------------------------------------------------------------- #
# Section 3.3 / Fig 3C (all units OFF): three complexes, {F} phi_s = 0.49,
# {A,B,C,D,E} 0.12, {G,H} 0.06. Fig 3E-F: along {A} c {A,B} c ... c
# {A,...,E} intrinsic information rises with each unit while phi_s stays
# low until all five units remove the last fault line.
_MARSHALL_FIG3_STATE = (0,) * 8


@pytest.mark.slow
def test_marshall_2023_fig3_condensation(_iit4_2023_current_state_background):
    substrate = examples.marshall_2023_fig3_substrate()
    found = {
        tuple(c.node_indices): round(float(c.phi), 2)
        for c in substrate.complexes(_MARSHALL_FIG3_STATE)
    }
    assert found == {(5,): 0.49, (0, 1, 2, 3, 4): 0.12, (6, 7): 0.06}


@pytest.mark.slow
def test_marshall_2023_fig3_nested_sequence(_iit4_2023_current_state_background):
    """Fig 3E-F: ii_c and ii_e increase along the nested sequence; phi_s of the
    five-unit system exceeds every proper prefix."""
    substrate = examples.marshall_2023_fig3_substrate()
    prefixes = [tuple(range(k)) for k in range(1, 6)]
    sias = [System(substrate, _MARSHALL_FIG3_STATE, node_indices=p).sia() for p in prefixes]
    for direction in Direction.both():
        values = [s.intrinsic_specification[direction] for s in sias]
        assert values == sorted(values)
    phis = [float(s.phi) for s in sias]
    assert phis[-1] > max(phis[:-1])
```

- [ ] **Step 3: Run (slow lane) and perturbation-verify**

```bash
uv run pytest test/integration/test_paper_reproduction.py -m slow --slow -q -k marshall_2023_fig3 > /tmp/t12.log 2>&1; echo $?; tail -1 /tmp/t12.log
```
Expected: exit 0 (minutes). Perturb `0.49` → `0.59`: FAIL; restore. If the condensation test misses: try `moderate="forward"`; then try the plain `_iit4_2023` fixture (causal marginalization — the paper predates the 4.0 background convention, so this is the D3 fallback); if the complexes match but φₛ values differ in the second decimal, pin to two decimals with `abs=0.01` and document; if the partition of the universe itself differs, pin the observed condensation with a docstring stating the paper's and note the deviation — do not delete the test.

- [ ] **Step 4: Commit**

```bash
uvx ruff format pyphi test && uvx ruff check pyphi test
git add pyphi/examples.py test/integration/test_paper_reproduction.py
git commit -m "Pin Marshall et al. (2023) Fig 3: the eight-unit universe condenses into three complexes (perturbation-verified)"
```

---

### Task 13: S1 terminal tie branch, end to end

**Files:**
- Test: `test/formalism/test_complexes.py` (append)

- [ ] **Step 1: Write the test**

Two disjoint XOR-ish 2-unit blocks sharing one unit in a symmetric 3-unit substrate give two overlapping candidates {A, B} and {B, C} that are mirror images (identical φₛ, identical Φ). Build it with a mirror-symmetric TPM so the tie is structural rather than numerical:

```python
def _mirror_symmetric_substrate():
    """A 3-unit substrate invariant under swapping A and C: candidates {A, B}
    and {B, C} are mirror images, so they tie in phi_s and in Phi."""
    from pyphi.substrate import Substrate

    def f_a(a, b, c):
        return 0.9 if (a ^ b) else 0.1

    def f_c(a, b, c):
        return 0.9 if (c ^ b) else 0.1

    def f_b(a, b, c):
        return 0.8 if (a + c) >= 1 else 0.2

    marginals = []
    for f in (f_a, f_b, f_c):
        factor = np.zeros((2, 2, 2, 2))
        for state in np.ndindex(2, 2, 2):
            p_on = f(*state)
            factor[(*state, 1)] = p_on
            factor[(*state, 0)] = 1 - p_on
        marginals.append(factor)
    return Substrate(marginals=marginals, state_space=((0, 1),) * 3, node_labels=("A", "B", "C"))


class TestS1TerminalTieBranch:
    """Albantakis et al. (2023) S1 Text: overlapping systems tied in phi_s and
    then in Phi do not comply with exclusion; neither is a complex and the
    next best unique system is chosen."""

    def test_mirror_tied_pair_fails_exclusion_end_to_end(self):
        from pyphi.conf import config, presets
        from pyphi.substrate import Substrate

        substrate = _mirror_symmetric_substrate()
        state = (0, 0, 0)
        with config.override(**presets.iit4_2023, validate_system_states=False):
            ab = System(substrate, state, node_indices=(0, 1)).sia()
            bc = System(substrate, state, node_indices=(1, 2)).sia()
            # The construction really does tie (power check for the test itself).
            assert float(ab.phi) == pytest.approx(float(bc.phi), abs=1e-12)
            assert float(ab.phi) > 0
            complexes = substrate.complexes(state)
        footprints = {tuple(c.node_indices) for c in complexes}
        assert (0, 1) not in footprints
        assert (1, 2) not in footprints
        # If the full system or a monad is irreducible it is accepted instead.
        if complexes:
            assert all(len(c.node_indices) != 2 or set(c.node_indices) == {0, 2} for c in complexes)
```

Then strengthen the assertion once the actual outcome is known: if `{A, B, C}` has higher φₛ than the pair, the pair never reaches the cascade — in that case lower the full system's φₛ by weakening `f_b` (e.g. `0.6/0.4`) until the pair tier is the top tier, and assert the specific accepted complexes. The test must exercise the failed-clique path: assert it by checking that no accepted complex overlaps the tied pair and that at least one candidate with lower φₛ than the pair *is* accepted (the "next best unique" clause). Add `import numpy as np`, `import pytest`, `from pyphi.system import System` if the module lacks them.

- [ ] **Step 2: Run, iterate on the fixture, perturbation-verify**

```bash
uv run pytest test/formalism/test_complexes.py -q -k TerminalTie > /tmp/t13.log 2>&1; echo $?; tail -1 /tmp/t13.log
```
Expected: exit 0 after fixture tuning. Perturbation: break the symmetry (`f_c` uses `0.85`) → the tie assertion FAILS; restore.

- [ ] **Step 3: Commit**

```bash
uvx ruff format test && uvx ruff check test
git add test/formalism/test_complexes.py
git commit -m "Test the S1 terminal tie branch end to end through Substrate.complexes (perturbation-verified)"
```

---

### Task 14: AC 2019 pins — Figs 7, 8A, 9, 12

**Files:**
- Create: `test/integration/test_paper_reproduction_ac.py`

**Interfaces:**
- Produces: the module-level helper `_gate_substrate(n_inputs, outputs)` reused by Task 15.

- [ ] **Step 1: Create the module with the helper and the Fig 7/8A/9/12 tests**

```python
"""Actual-causation paper reproductions -- Albantakis, Marshall, Hoel & Tononi
(2019), "What caused what?", Entropy 21(5):459.

Paper-sourced pins of the causal accounts in Figs 7-16 and Appendix A. Values
are quoted to the paper's precision (three decimals, or two where the figure
gives two). Input units are modeled as self-copying (their next state repeats
their current state), the convention the Fig 8B reproduction in
``test_paper_reproduction.py`` uses; input dynamics do not enter links whose
effect set is the output unit.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from pyphi import actual
from pyphi import examples
from pyphi.conf import config
from pyphi.conf import presets
from pyphi.direction import Direction
from pyphi.substrate import Substrate


@pytest.fixture
def _iit3():
    with config.override(
        **presets.iit3, validate_system_states=False, progress_bars=False
    ):
        yield


def _gate_substrate(n_inputs, outputs, labels=None, alphabet=None):
    """A substrate of ``n_inputs`` self-copying input units followed by output
    units. ``outputs`` maps an output name to ``f(*input_states) -> state``
    (deterministic) or ``-> dict[state, probability]`` (stochastic). Output
    units read only the inputs. ``alphabet`` gives per-unit alphabet sizes
    (default binary)."""
    names = list(outputs)
    n = n_inputs + len(names)
    sizes = tuple(alphabet) if alphabet is not None else (2,) * n
    marginals = []
    for i in range(n):
        factor = np.zeros(sizes + (sizes[i],))
        for state in np.ndindex(*sizes):
            if i < n_inputs:
                factor[(*state, state[i])] = 1.0
            else:
                result = outputs[names[i - n_inputs]](*state[:n_inputs])
                if isinstance(result, dict):
                    for target, p in result.items():
                        factor[(*state, target)] = p
                else:
                    factor[(*state, result)] = 1.0
        marginals.append(factor)
    if labels is None:
        labels = tuple("ABCDEFGHIJ"[:n_inputs]) + tuple(names)
    return Substrate(
        marginals=marginals,
        state_space=tuple(tuple(range(k)) for k in sizes),
        node_labels=labels,
    )


def _account(substrate, before, after, cause_indices, effect_indices):
    """{(direction, mechanism): (purview, alpha rounded to 3 dp)}."""
    transition = actual.Transition(substrate, before, after, cause_indices, effect_indices)
    links = actual.account(transition)
    return {
        (link.direction, tuple(link.mechanism)): (tuple(link.purview), round(float(link.alpha), 3))
        for link in links
    }, links


# --------------------------------------------------------------------------- #
# Fig 7 -- four gates, one transition {AB = 11} -> {out = 1}
# --------------------------------------------------------------------------- #


def test_ac_fig7a_disjunction(_iit3):
    """Fig 7A (OR): {A} -> {C} and {B} -> {C} at 0.415 each; the actual cause
    of {C = 1} is undetermined between {A} and {B} at 0.415 (symmetric
    over-determination); {AB} is reducible on the effect side."""
    s = _gate_substrate(2, {"C": lambda a, b: a | b})
    account, links = _account(s, (1, 1, 0), (1, 1, 1), (0, 1), (2,))
    assert account[(Direction.EFFECT, (0,))] == ((2,), 0.415)
    assert account[(Direction.EFFECT, (1,))] == ((2,), 0.415)
    assert (Direction.EFFECT, (0, 1)) not in account
    cause = next(l for l in links if l.direction == Direction.CAUSE and tuple(l.mechanism) == (2,))
    assert round(float(cause.alpha), 3) == 0.415
    assert cause.purview_ties is not None and len(cause.purview_ties) == 2


def test_ac_fig7b_conjunction(_iit3):
    """Fig 7B (AND): {A}, {B}, {AB} each -> {D} at 1.0; the one actual cause
    of {D = 1} is {AB = 11} at 2.0."""
    s = _gate_substrate(2, {"D": lambda a, b: a & b})
    account, _ = _account(s, (1, 1, 0), (1, 1, 1), (0, 1), (2,))
    for mech in ((0,), (1,), (0, 1)):
        assert account[(Direction.EFFECT, mech)] == ((2,), 1.0)
    assert account[(Direction.CAUSE, (2,))] == ((0, 1), 2.0)


def test_ac_fig7c_biconditional(_iit3):
    """Fig 7C (XNOR): only the second-order occurrence links, {AB} <-> {E}
    at 1.0 each way; the parts have zero effect information."""
    s = _gate_substrate(2, {"E": lambda a, b: int(a == b)})
    account, _ = _account(s, (1, 1, 0), (1, 1, 1), (0, 1), (2,))
    assert account[(Direction.EFFECT, (0, 1))] == ((2,), 1.0)
    assert account[(Direction.CAUSE, (2,))] == ((0, 1), 1.0)
    assert (Direction.EFFECT, (0,)) not in account
    assert (Direction.EFFECT, (1,)) not in account


def test_ac_fig7d_prevention(_iit3):
    """Fig 7D (prevention: F = 0 only for AB = 10): {B} <-> {F} at 0.415;
    {A} has no effect and is not a cause."""
    s = _gate_substrate(2, {"F": lambda a, b: 0 if (a, b) == (1, 0) else 1})
    account, _ = _account(s, (1, 1, 0), (1, 1, 1), (0, 1), (2,))
    assert account[(Direction.EFFECT, (1,))] == ((2,), 0.415)
    assert account[(Direction.CAUSE, (2,))] == ((1,), 0.415)
    assert (Direction.EFFECT, (0,)) not in account


# --------------------------------------------------------------------------- #
# Fig 8A -- majority gate, ABCD = 1110 -> M = 1
# --------------------------------------------------------------------------- #


def test_ac_fig8a_majority(_iit3):
    """Fig 8A: singleton effects 0.678, pairs 0.585, {ABC} 0.415; the actual
    cause {ABC = 111} <- {M = 1} at 1.678; D = 0 links nowhere."""
    s = _gate_substrate(4, {"M": lambda a, b, c, d: int(a + b + c + d >= 3)})
    account, _ = _account(s, (1, 1, 1, 0, 0), (1, 1, 1, 0, 1), (0, 1, 2, 3), (4,))
    for mech in ((0,), (1,), (2,)):
        assert account[(Direction.EFFECT, mech)] == ((4,), 0.678)
    for mech in ((0, 1), (0, 2), (1, 2)):
        assert account[(Direction.EFFECT, mech)] == ((4,), 0.585)
    assert account[(Direction.EFFECT, (0, 1, 2))] == ((4,), 0.415)
    assert account[(Direction.CAUSE, (4,))] == ((0, 1, 2), 1.678)
    assert not any(3 in mech for (d, mech) in account if d == Direction.EFFECT)


# --------------------------------------------------------------------------- #
# Fig 9 -- disjunction of conjunctions (A and B) or C, ABC = 101 -> D = 1
# --------------------------------------------------------------------------- #


def test_ac_fig9a_disjunction_of_conjunctions(_iit3):
    """Fig 9A: {A} -> {D} at 0.263, {C} -> {D} at 0.678; the actual cause of
    {D = 1} is {C} at 0.678."""
    s = examples.disjunction_conjunction_substrate()
    account, _ = _account(s, (1, 0, 1, 0), (1, 0, 1, 1), (0, 1, 2), (3,))
    assert account[(Direction.EFFECT, (0,))] == ((3,), 0.263)
    assert account[(Direction.EFFECT, (2,))] == ((3,), 0.678)
    assert account[(Direction.CAUSE, (3,))] == ((2,), 0.678)


def test_ac_fig9b_background_b(_iit3):
    """Fig 9B: with B = 0 a fixed background condition, {C} <-> {D} at 1.0 and
    {A} has no actual effect."""
    s = examples.disjunction_conjunction_substrate()
    account, _ = _account(s, (1, 0, 1, 0), (1, 0, 1, 1), (0, 2), (3,))
    assert account[(Direction.EFFECT, (2,))] == ((3,), 1.0)
    assert account[(Direction.CAUSE, (3,))] == ((2,), 1.0)
    assert (Direction.EFFECT, (0,)) not in account


# --------------------------------------------------------------------------- #
# Fig 12 -- a noisy COPY
# --------------------------------------------------------------------------- #


def test_ac_fig12_noisy_copy(_iit3):
    """Fig 12: N copies A with probability 0.9. {A = 1} -> {N = 1} and
    {A = 1} <- {N = 1} at 0.848 (12A); {A = 1} -> {N = 0} has no causal links
    (12B)."""
    s = _gate_substrate(1, {"N": lambda a: {1: 0.9, 0: 0.1} if a else {1: 0.1, 0: 0.9}})
    account, _ = _account(s, (1, 0), (1, 1), (0,), (1,))
    assert account[(Direction.EFFECT, (0,))] == ((1,), 0.848)
    assert account[(Direction.CAUSE, (1,))] == ((0,), 0.848)
    account_b, _ = _account(s, (1, 0), (1, 0), (0,), (1,))
    assert account_b == {}
```

Note the `disjunction_conjunction_substrate` example sets every input's next state to 0 (its TPM rows), so the "after" state of the inputs in the transition must be what the substrate produces — check its TPM (`examples.py:1235`): inputs go to 0 next. Use after = `(0, 0, 0, 1)` if `(1, 0, 1, 1)` violates realization (`p(after | before) > 0`); the paper's Fig 9 says nothing about the inputs' next state.

- [ ] **Step 2: Run and perturbation-verify**

```bash
uv run pytest test/integration/test_paper_reproduction_ac.py -q > /tmp/t14.log 2>&1; echo $?; tail -1 /tmp/t14.log
```
Expected: exit 0. Perturb one value per figure (e.g. Fig 8A `0.678` → `0.688`): FAIL; restore. Where an assertion on the *structure* of the account fails (a link present/absent), print the whole account dict and compare against the figure before changing anything; a genuine mismatch is documented as a deviation, never silenced.

- [ ] **Step 3: Commit**

```bash
uvx ruff format test && uvx ruff check test
git add test/integration/test_paper_reproduction_ac.py
git commit -m "Pin Albantakis et al. (2019) Figs 7, 8A, 9, 12 causal accounts (perturbation-verified)"
```

---

### Task 15: AC 2019 pins — Figs 10, 11, 13, 15, 16

**Files:**
- Modify: `test/integration/test_paper_reproduction_ac.py` (append)
- Modify: `pyphi/examples.py` (add `ac_2019_three_candidate_election_substrate()`)

- [ ] **Step 1: Add the election fixture**

```python
@register_example
def ac_2019_three_candidate_election_substrate():
    """The three-candidate, seven-voter election of Albantakis et al.
    (2019), Fig 11: voters ``A``–``G`` each in state 0, 1, or 2 (candidates
    "1", "2", "3"), and ``W`` in state 1, 2, or 3 for the candidate with a
    strict majority of votes, or 0 for a tie. Voters repeat their state.
    """
    n_voters = 7
    sizes = (3,) * n_voters + (4,)
    marginals = []
    for i in range(n_voters):
        factor = np.zeros(sizes + (3,))
        for state in np.ndindex(*sizes):
            factor[(*state, state[i])] = 1.0
        marginals.append(factor)
    factor = np.zeros(sizes + (4,))
    for state in np.ndindex(*sizes):
        counts = [state[:n_voters].count(c) for c in range(3)]
        best = max(counts)
        winner = counts.index(best) + 1 if counts.count(best) == 1 else 0
        factor[(*state, winner)] = 1.0
    marginals.append(factor)
    return Substrate(
        marginals=marginals,
        state_space=tuple(tuple(range(k)) for k in sizes),
        node_labels=tuple("ABCDEFG") + ("W",),
    )
```

- [ ] **Step 2: Append the tests**

```python
# --------------------------------------------------------------------------- #
# Fig 10 -- complicated voting, ABCDE = 11000 -> F = 1
# --------------------------------------------------------------------------- #


def _complicated_vote(a, b, c, d, e):
    if a == b:
        return a
    if b == c == d == e:
        return a
    return int(a + b + c + d + e >= 3)


def test_ac_fig10_complicated_voting(_iit3):
    """Fig 10: effects {A} 0.70, {B} 0.46, {AB} 0.30, {ACDE} 0.30; the actual
    cause of {F = 1} is undetermined between {AB = 11} and {ACDE = 1000},
    both at 1.0."""
    s = _gate_substrate(5, {"F": _complicated_vote})
    account, links = _account(s, (1, 1, 0, 0, 0, 0), (1, 1, 0, 0, 0, 1), (0, 1, 2, 3, 4), (5,))
    two = {k: (p, round(a, 2)) for k, (p, a) in account.items()}
    assert two[(Direction.EFFECT, (0,))] == ((5,), 0.70)
    assert two[(Direction.EFFECT, (1,))] == ((5,), 0.46)
    assert two[(Direction.EFFECT, (0, 1))] == ((5,), 0.30)
    assert two[(Direction.EFFECT, (0, 2, 3, 4))] == ((5,), 0.30)
    cause = next(l for l in links if l.direction == Direction.CAUSE)
    assert round(float(cause.alpha), 3) == 1.0
    tied = {tuple(ria.purview) for ria in (cause.purview_ties or ())}
    assert tied == {(0, 1), (0, 2, 3, 4)}


# --------------------------------------------------------------------------- #
# Fig 11 -- three candidates, seven voters (multi-valued)
# --------------------------------------------------------------------------- #


def test_ac_fig11_three_candidate_election(_iit3):
    """Fig 11: five votes for "1" (state 0) and two for "2" (state 1) elect
    "1" (W = 1). Effects of the "1" voters by occurrence order: 0.718 (one),
    0.581 (two), 0.404 (three), 0.190 (four); the actual cause of {W = 1} is
    an undetermined set of four "1" voters at 1.893; the "2" votes {F}, {G}
    have alpha = 0 (no links). The suite's first multi-valued AC pin."""
    s = examples.ac_2019_three_candidate_election_substrate()
    before = (0, 0, 0, 0, 0, 1, 1, 0)
    after = (0, 0, 0, 0, 0, 1, 1, 1)
    account, links = _account(s, before, after, tuple(range(7)), (7,))
    assert account[(Direction.EFFECT, (0,))] == ((7,), 0.718)
    assert account[(Direction.EFFECT, (0, 1))] == ((7,), 0.581)
    assert account[(Direction.EFFECT, (0, 1, 2))] == ((7,), 0.404)
    assert account[(Direction.EFFECT, (0, 1, 2, 3))] == ((7,), 0.190)
    assert not any(5 in m or 6 in m for (d, m) in account if d == Direction.EFFECT)
    cause = next(l for l in links if l.direction == Direction.CAUSE)
    assert round(float(cause.alpha), 3) == 1.893
    tied = {tuple(ria.purview) for ria in (cause.purview_ties or ())}
    assert tied == set(itertools.combinations(range(5), 4))


# --------------------------------------------------------------------------- #
# Fig 13 -- dot / segment / line classifier, ABC = 001 -> DSL = 100
# --------------------------------------------------------------------------- #


def test_ac_fig13_classifier(_iit3):
    """Fig 13: effects {A=0} -> {DL=10} 0.608, {B=0} -> {DSL=100} 1.02,
    {ABC=001} -> {D=1} 1.0; causes {ABC=001} <- {D=1} 1.415, {B=0} <- {S=0}
    0.415, {{A=0},{B=0}} <- {L=0} 0.193, {B=0} <- {DS=10} 0.263,
    {{A=0},{B=0}} <- {DL=10} 0.126, {B=0} <- {SL=00} 0.126,
    {B=0} <- {DSL=100} 0.074."""
    s = _gate_substrate(
        3,
        {
            "D": lambda a, b, c: int(a + b + c == 1),
            "S": lambda a, b, c: int((a, b, c) in ((1, 1, 0), (0, 1, 1))),
            "L": lambda a, b, c: int((a, b, c) == (1, 1, 1)),
        },
    )
    account, links = _account(s, (0, 0, 1, 0, 0, 0), (0, 0, 1, 1, 0, 0), (0, 1, 2), (3, 4, 5))
    assert account[(Direction.EFFECT, (0,))] == ((3, 5), 0.608)
    assert account[(Direction.EFFECT, (1,))] == ((3, 4, 5), 1.02)
    assert account[(Direction.EFFECT, (0, 1, 2))] == ((3,), 1.0)
    assert account[(Direction.CAUSE, (3,))] == ((0, 1, 2), 1.415)
    assert account[(Direction.CAUSE, (4,))] == ((1,), 0.415)
    assert account[(Direction.CAUSE, (3, 4))] == ((1,), 0.263)
    assert account[(Direction.CAUSE, (4, 5))] == ((1,), 0.126)
    assert account[(Direction.CAUSE, (3, 4, 5))] == ((1,), 0.074)
    # Undetermined causes: {A=0} or {B=0} for {L=0} (0.193) and {DL=10} (0.126).
    by_mech = {tuple(l.mechanism): l for l in links if l.direction == Direction.CAUSE}
    for mech, alpha in (((5,), 0.193), ((3, 5), 0.126)):
        link = by_mech[mech]
        assert round(float(link.alpha), 3) == alpha
        assert {tuple(r.purview) for r in (link.purview_ties or ())} == {(0,), (1,)}


# --------------------------------------------------------------------------- #
# Fig 15 -- double bi-conditional, ABC = 111 -> DE = 11
# --------------------------------------------------------------------------- #


def test_ac_fig15_double_biconditional(_iit3):
    """Fig 15: {AB} -> {D}, {BC} -> {E}, {ABC} -> {DE} and the three reverse
    causes all at 1.0 bits; no first-order links."""
    s = _gate_substrate(
        3, {"D": lambda a, b, c: int(a == b), "E": lambda a, b, c: int(b == c)}
    )
    account, _ = _account(s, (1, 1, 1, 0, 0), (1, 1, 1, 1, 1), (0, 1, 2), (3, 4))
    assert account[(Direction.EFFECT, (0, 1))] == ((3,), 1.0)
    assert account[(Direction.EFFECT, (1, 2))] == ((4,), 1.0)
    assert account[(Direction.EFFECT, (0, 1, 2))] == ((3, 4), 1.0)
    assert account[(Direction.CAUSE, (3,))] == ((0, 1), 1.0)
    assert account[(Direction.CAUSE, (4,))] == ((1, 2), 1.0)
    assert account[(Direction.CAUSE, (3, 4))] == ((0, 1, 2), 1.0)
    assert not any(len(m) == 1 for (d, m) in account if d == Direction.EFFECT)


# --------------------------------------------------------------------------- #
# Fig 16 -- irreducible vs reducible second-order occurrence
# --------------------------------------------------------------------------- #


def test_ac_fig16a_shared_inputs_irreducible(_iit3):
    """Fig 16A: OR and AND share inputs A, B; {AB = 10} <- {(OR,AND) = 10} at
    0.170 in addition to the four first-order links at 0.415; the
    transition's irreducibility is 0.17 bits."""
    s = _gate_substrate(2, {"OR": lambda a, b: a | b, "AND": lambda a, b: a & b})
    account, _ = _account(s, (1, 0, 0, 0), (1, 0, 1, 0), (0, 1), (2, 3))
    assert account[(Direction.EFFECT, (0,))] == ((2,), 0.415)
    assert account[(Direction.EFFECT, (1,))] == ((3,), 0.415)
    assert account[(Direction.CAUSE, (2,))] == ((0,), 0.415)
    assert account[(Direction.CAUSE, (3,))] == ((1,), 0.415)
    assert account[(Direction.CAUSE, (2, 3))] == ((0, 1), 0.170)
    transition = actual.Transition(s, (1, 0, 0, 0), (1, 0, 1, 0), (0, 1), (2, 3))
    assert round(float(actual.sia(transition).alpha), 2) == 0.17


def test_ac_fig16c_independent_inputs_reducible(_iit3):
    """Fig 16C: with independent inputs (A, B -> OR; C, D -> AND) the
    second-order link is absent and the transition is reducible (0 bits)."""
    s = _gate_substrate(4, {"OR": lambda a, b, c, d: a | b, "AND": lambda a, b, c, d: c & d})
    before, after = (1, 0, 1, 0, 0, 0), (1, 0, 1, 0, 1, 0)
    account, _ = _account(s, before, after, (0, 1, 2, 3), (4, 5))
    assert (Direction.CAUSE, (4, 5)) not in account
    transition = actual.Transition(s, before, after, (0, 1, 2, 3), (4, 5))
    assert float(actual.sia(transition).alpha) == 0.0
```

If `actual.sia(...).alpha` is not the paper's 𝒜 (Appendix A: the total link strength lost under the transition MIP) — check `pyphi/formalism/actual_causation/compute.py` for how the AC system analysis defines its α — replace the two 𝒜 assertions with a docstring note recording that PyPhi's system-level AC quantity is defined differently and is not pinned here. Fig A1C is out of scope.

- [ ] **Step 3: Run and perturbation-verify**

```bash
uv run pytest test/integration/test_paper_reproduction_ac.py -q > /tmp/t15.log 2>&1; echo $?; tail -1 /tmp/t15.log
uv run pytest test/test_examples.py -q > /tmp/t15b.log 2>&1; echo $?; tail -1 /tmp/t15b.log
```
Expected: exit 0. Perturb one α per figure: FAIL; restore. Fig 11 may take tens of seconds (3⁷·4 states); mark it slow if it exceeds ~10 s.

- [ ] **Step 4: Commit**

```bash
cat > changelog.d/paper-reproduction-ac-2019.feature.md <<'EOF'
The paper-reproduction acceptance suite now pins the causal accounts of Albantakis et al. (2019), "What caused what?", Figs 7–16 — including the three-candidate election of Fig 11, the suite's first multi-valued actual-causation reproduction (new example `ac_2019_three_candidate_election_substrate`).
EOF
uvx ruff format pyphi test && uvx ruff check pyphi test
git add pyphi/examples.py test/integration/test_paper_reproduction_ac.py changelog.d/paper-reproduction-ac-2019.feature.md
git commit -m "Pin Albantakis et al. (2019) Figs 10, 11, 13, 15, 16, incl. the first multi-valued AC reproduction (perturbation-verified)"
```

---

### Task 16: Barbosa et al. (2020) — the intrinsic-difference primitive

**Files:**
- Test: `test/measures/test_measures_distribution.py` (append)

- [ ] **Step 1: Append the tests**

```python
# --------------------------------------------------------------------------- #
# Barbosa, Marshall, Streipert, Albantakis & Tononi (2020), "A measure for
# intrinsic information", Sci Rep 10:18803 -- the ID on the paper's channels
# --------------------------------------------------------------------------- #
from pyphi.measures.distribution import intrinsic_difference as _id


def _wire_channel(n_wires, reliabilities):
    """P over 2^N symbols given the sent symbol: wire i delivers its bit with
    probability reliabilities[i] (independent); the sent symbol is all-zeros."""
    p = np.ones(1)
    for r in reliabilities:
        p = np.kron(p, np.array([r, 1 - r]))
    return p


def _uniform(n_wires):
    return np.full(2**n_wires, 2.0**-n_wires)


def test_barbosa_2020_fig2_enclosures():
    """Fig 2A-C: a noiseless bit carries 1 ibit, a noiseless byte 8 ibits,
    and a byte with one noiseless and seven fully noisy wires close to 0."""
    assert float(_id(_wire_channel(1, [1.0]), _uniform(1))) == pytest.approx(1.0)
    assert float(_id(_wire_channel(8, [1.0] * 8), _uniform(8))) == pytest.approx(8.0)
    assert float(_id(_wire_channel(8, [1.0] + [0.5] * 7), _uniform(8))) < 0.01


def test_barbosa_2020_fig2e_one_ibit_byte():
    """Fig 2E: one noiseless wire plus seven at s ~ 0.78 conveys 1 ibit."""
    value = float(_id(_wire_channel(8, [1.0] + [0.78] * 7), _uniform(8)))
    assert value == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize(("n_wires", "expected"), [(1, 0.72), (8, 2.41), (16, 1.77)])
def test_barbosa_2020_fig3_channel_size(n_wires, expected):
    """Fig 3: at r = 0.88 per wire the ID peaks at N = 8 wires
    (0.72 / 2.41 / 1.77 ibits for N = 1 / 8 / 16)."""
    value = float(_id(_wire_channel(n_wires, [0.88] * n_wires), _uniform(n_wires)))
    assert value == pytest.approx(expected, abs=0.01)


def _neuron_fanout(n, t=1.0):
    """Fig 4: N sender neurons all firing (+1) drive N outputs, each firing
    with sigmoid probability of h + b, h = sum of inputs, b = 1 - N. Returns
    (P, Q) over the 2^N output patterns, grouped by firing count."""
    b = 1 - n

    def p_fire(h):
        return 1.0 / (1.0 + np.exp(-2.0 / t * (h + b)))

    # Inputs x in {-1, +1}^N: h = 2*ones - N, weight C(N, ones) / 2^N.
    from math import comb

    p_all = p_fire(n)  # every input +1
    P = np.zeros(2**n)
    Q = np.zeros(2**n)
    for y in range(2**n):
        k = bin(y).count("1")  # outputs firing
        P[y] = p_all**k * (1 - p_all) ** (n - k)
        Q[y] = sum(
            comb(n, ones) / 2**n * p_fire(2 * ones - n) ** k * (1 - p_fire(2 * ones - n)) ** (n - k)
            for ones in range(n + 1)
        )
    return P, Q


@pytest.mark.parametrize(("n", "expected"), [(1, 0.72), (8, 2.90), (16, 2.10)])
def test_barbosa_2020_fig4_neuron_fanout(n, expected):
    """Fig 4C-E (t = 1): senders with 1 / 8 / 16 outputs convey 0.72 / 2.90 /
    2.10 ibits; the maximum is at eight outputs."""
    P, Q = _neuron_fanout(n)
    assert float(_id(P, Q)) == pytest.approx(expected, abs=0.01)
```

`intrinsic_difference(p, q)` takes two distributions (`distribution.py:1111`); confirm it returns the signed max of `p log2(p/q)` (not the absolute variant) and accepts flat arrays. Check the existing parametrized `test_intrinsic_difference` (line ~376) for the call shape.

- [ ] **Step 2: Run and perturbation-verify**

```bash
uv run pytest test/measures/test_measures_distribution.py -q -k barbosa > /tmp/t16.log 2>&1; echo $?; tail -1 /tmp/t16.log
```
Expected: exit 0 for Fig 2 and Fig 4. **Known risk:** for Fig 3 at N = 8 and 16 the analytic value of `r^N · N · log2(2r)` at r = 0.88 is 2.35 and 1.69, not the paper's 2.41 / 1.77 (N = 1 gives 0.72 exactly). If the test reproduces that discrepancy, keep the N = 1 pin, pin N = 8 and 16 with `abs=0.07` and a docstring stating the computed values and that the paper's two larger numbers are consistent with a per-wire reliability of about 0.883 rather than 0.88; do not widen the tolerance further. Perturb one value per test: FAIL; restore.

- [ ] **Step 3: Commit**

```bash
cat > changelog.d/paper-reproduction-barbosa-2020.feature.md <<'EOF'
The intrinsic-difference measure is now pinned against the channel and neuron examples of Barbosa et al. (2020), "A measure for intrinsic information", Figs 2–4.
EOF
uvx ruff format test && uvx ruff check test
git add test/measures/test_measures_distribution.py changelog.d/paper-reproduction-barbosa-2020.feature.md
git commit -m "Pin Barbosa et al. (2020) channel and neuron examples on the intrinsic-difference measure (perturbation-verified)"
```

---

### Task 17: Terminology sweep

**Files:**
- Modify: every hit of the grep below under `pyphi/`, `docs/` (not `_build/`, not `docs/superpowers/`), `pyphi/mcp/content/`

- [ ] **Step 1: List the Eq. 23 uses of the word**

```bash
grep -rn -i "ii cap\|ii-cap\|ii(s) cap\|intrinsic-information cap\|intrinsic information cap\|Eq\. 23 cap\|the cap\b\|capped\|uncapped\|cap term\|cap is\|cap on\|cap applied\|applies the cap\|apply the cap" pyphi docs --include='*.py' --include='*.md' --include='*.rst' | grep -v "_build/\|docs/superpowers/\|purview.*cap\|order cap\|cost cap\|budget cap\|dimension\|max_purview\|\bcaps\b"
```
Read each hit in context; keep only those about Eq. 23.

- [ ] **Step 2: Rewrite each**

Rules: "the ii(s) cap" → "the intrinsic-information requirement"; "capped φ" → "φₛ under the requirement" (or "with the requirement applied"); "uncapped" → "without the requirement"; "applies the cap" → "applies the requirement"; "Eq. 23 cap" → "Eq. 23 requirement"; "cap-biting" → "requirement-binding". Private identifiers (`_apply_ii_cap`, `_cap_one`, `apply_cap` locals) stay. Do not touch `docs/superpowers/` (historical specs/plans) or test names.

- [ ] **Step 3: Verify**

```bash
# Re-run the grep from Step 1: only private identifiers and non-Eq.-23 uses may remain.
uv run pytest pyphi -q --doctest-modules -p no:cacheprovider > /tmp/t17.log 2>&1; echo $?; tail -1 /tmp/t17.log
just docs > /tmp/t17docs.log 2>&1; echo $?; tail -2 /tmp/t17docs.log
```
Expected: doctests pass; docs "build succeeded" with `-W`.

- [ ] **Step 4: Commit**

```bash
cat > changelog.d/requirement-terminology.doc.md <<'EOF'
Public docstrings, the documentation, and the bundled MCP reference now consistently call Mayner et al. (2026) Eq. 23 the intrinsic-information requirement ("with" / "without the requirement") rather than a cap.
EOF
git add -A pyphi docs changelog.d/requirement-terminology.doc.md
git commit -m "Use 'intrinsic-information requirement' consistently in public prose"
```

---

### Task 18: Documentation, changelog fragments, ROADMAP, reproduction docstring

**Files:**
- Modify: `docs/theory/intrinsic-information.md` (new section after "Differentiation and determinism")
- Modify: `docs/tutorials/macro.md` or wherever the Fig 7 lesion is shown (`grep -rn "fig7_inactivated\|condition({" docs/*.md docs/**/*.md`)
- Modify: `test/integration/test_paper_reproduction.py:1-90` (module docstring "Currently covered")
- Modify: `ROADMAP.md` (N22, N23 rows; the "two paper-derived validation directions" paragraph; the N1 row's paper list)
- Create: `changelog.d/paper-reproduction-2026.misc.md`, `changelog.d/paper-reproduction-marshall-2023.misc.md`

- [ ] **Step 1: Theory page**

After the "Differentiation and determinism" section, add:

```markdown
## Reading the two terms

Both terms of the requirement are available on the result. `intrinsic_specification`
gives, per direction, the selectivity-times-informativeness of the specified
state (Eqs. 7 and 9); `intrinsic_differentiation` gives the surprisal of that
state (Eqs. 4 and 6); `intrinsic_information` is their joint minimum (Eq. 13),
and `integrated_fraction` is $\varphi_s / \mathit{ii}(s)$.

```{code-cell} python
sia = pyphi.analyze(pyphi.examples.iit4_2023_fig1a_substrate(), (0, 1, 1), subset=(0, 1)).sia
sia.intrinsic_specification, sia.intrinsic_differentiation, sia.intrinsic_information
```

When the requirement sets $\varphi_s$, `explain()` says which direction and
which term did so:

```{code-cell} python
[f for f in sia.explain().findings if f.kind == "requirement_binding"]
```
```

Adjust the `analyze(...)` call to the real signature (check `pyphi/analyze.py`); the page's first cell already pins the formalism, keep that. Run `just docs` — every cell executes at build.

- [ ] **Step 2: Lesion docs**

Where the docs build the Fig 7C substrate by hand, replace with `substrate.inactivate({"E": 0})` and one sentence: "`Substrate.inactivate` freezes the unit into the dynamics — the paper's *inactivated* unit, as opposed to an *inactive* one (Fig 7B)."

- [ ] **Step 3: Reproduction-suite docstring**

Add to the "Currently covered" list of `test/integration/test_paper_reproduction.py` four bullets — IIT 4.0 (2026) Figs 2, 3, 4; Marshall et al. (2023) Figs 1, 2, 3 — each naming the fixtures and the pinned values, and a pointer to `test_paper_reproduction_ac.py` for the AC figures and to `test/measures/test_measures_distribution.py` for Barbosa (2020).

- [ ] **Step 4: ROADMAP**

- N22 row: "*`2.x`; partially concretized.*" → "**landed 2026-09** (2.0.0 release pass): `intrinsic_specification` / `intrinsic_differentiation` / `intrinsic_information` / `integrated_fraction` on the SIA; `explain()` names the binding term; the determinism sweep is pinned as the 2026 Fig 3 reproduction." Keep the refuted-inequality note.
- N23 row: mark the integrated-fraction ratio landed; the remaining items (determinism/degeneracy covariates, landscape display) stay.
- N1 row (dashboard): append "**2026-09:** Mayner et al. 2026 Figs 2/3/4, Marshall et al. 2023 Figs 1/2/3, AC 2019 Figs 7–16, Barbosa 2020 Figs 2–4."
- The "two paper-derived validation directions" paragraph: the S1 terminal-branch test landed (`test/formalism/test_complexes.py::TestS1TerminalTieBranch`).
- Dashboard "Landed" line: add `Substrate.inactivate` (N14's smallest slice).

- [ ] **Step 5: Fragments and commit**

```bash
cat > changelog.d/paper-reproduction-2026.feature.md <<'EOF'
The paper-reproduction acceptance suite now pins Mayner, Marshall & Tononi (2026): the monad's φₛ peak (Fig 2), the complex-size and differentiation crossovers of the Fig 6D lattice under a determinism sweep (Fig 3), and the macro/micro crossover of the intrinsic-units example (Fig 4). New examples: `mayner_2026_monad_substrate`; `iit4_2023_fig6d_substrate` takes the slope `k`.
EOF
cat > changelog.d/paper-reproduction-marshall-2023.feature.md <<'EOF'
The paper-reproduction acceptance suite now pins Marshall et al. (2023), System Integrated Information: determinism and degeneracy (Fig 1), fault lines and integrated fractions (Fig 2), and the eight-unit universe condensing into three complexes (Fig 3), with new `marshall_2023_fig{1,2,3}_substrate` examples.
EOF
just docs > /tmp/t18docs.log 2>&1; echo $?; tail -2 /tmp/t18docs.log
git add docs ROADMAP.md test/integration/test_paper_reproduction.py changelog.d
git commit -m "Document the 2026 quantities, the lesion helper, and the new paper reproductions"
```

---

### Task 19: Verification and merge

- [ ] **Step 1: Full fast suite (bare invocation — doctest sweep included)**

```bash
uv run pytest -q > /tmp/final-fast.log 2>&1; echo $?; tail -1 /tmp/final-fast.log
```
Expected: exit 0; "N passed" with N > 4563, 0 failed.

- [ ] **Step 2: Slow lane**

```bash
uv run pytest -m slow --slow -q > /tmp/final-slow.log 2>&1; echo $?; tail -1 /tmp/final-slow.log
```
Expected: exit 0; passed count ≥ 254 + the new slow tests.

- [ ] **Step 3: Perf gate and docs**

```bash
uv run pytest test/integration/test_perf_counters.py -q > /tmp/final-perf.log 2>&1; echo $?; tail -1 /tmp/final-perf.log
just docs > /tmp/final-docs.log 2>&1; echo $?; tail -2 /tmp/final-docs.log
```
Expected: all green. (Nothing in this pass touches caching or hot paths, so the call-count pins should be unchanged; a moved pin is a finding to report.)

- [ ] **Step 4: Merge to main (no push)**

```bash
cd /Users/will/projects/pyphi
git merge --no-ff release-pass -m "Merge the 2.0.0 release pass"
git worktree remove .claude/worktrees/release-pass
git branch -d release-pass
```

Then hand back to the release walk: rebuild the 2.0.0 changelog section from the new fragments (`towncrier` over the ~11 new fragments, folded into the existing 2.0.0 section rather than a second heading), and resume `RELEASING.md` from gate 1. Pushing and tagging need the user's explicit consent each time.
