# B13 — Partition-scheme × version constraint: Design

**Status:** approved
**Date:** 2026-06-18
**Wave:** 2 (pre-freeze, config-surface)
**Closes:** the two B13 deferrals (EMD-precision floor; partition-scheme × version)
that were left "pending confirmation experiments" in `pyphi/conf/constraints.py`
and the ROADMAP B13 row.

---

## 1. Motivation

The B13 eager config-combination validator (`pyphi/conf/constraints.py`) ships
one constraint (measure ↔ version) and explicitly defers two others pending
confirmation experiments. Those experiments have now run (see
`b13_experiments/FINDINGS.md`, seed 20260618, raw data saved):

**Q1 — does EMD require `precision ≤ 6`?** No. 2.0 uses POT (`ot.emd2`, an exact
network-simplex LP). Over 5000 random repertoire pairs the EMD identity
`EMD(p, p)` is exactly 0.0, symmetry residual is 5.6e-16 (machine epsilon), and
the solver is deterministic. An IIT 3.0 phi precision sweep (6→13) converges to
2.3125 with an identical MIP at every precision. POT's noise floor sits ten
orders of magnitude below the 1e-6 the historical `pyemd` concern was about. The
`precision: 6` pin is a goldens-calibration choice, not a correctness
requirement. **No constraint** — encoding one would be a false rejection.

**Q2 — are any (system_partition_scheme, version) pairings invalid?** One is.
Running a SIA under every registered system partition scheme:

- **IIT 3.0** raises `ValueError` for 6 of 8 schemes, reactively, inside
  `sia_partitions()` (`pyphi/formalism/iit3/__init__.py:300`). Only
  `DIRECTED_BIPARTITION` and `DIRECTED_BIPARTITION_CUT_ONE` compute. This is a
  real, already-enforced restriction surfaced only at compute time, deep in the
  math.
- **IIT 4.0 (2023)** computes for all 8 schemes without raising; 7 return the
  canonical Φ, while `EDGE_CUT_BIDIRECTIONAL` silently returns a different
  (over-estimated) Φ. The value is well-defined per scheme, there is no existing
  hard rejection in the 4.0 main path, and the *mechanism* partition scheme is
  deliberately left version-agnostic. **Decision (confirmed with the maintainer):
  leave IIT 4.0 open** — no constraint, no warning.

So exactly one constraint is added: make IIT 3.0's existing reactive restriction
eager, the same way the shipped measure constraint makes `check_measure_compatible`
eager.

## 2. Goals

- Add an eager constraint rejecting `IIT_3_0` paired with a
  `system_partition_scheme` outside its compatible set, raising a
  `ConfigurationError` at configuration time (`override` / `load_yaml`) that names
  the two conflicting fields and a concrete fix.
- Keep the eager constraint and the reactive `sia_partitions()` raise reading a
  **single source of truth**, so they cannot drift.
- Leave IIT 4.0 unconstrained on the system scheme (encoded as data, not as a
  special case in the constraint logic).
- Close both deferrals in the docs (the `constraints.py` docstring and ROADMAP),
  recording that EMD-precision needs nothing and IIT 4.0 stays open.

## 3. Non-goals

- Any EMD/precision constraint (confirmed unnecessary).
- Any IIT 4.0 system-scheme constraint or warning (confirmed: leave open).
- Mechanism partition scheme constraints (version-agnostic by design; IIT 3.0
  validly uses `WEDGE_TRIPARTITION` for multivalued networks).
- Changing any computed value — this is validation only; the accepted set is
  exactly what already computes.

## 4. Design

### 4.1 Single source of truth on the formalism

Add a class attribute to the IIT formalism objects, mirroring the existing
`compatible_measures`:

```python
compatible_system_partition_schemes: ClassVar[frozenset[str] | None]
```

- **IIT 3.0**: `frozenset({"DIRECTED_BIPARTITION", "DIRECTED_BIPARTITION_CUT_ONE"})`.
- **IIT 4.0 (2023 and 2026)**: `None` — the sentinel for "unconstrained" (every
  registered scheme is accepted). This encodes the maintainer's "leave open"
  decision as data rather than as a branch in the constraint.

The concrete IIT formalism classes (`IIT3Formalism`, `IIT4_2023Formalism`,
`IIT4_2026Formalism`) structurally satisfy the `PhiFormalism` Protocol rather than
inheriting from a shared concrete base, so there is no base default to rely on.
The attribute is declared on the `PhiFormalism` Protocol (for shape/type
documentation, next to `compatible_measures`/`partition_scheme`) and set
explicitly on all three IIT formalisms (3.0 restricted; both 4.0 = `None`). The
constraint reads it with `getattr(formalism, "compatible_system_partition_schemes",
None)`, so a formalism that omits it is treated as open. (The AC formalism has no
system-partition-scheme concept and is unaffected: the constraint keys off
`config.formalism.iit.version`, always an IIT formalism.)

### 4.2 Reactive site reads the attribute

`pyphi/formalism/iit3/__init__.py`'s `sia_partitions()` currently hardcodes
`valid = ["DIRECTED_BIPARTITION", "DIRECTED_BIPARTITION_CUT_ONE"]`. Refactor it to
read the active formalism's `compatible_system_partition_schemes`, so the reactive
raise and the eager constraint share one definition. The raised message and
behavior are preserved (same set, same `ValueError` at the same point for direct
compute-path callers who bypass config validation).

### 4.3 Eager constraint

Add to `pyphi/conf/constraints.py`, registered alongside the measure constraint:

```python
@register_constraint("system_partition_scheme_compatible_with_version")
def _system_partition_scheme_compatible_with_version(config) -> str | None:
    ...
```

Logic, mirroring `_measure_compatible_with_version`:
- Read `version = config.formalism.iit.version` and
  `scheme = config.formalism.iit.system_partition_scheme`.
- Resolve the active formalism's `compatible_system_partition_schemes` via the
  same lazy-import + bootstrap-window guard already used for `_compatible_measures`
  (return `None`/skip if the formalism registry is not importable yet, or if the
  version is unregistered — the measure constraint already reports an unregistered
  version, so this constraint need not duplicate that).
- If the compatible set is `None`, pass (unconstrained — IIT 4.0).
- If `scheme` not in the set, return a message naming
  `formalism.iit.system_partition_scheme` and `formalism.iit.version`, listing the
  compatible schemes, with a fix ("set the scheme to one of those, or change the
  version").

A shared helper retrieves a named formalism attribute (`compatible_measures`,
`compatible_system_partition_schemes`) with the existing bootstrap/unregistered
handling, so the two constraints do not duplicate the registry-import dance.

### 4.4 Evaluation path

No new wiring: `check_config_constraints` already runs every registered constraint
on `override` / `load_yaml` (gated by `infrastructure.validate_config`), and a
failed apply already restores prior state. The new constraint participates
automatically.

## 5. Testing

`test/test_conf_constraints.py` (extend the existing B13 tests):

- **Enumeration / behavior agreement:** for every registered system partition
  scheme, assert the eager constraint's accept/reject classification under IIT 3.0
  matches whether a real `basic_system().sia()` computes vs raises. (The
  confirmation experiment's table, now a regression.) IIT 4.0 accepts every
  scheme.
- **Eager rejection fires:** `config.override(version="IIT_3_0",
  system_partition_scheme="DIRECTED_SET_PARTITION")` raises `ConfigurationError`
  naming both fields; via `load_yaml` too.
- **IIT 3.0 valid schemes pass:** `DIRECTED_BIPARTITION` and
  `DIRECTED_BIPARTITION_CUT_ONE` are accepted under IIT 3.0.
- **IIT 4.0 unconstrained:** every registered scheme is accepted under
  `IIT_4_0_2023` and `IIT_4_0_2026` (no `ConfigurationError`).
- **Opt-out:** with `validate_config=False`, an otherwise-rejected combination
  applies without raising.
- **Failed apply restores state:** after a rejected `override`, the prior
  `version`/`system_partition_scheme` are intact.
- **Single source of truth:** the reactive `sia_partitions()` raise and the eager
  constraint reject the same set (assert the reactive raise still fires for a
  scheme outside the set when validation is off).
- **Presets pass:** the `iit3`, `iit4_2023`, `iit4_2026` presets all validate.

Verification runs `uv run pytest` **with no path argument** (config surface;
doctest sweep).

## 6. Risks and mitigations

- **Drift between reactive and eager checks.** Mitigated by the single
  source of truth (§4.1–4.2): both read the formalism attribute.
- **False rejection.** The accepted set is exactly what already computes
  (confirmed by experiment); IIT 4.0 stays open, so no currently-working config
  is newly rejected. `validate_config=False` remains a global opt-out.
- **Bootstrap window.** Reuses the established `_FORMALISM_UNAVAILABLE` guard, so
  the constraint is inert during the conf package's auto-load and active for every
  post-import `override`/`load_yaml`.

## 7. Acceptance criteria

- `compatible_system_partition_schemes` on the IIT formalisms (3.0 restricted,
  4.0 `None`); `sia_partitions()` reads it.
- `system_partition_scheme_compatible_with_version` constraint registered and
  rejecting only `IIT_3_0` + out-of-set schemes, with a two-field message + fix.
- IIT 4.0 accepts every registered scheme; presets validate; opt-out works;
  failed apply restores state.
- `constraints.py` docstring + ROADMAP B13 row updated: EMD-precision deferral
  closed (no constraint), IIT 4.0 left open, scheme constraint landed → B13 ✅.
- `uv run pytest` (no path argument) green, including doctests.
