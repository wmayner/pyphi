# Findings from the baseline run (Claude's own, outside the workflow)

---

## F7. ★ RELEASE BLOCKER — the built wheel and sdist contain **zero Python source files**

**File:** `pyproject.toml:119-121` (wheel), `pyproject.toml:109-118` (sdist)
**Severity:** CRITICAL — publishing 2.0 from this config ships an empty package
**Status:** CONFIRMED end-to-end (built, installed, imported)

Surfaced by the audit workflow; I verified it independently from a clean build.

```toml
[tool.hatch.build.targets.wheel]
packages = ["pyphi"]
include = ["pyphi/data/**"]     # <- Hatchling treats `include` as an ALLOWLIST
```

Hatchling's `include` is a whitelist, not an addition. `packages = ["pyphi"]` sets the
package root, but `include` then restricts the payload to `pyphi/data/**` only, so every
`.py` file is filtered out. The sdist allowlist (`README.md`, `CACHING.rst`, `LICENSE.md`,
`CHANGELOG.md`, `pyphi_config.yml`, `redis.conf`, `pyphi/data/**`) likewise has no pattern
matching `pyphi/**/*.py`.

Verified by building:
```
$ python -m build
Successfully built pyphi-...tar.gz and pyphi-...-py3-none-any.whl

wheel: total entries: 25, .py files: 0     # only pyphi/data/*.npy + dist-info
sdist: .py files: 0
```

### The failure is SILENT, which is why it survived

Installed into a clean venv:
```
$ pip install pyphi-...whl && python -c "import pyphi"
import pyphi SUCCEEDED
  __file__: None
  __path__: ['.../site-packages/pyphi']
  is namespace pkg: True
  pyphi.analyze: MISSING -> module 'pyphi' has no attribute 'analyze'
```

Because `pyphi/data/` is installed with no `__init__.py`, Python treats `pyphi` as a
**PEP 420 namespace package**. So `import pyphi` *succeeds* and every actual use fails
with `AttributeError`.

### The CI gate designed to catch this is blind to it

`.github/workflows/build.yml:66-73` — the release-verification job's entire smoke test:

```yaml
- name: Test import
  run: .venv/bin/python -c "import pyphi; print('PyPhi imported successfully')"
```

That passes on the empty package, for the namespace-package reason above. The two
defects compound: the packaging bug empties the wheel, and the only gate that would
catch it is satisfied by a bare import.

### Why nobody noticed

Development always installs editable (`uv pip install -e`), which imports from the
source tree and never exercises the built artifact.

### History

Introduced 2026-01-02 in `ed2fb3db` ("Fix pyright errors in remaining core modules"),
4945 commits ago — i.e. present for the whole 2.0 development period. No release tags
in this clone, so it appears never to have shipped to PyPI; this would break the 2.0
release rather than something already published.

### Fix

Drop the `include` from the wheel target (`packages = ["pyphi"]` alone already picks up
the package, data included via `force-include` or package-data if needed), and add
`pyphi/**/*.py` to the sdist allowlist. Then harden the CI gate so it cannot pass on an
empty package — e.g.:

```yaml
run: |
  .venv/bin/python -c "
  import pyphi, pathlib
  assert pyphi.__file__ is not None, 'pyphi installed as a namespace package (no source!)'
  pyphi.analyze  # a real public symbol
  print('ok', pyphi.__version__)"
```
Better still, run a small slice of the real test suite against the installed wheel.

---

## Findings from the baseline test run

Baseline on `origin/main` @ ce2b2832: **4300 passed, 2 failed, 290 skipped**, 366s.

---

## F1. `test_shard_execution_bounds_caches_by_its_memory_request` is not isolated from the host cgroup

**File:** `test/campaign/test_runner_shards.py:56-86` (assertion at :74)
**Severity:** medium (test defect) — but see F2, which it exposes
**Status:** CONFIRMED by bisection

Failure: `assert 14059220992 == 4026531840`.

`_shard_config()` (`pyphi/campaign/runner.py:119-126`) resolves the cache ceiling by
precedence: `_cgroup_memory_limit() or _granted_memory_bytes() or spec.memory_bytes`.
This container exposes cgroup **v1** `/sys/fs/cgroup/memory/memory.limit_in_bytes`
= 14327656448 (13.34 GiB), so the cgroup branch wins and yields a budget of
14059220992. The test asserts the *planned* figure (`spec.memory_bytes`
→ 4026531840) is used, which only holds when no cgroup limit is readable.

Proof — neutralizing the cgroup read makes it pass:
```python
monkeypatch.setattr(cache_utils, "_cgroup_memory_limit", lambda: 0)
```
→ `1 passed` (vs `1 failed` unpatched).

The isolation pattern already exists in the same file: the sibling test
`test_granted_memory_sets_the_cache_ceiling` (:89) takes `monkeypatch` and its
docstring says "With no cgroup limit to read". This test just doesn't do it.

**Impact:** fails in any container with a memory cgroup — i.e. most CI runners and
most contributor sandboxes. Green locally on bare metal, red in a container.

**Fix:** monkeypatch `_cgroup_memory_limit` to 0 in this test, matching its sibling.

---

## F2. Shard cache ceiling can exceed the shard's own memory request (design hazard)

**File:** `pyphi/campaign/runner.py:119-126`
**Severity:** high if confirmed on a real scheduler — needs maintainer judgement
**Status:** mechanism confirmed locally; operational impact NOT confirmed

The failing test's *other* assertion encodes the intended invariant:

```python
assert 0 < budget < task.spec.memory_bytes   # line 75
```

i.e. a shard's cache ceiling should stay **inside** the memory it requested. The
shipped precedence violates that whenever the readable cgroup is larger than the
request: here a shard that planned for 3.75 GiB gets a 13.1 GiB ceiling.

The hazard is *which* cgroup gets read. `_cgroup_memory_limit()` falls back to
"the hierarchy root inside a container's cgroup namespace" (per
`changelog.d/cgroup-memory-allowance.fix.md`) — that is the **node/container**
limit, not a per-job one. On a pool where several shards share a node, each shard
would size its caches to the whole node's memory while the scheduler only reserved
a fraction for it. That overcommits by roughly the number of co-resident shards.

This is intentional where the cgroup *is* the job's confinement ("asking a
scheduler for more memory now grows the caches to match" — same fragment). The
question is whether the code can distinguish a job-level cgroup from a node-level
one. It currently cannot.

**Open question for the maintainer:** on the target cluster, is the readable
cgroup per-job? If not, the precedence should clamp:
`min(cgroup_limit, granted or spec.memory_bytes)`.

---

## F3. `test_shortcircuit` Hypothesis health-check failure

**File:** `test/parallel/test_parallel.py:67`
**Severity:** low
**Status:** CONFIRMED load-induced, not a code defect

`hypothesis.errors.FailedHealthCheck: Input generation is slow`. Re-run in
isolation: **passes in 0.54s**. The full-suite failure happened while audit agents
saturated all 4 CPUs, so Hypothesis's wall-clock health check tripped on
contention rather than on an expensive strategy.

Not a code defect, but it *is* a CI fragility: any loaded or throttled runner can
trip this. Worth `@settings(suppress_health_check=[HealthCheck.too_slow])` on
timing-sensitive property tests, since a wall-clock assertion in a parallel suite
is inherently flaky.

---

## F4. `.python-version` pins an unresolvable interpreter

**File:** `.python-version`
**Severity:** LOW (onboarding papercut) — downgraded after checking CI
**Status:** CONFIRMED locally; **CI is NOT affected**

Pins `3.13.13`. In this container `uv run` fails outright:

```
error: No interpreter found for Python 3.13.13 in managed installations or search path
uv python install 3.13.13 -> error: No download found for request: cpython-3.13.13-linux-x86_64-gnu
```

**Correction to an earlier assessment:** this does NOT break CI. Every workflow
explicitly overrides the pin — `.github/workflows/test.yml:29` runs
`uv python install ${{ matrix.python-version }}` and `:32` runs
`uv sync --python ${{ matrix.python-version }}`, with the matrix set to
`["3.13", "3.14"]`. The `.python-version` file is never consulted there.

The local failure is most likely uv version skew: this container has uv 0.8.17,
which predates 3.13.13 and so has no download entry for it. A current uv would
probably resolve the pin fine.

Residual (small) issue: a contributor on an older uv, following the
`CLAUDE.md`/`AGENTS.md` instruction to "always use `uv run`", hits a confusing
error in a fresh clone with no hint that the fix is `uv sync --python 3.13`.

**Fix (optional):** pin `3.13` instead of a patch version, or note the minimum uv
version in the contributor docs.

---

## F6. `Complex.__hash__` and `Complex.__eq__` key on different things — contract violation

**File:** `pyphi/models/complex.py:232` (`__eq__`) and `:244` (`__hash__`)
**Severity:** medium — latent internally, but a public-API trap in a 2.0 release
**Status:** CONFIRMED (contract violation demonstrated); internal impact NIL (nothing hashes a Complex today)

Same family as the four bugs that prompted this audit: a silently violated contract.

```python
def __eq__(self, other):                      # :232
    return (self.sia == other.sia
            and self.is_maximal == other.is_maximal
            and self.excluded == other.excluded)

def __hash__(self):                           # :244
    return hash((self.node_indices, self.is_maximal))
```

`__hash__` keys on `node_indices`, which `__eq__` never looks at. Python's contract
requires `a == b ⟹ hash(a) == hash(b)`; hashing on a field equality ignores is the
direction that breaks it. (The reverse — hashing *fewer* fields — is merely extra
collisions, which is what the earlier `CauseEffectStructure` fix addressed.)

Demonstrated:
```
a == b        : True
hash(a) == hash(b) : False
len({a, b})   : 2       # equal objects both survive set insertion
d = {a: ...}; d[b] -> MISS   # an equal key cannot be found
```

**Reachability.** `node_indices` returns the explicit `_node_indices` when set,
else derives from the SIA. Two of three paths are safe:
  - Fresh non-macro complexes (`substrate.py:889`) pass no `node_indices`, so it
    derives from `sia`, and `SystemIrreducibilityAnalysis.__eq__` *does* compare
    `node_indices` (`pyphi/formalism/iit4/__init__.py:206`). Consistent.
  - The serializer round-trips the derived value (`serialize/convert.py:1203-1210`).
    Consistent.
  - **Macro complexes are not.** `macro/search.py:1347` passes
    `node_indices=tuple(sorted(cand.footprint))` — the *micro* footprint — while
    `sia.node_indices` holds the *macro* unit indices. The hash key and the
    equality key therefore describe different things by construction. Two macro
    complexes over different micro constituents whose SIAs compare equal (same
    macro indices, phi, partition, cause/effect) would be `==` with different
    hashes. Symmetric substrates are exactly where SIAs coincide — the same
    permutation-symmetry situation as the earlier tied-specified-state bug.

**Why it isn't biting yet:** no production site hashes a `Complex` (grepped
`condensation.py`, `substrate.py`, `macro/search.py`, `resolve_ties.py`). The
exposure is the public API — `analyze()` hands users `Complex` objects, and
`set(complexes)` or `{complex: value}` is an obvious thing for a researcher to do.

**Fix:** make the two agree. Either hash `(sia, is_maximal, excluded)` to match
`__eq__`, or add `node_indices` to `__eq__` if two complexes over different micro
units are genuinely meant to be distinct — that's a semantic call for the
maintainer. The macro/micro divergence suggests the latter is what's intended.

---

## F5. Skip census — 290 skipped, and one real gap

**Status:** CHARACTERIZED

| Count | Reason | Assessment |
|---:|---|---|
| 254 | `conftest.py:64: need --slow option to run` | **Fine.** Covered by `.github/workflows/slow-tests.yml`, a nightly full-suite run with `--slow`. Not a gap. |
| 29 | `test/macro/test_macro_goldens.py:238` — irreducible large blackbox subsystems, hours; gated on `PYPHI_MACRO_FULL_SWEEP=1` | Reasonable gate. Worth confirming someone runs it before tagging 2.0. |
| 6 | doctests skipped by `+SKIP` | Benign. |
| 1 | `test/test_validate.py:116` | **Real gap — see below.** |

### F5a. Backward-reachability validation is unimplemented, and the test is `skip` not `xfail`

**File:** `test/test_validate.py:116`
**Severity:** medium

```python
@pytest.mark.skip(
    reason="StateUnreachableBackwardsError not raised by current state_reachable; "
           "backward-reachability check pending implementation"
)
def test_validate_state_no_error_2():
```

`StateUnreachableBackwardsError` exists as an exception type but `state_reachable`
never raises it: only forward reachability is checked. So PyPhi accepts a system
state that is globally impossible backwards, silently, and the only test that
would catch it is disabled.

`@pytest.mark.skip` is the wrong marker for a known-missing feature — it reports
green and vanishes into a count of 290. `@pytest.mark.xfail(strict=True)` would
report the gap and, more importantly, **fail loudly the day someone implements
the check**, which is exactly when you want to know.

**For the maintainer:** is backward reachability intended to ship in 2.0, or is
the exception type vestigial? If vestigial, delete both the exception and the
test rather than carrying a disabled test into the release.

---

## F8. `Substrate._fingerprint` is coarser than `__eq__` — content-cache collision returns another substrate's repertoire

**File:** `pyphi/substrate.py:349-362`
**Severity:** HIGH (real wrong-value defect at the kernel cache; end-to-end Φ impact NOT demonstrated)
**Status:** PARTIALLY CONFIRMED — see the split below. Surfaced by the workflow; I verified it directly.

```python
h.update(repr(ftpm.alphabet_sizes).encode())
for i in range(ftpm.n_nodes):
    h.update((ftpm.factor(i) + 0.0).tobytes())     # <- tobytes() is SHAPE-FREE
h.update(self._cm_fingerprint)
```

A `FactoredTPM` factor's size-1 axes encode which units the node depends on
(`core/tpm/factored.py` module docstring). `.tobytes()` discards shape, so two
substrates with identical flat factor values but different dependence structure
digest identically. `_cm_fingerprint` does not rescue it: `from_factored` leaves
`cm` all-ones in both cases (verified), so the CM component matches too.

The docstring claims the digest "covers exactly what `__eq__` compares". It does
not — `__eq__` uses `np.array_equal`, which compares shape.

### CONFIRMED

```python
flat = [0.9, 0.1, 0.2, 0.8]
A = Substrate.from_factored(FactoredTPM([flat.reshape(2,1,2)]*2))   # each node <- node 0
B = Substrate.from_factored(FactoredTPM([flat.reshape(1,2,2)]*2))   # each node <- node 1

A == B                      -> False        # genuinely different systems
A._fingerprint == B._fingerprint -> True    # COLLIDE
System(A,...)._fingerprint == System(B,...)._fingerprint -> True
cm A == cm B == [[1,1],[1,1]]
```

And the collision produces a wrong value at the memoized kernel:

```python
ka = repertoire_algebra._cause_repertoire_inner(sysA, (0,), (0,1))
kb = repertoire_algebra._cause_repertoire_inner(sysB, (0,), (0,1))
ka -> [0.409091 0.409091 0.090909 0.090909]
kb -> [0.409091 0.409091 0.090909 0.090909]   # A's value; B alone gives
                                              # [0.409091 0.090909 0.409091 0.090909]
cache_info()['_cause_repertoire_inner'] -> {'size': 1}   # one entry, B read A's
```

`ContentCache.get_or_compute` looks up `(fingerprint, args)` in a plain dict with
**no equality fallback** (`cache/content.py:108`), and `observe()` is refcount
bookkeeping for eviction only — it does not detect collisions.

### NOT CONFIRMED (contra the finder's claim of "silently share ... Φ")

I could not demonstrate end-to-end Φ corruption:
  - `System.cause_repertoire()` returned correct values for B whether or not A ran
    first, for both `(0,)` and `(0,1)` mechanisms — the public path did not hit the
    colliding kernel entry in these cases.
  - Every 2-node pair I could construct (including a 2-cycle vs. two disconnected
    self-loops, byte-identical) gives φ = 0.0 for both members, so φ cannot separate
    them. Showing order-dependent Φ needs a ≥3-node colliding pair with φ > 0, which
    I did not build.

**So:** the fingerprint is genuinely unsound and the kernel cache demonstrably
returns another substrate's repertoire. Whether a realistic analysis reaches that
entry is unproven. Treat as a real defect of unproven blast radius, not as a
confirmed wrong-Φ bug.

### Fix (trivially correct regardless of blast radius)

```python
for i in range(ftpm.n_nodes):
    f = ftpm.factor(i)
    h.update(repr(f.shape).encode())      # <-- add
    h.update((f + 0.0).tobytes())
```

Given "correctness > performance", this should land regardless of whether the
end-to-end path is provable — a content address that is coarser than equality is
wrong by construction.

**Secondary observation:** `Substrate.from_factored` produced an all-ones `cm` for
factors whose size-1 axes say otherwise. If `cm` is meant to reflect declared
connectivity that may over-specify, fine; if it is meant to track the factors, it
is not doing so. Worth a maintainer decision.

---

## F9. ★★ Φ is NOT invariant under node relabeling — φ-tied purviews decided by hash order

**File:** `pyphi/core/repertoire_algebra.py:901-910`; default at `pyphi/conf/formalism.py:117`
**Severity:** CRITICAL — theory-conformance violation producing wrong Φ on a canonical example
**Status:** FULLY CONFIRMED and ROOT-CAUSED (surfaced by the workflow; I reproduced and isolated it)

Φ is a property of the substrate, not of how its units happen to be indexed.
Relabeling the nodes must leave Φ unchanged. It does not.

### Reproduction — `examples.basic_substrate()`, swapping nodes 1 and 2

```
original   state=(1,0,0)   phi_s=0.0   big_phi=1.0
relabeled  state=(1,0,0)   phi_s=0.0   big_phi=1.125
```

(The state is unchanged by this particular swap, so *only* the labeling differs.)

The permutation is faithful — applying the same involution twice returns the
original substrate bit-for-bit:
```
tpm identical: True
cm  identical: True
```

### Root cause — confirmed by config bisection

| `purview_tie_resolution` | original | relabeled | invariant |
|---|---|---|---|
| `["PHI"]` — **shipped IIT 4.0 default** (`conf/formalism.py:117`) | 1.0 | 1.125 | **NO** |
| `["PHI", "PURVIEW_SIZE"]` — IIT 3.0 preset (`conf/presets.py:84`) | 1.125 | 1.125 | yes |

`potential_purviews` builds candidates as a Python `set` (`repertoire_algebra.py:901`)
and materializes the list by iterating it (`:908`), so candidates reach `find_mice`
in **hash order**. `find_mice` takes `ties[0]`. With the default single-key `"PHI"`
there is no tiebreaker, so which of two φ-tied purviews wins is decided by the hash
order of the candidate tuples — which changes when node indices are permuted.

### Why this looks like an oversight rather than a decision

1. **IIT 3.0 sets the secondary key; IIT 4.0 does not.** `presets.py:84` uses
   `["PHI", "PURVIEW_SIZE"]`; the 4.0 default is the bare string `"PHI"`.
2. **The rest of the pipeline already assumes the size heuristic.**
   `resolve_ties.py:401` breaks cross-purview ties with
   `max(congruent_purview, key=lambda m: len(m.purview))`, and its docstring at :383
   cites the IIT 4.0 S1 "typically favors larger purviews" rationale. So the
   downstream stage applies a heuristic the primary purview selection ignores.
3. **The invariant-restoring setting is also the one that agrees with the S1
   heuristic**, and it yields the same value (1.125) under both labelings — i.e.
   under the shipped default the *original* labeling returns 1.0, which appears to
   be the wrong value, not merely a different one.

### Same family as the two bugs that prompted this audit

The tied-specified-state fix (`876629b`) and this are the same defect shape: a tie
that theory says should be resolved by a principled rule is instead resolved by an
incidental ordering, and the asymmetry only shows up under permutation. The existing
guard `test/test_relabel.py::test_relabel_matches_recomputation` covers `grid3` and
passes; `basic` — a documented example — fails.

### Fix

Make `purview_tie_resolution` default to `("PHI", "PURVIEW_SIZE")` for IIT 4.0, to
match both the IIT 3.0 preset and the S1 heuristic already used downstream. Then
either give `potential_purviews` a canonical (sorted) order so ties are broken
deterministically rather than by hash, or extend the relabel-equivariance test to
the whole example zoo so a non-canonical order cannot pass unnoticed.

**Maintainer decision needed:** changing the default changes published φ values
(1.0 → 1.125 on `basic`), so the golden zoo will move. That is the correct
direction — the current default returns a label-dependent answer — but it is a
formalism change, not a silent bugfix.

---

## F10. B13 config validator never checks `ces_measure` — the measure that defines IIT 3.0's Φ

**File:** `pyphi/conf/constraints.py:158-163`
**Severity:** HIGH — silently wrong Φ (2.135×) in an accepted configuration
**Status:** CONFIRMED

```python
fields_to_check = ["mechanism_phi_measure"]
if getattr(formalism, "uses_system_phi_measure", False):
    fields_to_check.append("system_phi_measure")
# ces_measure is NEVER added
```

IIT 3.0 derives system Φ from the CES distance (`formalism/iit3/__init__.py` calls
`ces_distance`, which reads `config.formalism.iit.ces_measure` at
`measures/ces.py:370`). The validator checks the two measures IIT 3.0 mostly doesn't
use and skips the one that determines its answer.

- `IIT3Formalism.compatible_measures` = `['AID','EMD','ENTROPY_DIFFERENCE','ID','KLD','L1','MP2Q','PSQ2']`
- Global default `ces_measure` = `SUM_SMALL_PHI` — **not in that set**
- The IIT_3_0 preset uses `EMD`

Confirmed silent error, `examples.basic_substrate()` at state (1,0,0):

```
IIT 3.0 preset, ces_measure=EMD (correct) : 2.3125
IIT 3.0 preset, ces_measure=SUM_SMALL_PHI : 1.0833333333333
                            silent error  : 2.135x     (no error, no warning)
```

The pairing is accepted outright: `config.override({"iit.version":"IIT_3_0",
"iit.ces_measure":"SUM_SMALL_PHI"})` raises nothing and warns nothing.

The module docstring (`constraints.py:14-17`) claims the only unchecked fields are
ones "the active formalism never consults" — false for `ces_measure`. And B13's
stated purpose (`conf/infrastructure.py:150-153`) is "rejecting silently-wrong
combinations (e.g. an IIT version paired with a measure it does not define)" —
defeated for exactly the measure that sets the reported Φ.

**Fix:** append `ces_measure` to `fields_to_check` when the formalism derives system
Φ from the CES distance (mirror the existing `uses_system_phi_measure` pattern with a
`uses_ces_measure` declaration).

---

## F11. `pyphi.analyze()` crashes on a validator-approved IIT 3.0 config

**File:** `pyphi/formalism/iit3/` (`IIT3SystemIrreducibilityAnalysis`)
**Severity:** HIGH — public API crash on an accepted configuration
**Status:** CONFIRMED

Found while testing F10. Take `version="IIT_3_0"` and fix each field the validator
names, until it stops complaining (3 fixes: `mechanism_phi_measure="EMD"`,
`system_partition_scheme="DIRECTED_BIPARTITION"`,
`mechanism_partition_scheme="JOINT_BIPARTITION"`). The config is then accepted:

```
version    : IIT_3_0
ces_measure: SUM_SMALL_PHI
pyphi.analyze(basic_substrate(), (1,0,0))
  -> AttributeError: 'IIT3SystemIrreducibilityAnalysis' object has no attribute 'normalized_phi'
```

So the most likely user path to IIT 3.0 — set the version, follow the error messages
— ends in an unhandled `AttributeError` from a public entry point rather than a
usable result or a clear diagnostic.

This interacts with F10: the crash is arguably *lucky*, since it stops a silently
wrong Φ from being reported on that path. The silent-error path (F10) is reached by
users who start from the IIT_3_0 preset and override `ces_measure`, or who load a
YAML setting it.

**Fix:** whatever `analyze()` reads `normalized_phi` for must handle the IIT 3.0 SIA
type (which has no such attribute), and F10's validator gap should close so the
config never reaches that point with an incompatible `ces_measure`.

---

## F12. ★★ ROOT CAUSE of F9 — the IIT 4.0 "largest congruent purview" rule is unreachable

**File:** `pyphi/resolve_ties.py:392-402`; enabled by `pyphi/models/mice.py:232-240`
**Severity:** CRITICAL — understates Φ on the shipping default path
**Status:** FULLY CONFIRMED. Supersedes F9's diagnosis; F9's symptom and fix remain valid but treat the surface.

Surfaced by a round-3 agent (whose own follow-up run was killed by a container
restart); I reproduced and instrumented it.

```python
if state_ties:                                   # ALWAYS truthy — see below
    congruent_state = [m for m in state_ties if m.is_congruent(system_state_spec)]
    if congruent_state:
        return congruent_state[0]                # returns the enumeration-order winner
if purview_ties:                                 # <-- S1 rule, never reached
    congruent_purview = [m for m in purview_ties if m.is_congruent(system_state_spec)]
    if congruent_purview:
        return max(congruent_purview, key=lambda m: len(m.purview))
```

`MaximallyIrreducibleCauseOrEffect.state_ties` (`models/mice.py:232-240`) returns
`(self, *peers)` — it **always contains `self`**, even with no state tie at all. So
`if state_ties:` is always true, and whenever the enumeration-order purview winner is
congruent the function returns it immediately. The cross-purview branch implementing
the IIT 4.0 S1 rule "typically favors larger purviews" is dead code.

### Instrumented proof — full IIT 4.0 CES over `examples.basic_system()`

```
calls          : 4
state branch   : 4
purview branch : 0     <-- S1 "largest congruent purview" never fires
no congruent   : 0
```

Mechanism `(2,)`, EFFECT: purviews `(1,)` and `(0,1)` tie at φ = 1.0, both congruent.
PyPhi selects the *smaller* `(1,)`. A smaller purview means a smaller `purview_union`,
so fewer relations are generated and Φ is understated:

```
shipped (state-first, S1 unreachable) : sum_phi_d=1.0  relations=0.0    big_phi=1.0
purview-first (S1 reachable)          : sum_phi_d=1.0  relations=0.125  big_phi=1.125
```

### This unifies F9

Three independent routes give **1.125**, and only the shipped default gives 1.0:

| route | big_phi |
|---|---|
| shipped default | **1.0** |
| relabel nodes 1↔2 (F9) | 1.125 |
| `purview_tie_resolution=["PHI","PURVIEW_SIZE"]` (F9 bisection) | 1.125 |
| make the S1 purview branch reachable (this finding) | 1.125 |

F9 asked why the IIT 4.0 default `purview_tie_resolution` is a bare `"PHI"` with no
secondary key. This is the answer: the larger-purview preference was **delegated** to
`resolve_distinction_tie`, and the delegation silently never executes. The bare
default is not an oversight in isolation — it is correct *given* a working
delegation, and wrong because the delegation is dead.

So F9's "change the default" fix works but treats the symptom. The root fix belongs
in the tie function.

### Fix — needs a theory decision, not just a code change

The docstring at `resolve_ties.py:385-390` explicitly documents the current
precedence ("State-tie congruence is preferred over purview-tie heuristic ... only
when none is congruent does the cross-purview branch fire"), so the ordering is
deliberate. But given `state_ties` always contains `self`, that documented ordering
makes the second branch unreachable by construction rather than merely
lower-priority.

Two candidate repairs:
1. Guard the state branch on a *real* tie — `if len(state_ties) > 1:` — so a
   degenerate single-element `state_ties` falls through to the purview rule.
2. Run the purview rule first when `len(purview_ties) > 1`, as the round-3 agent's
   patch did.

**Maintainer call:** which precedence does IIT 4.0 actually specify — state-tie
congruence before purview size, or purview size before state congruence? Whichever
is chosen, the golden zoo moves (Φ 1.0 → 1.125 on `basic`, relations 0 → 1).

### Open — the paper reproductions

An agent was mid-way through comparing `test/integration/test_paper_reproduction.py`
under both variants when the container restarted. Both runs passed identically
through 13 of 19 tests (78%) before being killed, so the comparison is **suggestive
but incomplete**. It should be finished before either repair lands — those tests pin
published IIT 4.0, IIT 3.0, AC, and Gomez values, and are the real check on whether
this changes a reproduced result.
# Addendum — local audit rerun (2026-08-17)

Verified on `/Users/will/projects/pyphi`, branch `claude/pyphi-2.0-audit-r8qolt`.
New findings; the first pass's thirteen are in `2026-08-16__PRE-RELEASE-AUDIT.md`.

Run: 307 agents, 3 rounds. **182 completed, 125 failed on the account's monthly spend
limit** (including the synthesis agent and the entire tail of the drain).

---

## F14. ★★ `complexes()` reports a different major complex run-to-run under parallel evaluation

**File:** `pyphi/substrate.py:810` (the `map_reduce` call in `all_sias`)
**Severity:** CRITICAL — identical input, different scientific answer on rerun
**Status:** HAND-VERIFIED, with a sequential control

Fixture: one 4-unit substrate with two *independent* noisy copy-loop pairs `{A,B}` and
`{C,D}`, `p = 0.9`, state `(1,1,1,1)`. Both tie at φₛ = 0.304006187.

```
parallel complex evaluation ON, 12 runs:
  (0,1), (2,3), (2,3), (2,3), (0,1), (2,3), …   -> 2 distinct major complexes
parallel complex evaluation OFF (default), 12 runs:
  -> 1 distinct: ((2,3), 0.304006187)
```

The sequential control rules out the tie being resolved randomly elsewhere: sequential
is perfectly stable, so the nondeterminism is the parallel path.

Causal chain, every link checked against the code:
1. `parallel/__init__.py:166` — `map_reduce` defaults to `ordered: bool = False`.
2. Process backend collects via `as_completed` → worker-completion order.
3. `substrate.py:810` — `all_sias` passes neither `ordered=True` nor a re-sort.
4. `condensation.py:241` — docstring: *"Within-tier presentation order follows the input order."*
5. `condensation.py:251-252` — `tier_accepted.sort(key=lambda c: position[id(c)])` — sorts by input position.
6. `substrate.py:890` — `complexes()` stamps `is_maximal=(i == 0)`.

**Qualifier:** `parallel_complex_evaluation["parallel"]` is False by default, so a stock
install is unaffected. It is enabled by a documented switch (`pyphi.config`, MCP
`configure_parallel(levels=["complexes"])`) — so it hits the users who turned on
parallelism for large substrates, i.e. the runs most expensive to repeat and least
likely to be spot-checked.

**Fix:** `ordered=True` at `substrate.py:810`. `all_sias` has no short-circuit predicate
and no `size_func`, so ordering is free (`map_reduce` rejects `size_func` with
`ordered=True` at `parallel/__init__.py:195-198`; not applicable here). Or sort
`candidate_sias` canonically in `complexes()` — `queries.py:349` already does this for
the purview sweep. Regression test: repeated `complexes()` under
`parallel_complex_evaluation` on the tied fixture; confirm it fails unfixed.

Same disease as the two confirmed tie bugs, but worse: those are *wrong*, this one is
*non-reproducible*.

---

## F15. `towncrier build` DOES abort — correcting my own earlier refutation

**File:** `CHANGELOG.md:6`
**Severity:** HIGH — blocks cutting the 2.0.0 release notes
**Status:** HAND-VERIFIED. **This reverses a "refuted" verdict I issued earlier.**

I previously tested `towncrier build --version 2.0.0 --draft`, saw exit 0 and a
well-formed changelog, and called the finding refuted. **That test was inadequate:**
`--draft` only renders; it never calls `append_to_newsfile`, which is where the failure
lives. The real write path aborts:

```
$ uv run towncrier build --version 2.0.0 --keep
ValueError: It seems you've already produced newsfiles for this version?
  towncrier/_writer.py:53, in append_to_newsfile
```

Cause: a hand-written `2.0.0` section already sits directly below the insertion marker.
`CHANGELOG.md:4` is the marker
(`<!-- Towncrier will insert release notes here. … -->`); `CHANGELOG.md:6-8` is
`2.0.0` / `-----` / `_(unreleased)_`. towncrier refuses to insert a version it can
already see.

(Run with `--keep` and the file restored from git immediately after; all 31 fragments
intact, tree clean.)

**Fix (maintainer's call):** either delete the hand-written 2.0.0 section and let
towncrier generate it from the 31 fragments, or keep the hand-written notes and consume
the fragments some other way. The two cannot coexist as they stand.

**Lesson for the rest of this report:** a smoke test that avoids the mutating path can
"pass" a broken pipeline. Same shape as the release blocker in the first pass, where CI's
`import pyphi` passed on an empty wheel.

---

## The other 45 confirmed findings — READ THE VERIFICATION TIER

The run returned **47 confirmed (46 after dedup): 7 critical, 39 high**. I have
hand-verified only the two above. The rest carry the panel's word alone, and that word
is worth less than it looks:

**The 3-vote adversarial panel refuted nothing.** All 124 entries in the `rejected` list
have empty reasoning — they are agents that *failed on the spend limit*, not findings
that were argued down. Across three rounds the panel confirmed every finding it actually
adjudicated. A verifier that never says no is not verifying.

Independently, I disproved one panel-confirmed "critical" by hand in the first pass
(`.python-version` breaking CI — CI overrides the pin), and nearly disproved a second
incorrectly (above).

So treat the list below as **candidates ranked by plausibility**, not findings.

### Confirmed criticals (deduped) — none hand-verified except where noted

| File:line | Claim |
|---|---|
| `campaign/merge.py:80` | Sharded CES campaigns return wrong φ_d, wrong distinctions and wrong Φ — partition-stride merge |
| `serialize/schema.py:138` | Distinction `normalized_phi` silently recomputed from ambient global config on deserialization |
| `system.py:806` | `sia()` excludes `system_state` from the disk-cache key; a non-canonical state poisons the entry |
| `conf/_global.py:535` | `config.override()` is process-global, not thread-local; concurrent scopes return φ under the wrong config |
| `serialize/convert.py:596` | Saving a `Complex` from `macro.complexes` crashes on the default path |
| `docs/howto/configure.md:147` | Documented config example selects IIT 4.0 2023 by version alone, silently computes 2026 |

`campaign/merge.py:80` is the one I would check first: a wrong-Φ claim in a shipping
feature, and the same file appears again at `:52` for an exact-tie margin bug.

### Recurring themes in the high tier

- **`relations.py:153`** (3 separate agents) — `Relation` declares `OrderableByPhi` but
  `frozenset` wins the MRO, so `max()`/`sorted()` over relations order by *subset*, not φ.
- **`cache/policy.py:53-54`** (4 agents) — `clear_all()`/`clear()` bypass the owning
  store, leaving stale byte weight and a latched budget that can disable the cache.
- **`conf/_global.py`** (4 agents) — thread-scoping of `config.override()`, plus
  `:480` direct writes skipping the cross-field validator.
- **`serialize/convert.py`** (5 agents) — `result.config` degrading from `ConfigSnapshot`
  to plain `dict` across save/load, relabelling IIT 3.0's Φ as φₛ; MICE purview-tie
  tuples gaining a duplicate member.
- **`measures/distribution.py:1407`** (3 agents) — cause-side intrinsic differentiation
  from unnormalized likelihoods, contra Eqs. 6 and 8.
- **`pyphi_config_3.0.yml`** — shipped reference config cannot be loaded (stale field
  name) and does not reproduce the IIT 3.0 formalism it claims to mirror.

### Never adjudicated

**124 findings** were never verified — their agents died on the spend limit. They are
neither confirmed nor refuted. The synthesis agent died the same way, which is why this
section is written by hand.
# Round 2 hand-verified findings (local machine)

---

## R1. `numpy_aware_eq` broadcasts — docstring false, test gives false assurance

**File:** `pyphi/models/cmp.py:138-141` (docstring claim at `:135-136`); test at
`test/models/test_models.py:301`
**Severity:** LOW — real defect, **not reachable** through any current caller
**Status:** CONFIRMED as a defect; the alarming reachability claim is REFUTED

Surfaced by a completeness critic, which claimed it "blinds every golden fixture in
the suite." That part is wrong — see below.

The docstring states: *"Shape-mismatched or non-numeric arrays compare unequal rather
than raising."* The implementation delegates to `np.allclose`, which **broadcasts**:

```
(2,)   vs (2,2)   all ones      -> eq=True
(2,)   vs ()      scalar 1.0    -> eq=True
(0,)   vs (1,0)   empty         -> eq=True
(2,1)  vs (2,2)   repertoire    -> eq=True
(2,)   vs (3,)    non-broadcast -> eq=False   <- the only False
```

`np.allclose` raises `ValueError` only for *non-broadcastable* shapes, so the
`except` clause catches exactly the one case the test exercises:

```python
def test_numpy_aware_eq_array_shape_mismatch_returns_false():
    a_ = np.zeros(3); b_ = np.zeros(4)      # non-broadcastable
    assert not models.cmp.numpy_aware_eq(a_, b_)
```

The test pins the docstring's claim using the one shape pair where it happens to
hold. Every broadcast-compatible mismatch silently compares equal.

Same family as the already-fixed set-vs-list positional-zip bug (`cbe4ac71`): that
change repaired the set branch and left the array branch broadcasting.

### Reachability — REFUTED (this is the important correction)

The critic's theory was that repertoires over different purviews differ only in
singleton axes, so two RIAs over *different purviews* could compare equal. They
cannot. Every caller guards on purview first, and repertoire shape is determined by
the purview:

| caller | guard before `numpy_aware_eq` |
|---|---|
| `models/ria.py:510` | `self.purview != other.purview` at `:504` |
| `models/distinction.py:320,322` | `cause_purview` `:313`, `effect_purview` `:315` |
| `models/state_specification.py:202,204` | `self.purview != other.purview` at `:196` |

Those are the only three call sites in `pyphi/`. So no current path reaches the
broadcast case, and the golden fixtures are **not** blinded.

**What it actually is:** a latent trap. Any future caller comparing arrays whose
shapes are not already pinned by a preceding guard gets silent wrong equality, and
both the docstring and the test say that cannot happen.

**Fix:** compare shapes explicitly before `allclose`.

```python
if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
    if np.shape(a) != np.shape(b):
        return False
    try:
        return np.allclose(a, b, rtol=EQUALITY_TOLERANCE, atol=EQUALITY_TOLERANCE)
    except (ValueError, TypeError):
        return False
```
Add a broadcast-compatible case to the test — `np.zeros(2)` vs `np.zeros((2, 2))` —
since the existing one cannot fail against this defect.

---

## R2. `complexes()` order nondeterminism — SUPERSEDED BY F14 (my test was too weak)

> **SUPERSEDED.** F14 above confirms this defect with a stronger experiment: a
> purpose-built 4-unit fixture (two independent noisy copy-loop pairs, both tied at
> φₛ = 0.304006187) flips between 2 distinct major complexes over 12 parallel runs,
> with a stable sequential control. My test below used `grid3_substrate` and saw no
> flip in 10 runs — that fixture lacks the tie structure needed to expose it, so this
> is a weaker experiment, **not** a refutation. Retained only for the mechanism trace
> and the `ordered=True` recommendation, both of which still stand.


**File:** `pyphi/substrate.py:810` (inside `all_sias`), consumed via
`irreducible_sias` → `complexes`; tie-break at `pyphi/condensation.py:116,252,372`
**Severity:** LOW as demonstrated — mechanism is real, impact not shown
**Status:** claim NOT reproduced in 10 runs on a genuinely tied case

The agent claimed `complexes()` / `maximal_complex()` pick a different major complex
run-to-run under parallel evaluation. What I found:

### The mechanism is real in the code

- `map_reduce` signature defaults to `ordered: bool = False`
  (`pyphi/parallel/__init__.py:166`), and the call at `substrate.py:810` does not
  override it — so under parallelism results can arrive in completion order.
- The call chain is as claimed: `complexes()` (`:837`) → `irreducible_sias()`
  (`:822`) → `all_sias()` (`:782`) → that `map_reduce`.
- Condensation breaks φ-ties **by input position**: `condensation.py:109` sorts by
  `-phi` (stable), `:116` re-sorts each tolerant tier by the original index
  `pair[0]`, and `:252`/`:372` sort by a recorded `position[id(c)]`. So if input
  order varies, tie order varies with it.

### But it did not reproduce

`grid3_substrate` at `(0,0,0)` yields a genuine tie — complexes `(2,)` and `(0,)`
both at φ = 0.070096726353:

```
sequential order : ((2,), (0,), (1,))
10 parallel runs : ((2,), (0,), (1,))  x10
distinct orders  : 1
major complex    : (2,)   — stable, never flipped
```

`iit4_2023_fig1a_substrate` at `(0,0,0)` has distinct φ (0.214 / 0.065 / 0.039), so
the `-phi` sort determines order there regardless of arrival — also stable over 10 runs.

**Caveat on my own test:** an attempt to instrument `map_reduce` to confirm the
parallel branch actually engaged came back with no recorded calls, which I could not
explain. So I cannot prove parallelism was exercised, and the 10 stable runs may
reflect a workload too small to reorder rather than a real guarantee.

### Recommendation

Not worth further reachability work. The fix is one keyword argument and removes the
question permanently:

```python
result = map_reduce(sia_fn, iterable, ordered=True, ...)   # substrate.py:810
```

Ordering a few thousand candidate SIAs costs nothing next to computing them, and the
downstream tie-break at `condensation.py:116` explicitly assumes input position is
meaningful. Either pass `ordered=True`, or make the tie-break independent of input
order (e.g. secondary key on sorted node indices) so the assumption is not load-bearing.

---

## R3. `Relation` / `RelationFace` ordering is frozenset subset logic, not φ — `max()` returns the wrong face

**File:** `pyphi/relations.py:153` (`Relation` bases) and `:69` (`RelationFace.__lt__`)
**Severity:** HIGH as a public-API contract break; latent internally
**Status:** CONFIRMED, and worse than reported

```python
class Relation(Displayable, ToPandasMixin, frozenset, cmp.OrderableByPhi):
```

`frozenset` precedes `OrderableByPhi` in the MRO, so the φ ordering is shadowed:

```
Relation MRO        : Relation, Displayable, ToPandasMixin, frozenset, OrderableByPhi, Orderable
Relation.__lt__     -> frozenset.__lt__      (subset test)
Relation.__gt__     -> frozenset.__gt__      (superset test)
RelationFace.__lt__ -> RelationFace.__lt__   (phi, correct)
RelationFace.__gt__ -> frozenset.__gt__      (superset test)
```

`RelationFace` is the worse case, because its operators are *mixed*. With
`lo = RelationFace({1,2,3}, phi=0.01)` and `hi = RelationFace({1}, phi=99.0)`:

```
lo <  hi : True     (phi ordering, correct)
hi >  lo : False    (superset test — DISAGREES with the line above)
max(lo, hi).phi = 0.01     <- max returns the LOWER phi
sorted([lo, hi]) -> [0.01, 99.0]   <- sorted is correct
```

So `a < b` and `b > a` disagree (antisymmetry broken), and `sorted()` and `max()`
disagree with each other on the same pair. `max` uses `>`, which is the inherited
superset test.

The `@total_ordering` at `:69` is applied to the **method** `__lt__` rather than to
the class — the in-repo `type: ignore` comment even notes "total_ordering expects a
class not instance" — so it never fills in the missing operators.

### Reachability — latent

Production avoids bare comparison: `max_phi()` (`:606`) extracts `float(relation.phi)`
before `max`, and `strongest()` sorts with an explicit `key=`. I found no production
site doing `max(faces)` or comparing two Relations directly.

The exposure is the public API. `Relation` and `RelationFace` are public types with a
`phi` attribute and an ordering mixin that advertises φ ordering; `max(faces)` is the
obvious thing to write and silently returns the wrong one.

**Fix:** put `cmp.OrderableByPhi` before `frozenset` in the bases, or define all four
operators explicitly on both classes. Apply `@total_ordering` to the class, not the
method. Add a test asserting `max()` picks max φ and that `a < b` implies `b > a`.

---

## R4. ★★ Sharded campaign merge inflates distinction φ — reports distinctions that do not exist

**File:** `pyphi/campaign/merge.py:80` (`merge_stride_rias`); pin filtering at
`pyphi/formalism/iit4/formalism.py:259`
**Severity:** CRITICAL — wrong φ in the sharded/cluster path
**Status:** FULLY CONFIRMED and quantified

### The unsound step

φ for a pin is a **minimum over partitions**; selection among pins is a **maximum**.
Each stride reports `ria._state_ties`, which `_find_mip_iit4` has already reduced to
that stride's maximum-φ pins. `merge_stride_rias` then takes, per pin, the minimum
over *only the strides that reported it*.

Collapsing the max before the cross-stride min is unsound: if the partition that
drives pin *p* to its global minimum lies in a stride where some other pin *q* wins
locally, *p* is absent from that stride's report and its minimising value is never seen.

### Reproduced — `rule110_system`, mechanism (0,1), CAUSE, purview (0,2), k=3

```
FULL sweep:      phi = 0.0   state = (0, 0)
  stride 0/3: pins = [((0,0), 0.207519)]
  stride 1/3: pins = [((1,1), 0.207519)]   <- (0,0) dropped; this stride holds its phi=0 partition
  stride 2/3: pins = [((0,0), 0.207519), ((1,1), 0.207519)]
MERGED (sharded): phi = 0.20751874963942188   state = (0, 0)
```

A distinction that is genuinely **reducible (φ = 0)** is reported with **φ = 0.2075**.
A sharded campaign therefore reports distinctions that do not exist, which propagates
through `merge_purview_rias` → MICE → distinction → Φ-structure.

### Scope — swept every (mechanism, direction, purview, k) on two systems

```
basic_substrate (1,0,0)  : 84 combos checked, 0 mismatches
rule110_system           : 84 combos checked, 4 mismatches
   mech=(0,1) CAUSE  purv=(0,2)   k=3  full 0.0  -> merged 0.2075
   mech=(0,1) EFFECT purv=(0,1,2) k=3  full 0.25 -> merged 0.3538
   mech=(0,2) CAUSE  purv=(1,2)   k=3  full 0.0  -> merged 0.2075
   mech=(1,2) CAUSE  purv=(0,1)   k=3  full 0.0  -> merged 0.2075   state (0,0) -> (1,1)
```

Three of four turn a non-existent distinction into an existing one; one also changes
the specified state, which changes congruence resolution and therefore the relations.

### Why the guard missed it — fixture structure, not tuple choice

`test/campaign/test_merge.py:20 test_stride_merge_equals_full_find_mip` asserts
exactly the right invariant (merged φ, partition and specified state all equal the
full sweep). It runs it on **one** tuple: `basic_substrate` at (1,0,0), mechanism
(0,1), purview (0,2), EFFECT, k=3.

The sweep above shows `basic_substrate` has **zero** mismatches across all 84
combinations — the bug cannot manifest on that fixture at all. It needs enough
tie structure for a pin to lose locally in the stride holding its minimising
partition. So this is the project's own documented blind spot (`CLAUDE.md`: the
golden zoo tops out at four units and size-driven costs are invisible to it),
in a new place: a correct assertion pinned to a fixture that cannot fail it.

**Fix:** strides must report every pin's φ per stride, not only the stride-local
argmax — i.e. do the cross-stride **min per pin first**, and only then the max over
pins. Then extend the guard to a fixture with real tie structure (`rule110_system`)
and parametrise over direction and k; a single-tuple assertion cannot see this class.

---

# Round 3 hand-verified findings (Tier A of the handoff)

All six Tier A candidates were verified with executed repros on this machine
(2026-08-17). **All six are real.** One (R10) needed a scope correction: the
raw finding's evidence overstated where the failure manifests.

## R5. `fig5b_substrate` implements B as an OR gate; the paper and its own diagram say AND

**File:** `pyphi/examples.py:1031`
**Severity:** HIGH — a golden-fixture substrate is not the published one
**Status:** CONFIRMED (TPM tabulation + reading the 2014 paper figure)

The docstring diagram labels B `(AND)`. The shipped TPM column for B equals
`OR(A, C)` in all 8 rows — row `(1,0,0) -> B'=1` alone rules out AND. The
rendered figure (papers/2014__oizumi-et-al__iit-3.0.pdf, p. 8, panel B)
unambiguously labels the units NULL (A), AND (B), OR (C); C = `OR(A, B)` is
correctly transcribed. The sibling `fig5a_substrate` was verified correct
earlier, so this is a transcription error, not a labeling convention.
Consequence: `fig5b` goldens in `test/formalism/test_iit4.py` /
`test_iit4_robust.py` are locked to a substrate that is not Figure 5B.
Fixing the TPM (B column `[0,0,0,0,0,1,0,1]`) moves those goldens.

## R6. `differentiation_macro_tpm` divides the p² term by 3

**File:** `pyphi/examples.py:1531`
**Severity:** MEDIUM — public example returns a fabricated number (unused in-repo)
**Status:** CONFIRMED (by inspection; the ε=0 limit is decisive)

The comment defines the coarse-graining: micro `(1,1) -> ` macro 1, all other
micro states `->` macro 0. The macro row-0 probability is the uniform average
of P(next micro = (1,1)) over three micro states: `p² + (2/3)·p·ε`. The code
computes `(p*p + 2*p*epsilon) / 3`, also dividing p² by 3. At ε=0 the three
micro states have literally identical dynamics, so any defensible averaging
gives p²; the code gives p²/3. Fix: `p*p + 2*p*epsilon/3`.

## R7. AC MIP search returns None when tied partitions share an edge cut — links dropped, non-minimal purviews reported

**File:** `pyphi/formalism/actual_causation/compute.py:315`; cascade at
`pyphi/resolve_ties.py:283-288`
**Severity:** HIGH — silently wrong causal accounts
**Status:** CONFIRMED (live repro, 5 hits in 10 random 3-node substrates)

`resolve_ac_partition_tie` has two levels: argmin |α|, then argmin
`partition.lex_key()`. `lex_key()` is the cut-matrix bytes, and
`JOINT_PARTITION_ALL` generates the complete cut in two structurally distinct
forms for any mechanism of size ≥ 2 (`[(mech,∅),(∅,purview)]` and the
per-element split). Both forms share the cut matrix and the partitioned
probability, so both cascade levels leave two survivors, `cascade()` returns
`resolved=None`, `_find_mip` returns None (violating its documented return
type), and `_find_causal_link` silently drops that purview at
`compute.py:402`.

Repro (seed 20260817, ten random 3-node deterministic substrates, all
realizable transitions): 1003 (mechanism, purview) cells with all-positive α;
**5 returned None**. Damage confirmed on the first case: mechanism (1,2),
EFFECT — purview `(0,)` at α=0.415037 was dropped and the link reported
purview `(0,2)` instead, a non-minimal purview under the paper's Exclusion
minimality (Definition 1). The 2-unit OR-AND paper example never trips it
(0 of its cells), which is why the paper-reproduction suite is green.

Fix note: the two survivors are the *same partition* (identical edge cut), so
resolving to either is sound — dedupe candidates by `lex_key` before the
cascade, or take any survivor on exhaustion.

## R8. Sharded SIA merge selects a different MIP and specified system state

**File:** `pyphi/campaign/merge.py:144` (`merge_sia_strides`)
**Severity:** HIGH — sharded campaigns resolve congruence against a different state
**Status:** CONFIRMED (rule110 k=3 state flip reproduced; xor MIP flips at k=2 and k=3)

Each stride's `sia()` runs the full (cause, effect) system-state cascade
using stride-local φₛ and reports only the post-cascade tie set; partitions
belonging to a pair that lost locally never reach the merge. Reproduced with
`shortcircuit_sia=False` under IIT_4_0_2026:

```
xor_system      k=2: merged MIP ≠ full MIP (6-edge complete cut vs 4-edge cut)
xor_system      k=3: merged MIP ≠ full MIP (different 4-edge cut)
rule110_system  k=3: merged specified CAUSE state (1,1,1) vs full (0,0,0);
                     merged MIP also differs. φₛ = 0 in both.
```

`sia.system_state` is what `ces()` passes to `resolve_congruence`, so a
sharded campaign can produce a different Φ-structure than the same analysis
unsharded. No φₛ divergence surfaced on these fixtures (all mismatches at
φₛ=0), matching the raw finding; the max-of-min hazard for φₛ itself remains
demonstrated only at the distinction level (R4). The existing guard
`test_sia_stride_merge_equals_full` uses `basic_substrate` k=2, which cannot
fail (same blind-spot class as R4's guard).

## R9. `sweep(formalisms=None)` silently replaces the ambient formalism with the version preset

**File:** `pyphi/sweep.py:87` (`_normalize_formalisms`); override sites at
`:238` and `:266`
**Severity:** HIGH — silent wrong values in a documented API path
**Status:** CONFIRMED (2.13× Φ discrepancy demonstrated)

The docstring documents `formalisms=None` as "the active formalism", but the
code resolves None to `config.formalism.iit.version` and then applies
`config.override(**presets.by_name[formalism])` — resetting every iit field
to the preset. Demonstration: ambient config = IIT_3_0 preset with
`ces_measure="SUM_SMALL_PHI"`, `basic_substrate` (1,0,0):

```
analyze under ambient config:  Φ = 1.083333
sweep under the same ambient:  Φ = 2.312499   (the pure-preset value)
```

The sweep's `formalism` column reads `IIT_3_0` either way, so the table
looks correct. `analyze` (nullcontext when formalism is None) and
`optimize._eval_batch` both honor the ambient config; `sweep` is the odd one
out. Method note: the first attempted demonstration customized
`background_conditioning` — that field is inert on full-system analyses of
these fixtures (control: 0 of 72 cells changed), a reminder that a passing
comparison without a positive control is no evidence.

## R10. AC cause repertoires weight background units by the observed-state posterior instead of fixing them — scope corrected

**File:** `pyphi/system.py:363-386` (`cause_marginal` ignores
`external_indices` under CAUSAL_MARGINALIZATION);
`pyphi/core/tpm/marginalization.py:121`; docstring contradiction at
`pyphi/actual.py:132-136`
**Severity:** HIGH — wrong cause-side α whenever a background unit's observed
transition is informative about its past
**Status:** CONFIRMED, with the raw finding's evidence corrected

`TransitionSystem._underlying_system` sets `external_indices = substrate −
cause_indices` and its docstring claims those units "are held fixed in their
actual state as background conditions (Albantakis et al. 2019, Section
3.3)". The effect side does exactly that (`effect_marginal` conditions on
`external_indices`). The cause side cannot: under the pinned
`CAUSAL_MARGINALIZATION` convention, `System.cause_marginal` calls
`_marginalize_cause(tpm, state, node_indices)` — `external_indices` never
reaches it — and background past states are weighted by the IIT 4.0 Eq. 4
posterior `pr_bg/norm` computed from the observed after state.

**Scope correction:** the raw finding claimed pyphi returns 1.2345 on the
paper's Figure 8B itself. Not quite — modeling the inputs as self-loop
copies (the natural realizable encoding), the Eq. 4 posterior for background
D collapses onto its actual state, and pyphi reproduces **both** 8A and 8B
exactly (all seven 8B effect links 1.0, cause link 3.0000). The deviation
appears the moment the background unit's dynamics make its observed
transition informative without being determining. Discriminating fixture:
same majority gate, D stochastic with p(D'=0|D=0)=0.2, p(D'=0|D=1)=0.8.
Paper semantics (D fixed at 0) still give cause α = 3.0; pyphi returns
**1.2345** = log2(8/(0.2+4·0.8)) — the exact Eq.-4-posterior prediction —
while the seven effect links stay at 1.0. The asymmetry between the two
directions is the contract violation; which behavior is *wanted* for AC is a
theory decision (the paper is explicit: "the background variables are fixed
in their actual state U = u"), but the docstring is false as shipped either
way. Also confirms the sibling finding that `noise_background` is a no-op on
the cause side (nothing consumes `external_indices` there).

---

# Fixes applied on this branch (2026-08-17)

## F7 — FIXED, with a path clarification worth recording

Reproduction detail the original writeup lacked: the defect is **path-dependent**.
`uv build --wheel` (direct from the source tree) produced a *complete* wheel with
all 183 `.py` files even under the broken config — only the default path
(`uv build`, which builds the wheel **from the sdist**) produced the empty
artifact, because the sdist allowlist is what dropped the source. A negative
control that builds only the wheel directly "refutes" F7; the default path
reproduces it exactly. (Same lesson as F15 and R9: the check must go through
the failing path.)

Also surfaced while fixing: the sdist allowlist patterns were **unanchored**, so
`LICENSE.md` matched every `LICENSE.md` under a local virtualenv (24 junk files
in the sdist). Patterns are now anchored with a leading `/`.

Fix landed: wheel target uses `packages = ["pyphi"]` alone; sdist allowlist
gains `/pyphi/**`; `build.yml` gate now asserts `pyphi.__file__ is not None`
and resolves `pyphi.analyze` + `pyphi.examples.basic_substrate()`. Verified:
fixed artifacts install into a clean venv and run a real φₛ analysis; the
hardened gate fails against the broken wheel (old gate passes it).

## Tier B contract violations — FIXED (one commit)

All seven of the handoff's Tier B items landed with regression tests:
`numpy_aware_eq` shape guard · `complexes()` deterministic candidate order
under parallel evaluation (F14; guard test verified to fail with the fix
reverted) · `TransitionSystem.__eq__`/`__hash__` include `noise_background` ·
`Relation`/`RelationFace` order by φ (equality/hash keep set semantics) ·
`Account` order-insensitive equality/hash · `actual.account()` threads
`allow_neg` (fixture: stochastic 2-node TPM where Σα differs 1.1206 vs
1.4474) · `dynamics.simulate()` k-ary initial states + `seed` argument on
`simulate`/`mean_dynamics`.

Full suite after all fixes: **4312 passed, 290 skipped, 0 failed** (the two
baseline failures did not appear in this run).

## Still open (unchanged by this branch)

- F15 towncrier (maintainer's call: hand-written 2.0.0 section vs fragments).
- R5/R6 example fixtures (fixing `fig5b_substrate` moves the golden zoo — needs
  the maintainer to bless golden regeneration; `differentiation_macro_tpm` is a
  one-liner but is grouped with it as a published-numbers change).
- R7 AC partition-tie None return (fix is mechanical — dedupe candidates by
  `lex_key` before the cascade — but changes published accounts; grouped with
  the wrong-value tier).
- R8/R4 sharded merges, R9 sweep preset substitution, R10 AC cause-side
  background, F9/F12 tie precedence, F10 `ces_measure` validation, F8
  `_fingerprint`, F6 `Complex.__hash__` — as documented above.

---

# Second fix wave (2026-08-17, worktree on the audit branch)

Directed item by item; every fix carries a regression test verified where
noted.

- **Changelog staging (F15)** — RESOLVED without discarding either artifact:
  the curated 2.0.0 section moved to `RELEASE-NOTES-2.0.0.md` (release-day
  merge steps in its header), CHANGELOG.md keeps the marker and released
  history, `just changelog-draft` renders pending fragments. The real build
  path (`towncrier build --version 2.0.0 --keep`) verified working, 40
  fragments intact.
- **R9 sweep preset substitution** — FIXED: `formalisms=None` is now a
  no-preset sentinel; ambient config honored exactly as `analyze` honors it;
  table reports the active version name. Regression test asserts
  sweep == direct compute on a fixture where the customization moves Φ 2.13×.
- **F10/F8/F6 consistency trio** — FIXED: `ces_measure` validated against
  formalism-declared `compatible_ces_measures` (IIT 3.0 keeps the published
  Gómez SUM_SMALL_PHI configuration — the raw finding's "append to
  compatible_measures" fix would have broken that paper reproduction);
  `_fingerprint` digests factor shapes (F8; test verified to fail reverted);
  `Complex.__eq__` includes `node_indices` (F6, the macro-footprint reading).
- **R5/R6 example fixtures** — FIXED: fig5b B = AND(A, C) per the 2014 paper
  (golden regenerated; still exactly one distinction; panel values ci=0,
  ei=0.5 reproduce); `differentiation_macro_tpm` = p² + 2pε/3. Gate-level
  tests added; test-module misattribution of fig4/5a/5b to the 4.0 paper
  corrected.
- **R7 AC partition ties** — FIXED both ways per direction: (a)
  `JOINT_PARTITION_ALL` yields each induced cut once (structural rule proven
  equivalent to lex-key dedupe on 9 sizes; O(1) memory; counts regenerated
  from a closed form verified on 14 sizes; perf pins regenerated — every
  count decreased); (b) the AC cascade resolves identical-cut survivors
  instead of returning None (verified to fail reverted). ~15% fewer
  repertoire computations on the rule110 phi-structure frame.
- **R4 + R8 sharded merges** — RE-ENGINEERED: strides report every
  specified-state candidate's local minimum (one entry per pin at the
  distinction level, per (cause, effect) pair at the SIA level); the merge
  takes the cross-stride min per candidate, then runs the same selection the
  unsharded path runs (`_resolve_pair_sias`, extracted for reuse). Found in
  the process: under IIT 4.0 (2026) the intrinsic-information cap applied
  inside strides destroys the uncapped normalized-φ ordering the MIP
  selection needs — strides now search uncapped and the merge caps once the
  global MIP per pair is chosen. Guard tests parametrized over the exact
  divergence cases (rule110/xor, k∈{2,3}); campaign end-to-end
  sharded==unsharded green under both 4.0 presets.
- **R10 AC cause-side background** — ARCHEOLOGY DONE, no behavior change:
  the after-state convention was deliberately confirmed by Larissa
  (2026-05-29) and locked; but the freeze-vs-posterior-weight axis was not
  the question she was asked, the locked test cannot discriminate it, and
  the paper's own examples cannot either. The false docstring is fixed; a
  precise follow-up question (with the discriminating 3.0-vs-1.2345 fixture)
  is appended to `ignore/planning/AC_CAUSE_WEIGHTING_QUESTION.md`.
- **Dependabot** — CLEARED: all 14 alerts were transitive optional deps
  (12 × Pillow < 12.3.0 via visualize; cryptography < 50 via the mcp
  extra's pyjwt). Lockfile bumped to pillow 12.3.0 / cryptography 50.0.0.

Still open: F12/F9 tie precedence (theory decision), P15 docs sweep.

## F12/F9 — RESOLVED (2026-08-18): the S1 cascade form

The maintainer's call landed on neither of the two candidate repairs alone:
`resolve_distinction_tie` is now a true postulate cascade over the complete
tied-reading set (union of every purview-tied MICE's state ties). Congruence
is a requirement (filter; exclusion on empty), Composition selects the
largest congruent purview, Determinism (lexicographic purview) closes
residual ties — so the combined state-tie-plus-purview-tie case is also
covered, which the minimal `len(state_ties) > 1` guard would not have been.
`basic` gives Φ = 1.125 (matching the three independent routes) and Φ is
relabeling-invariant. Regenerated: 6 golden-zoo fixtures, phi_structure and
relations goldens for basic/fig4, and five value pins. Two theory-relevant
consequences surfaced: pqr's resolved CES now carries one relation (the
measured Σφ_r bound no longer collapses to 0 there: 0.1875 vs 0.125), and
the "inert_unit" CES-incompleteness counterexample dissolved — the larger
congruent purview absorbs a frozen unit, whose state then distinguishes the
substrates; the exhaustive n=2 re-search finds no surviving
different-constant pair with a nonempty CES.

## R10 — RESOLVED (2026-08-18): clamped background in both directions

The cause direction now fixes background inputs at the observed
before-state (`System.background_state`, pinned by `TransitionSystem`),
matching the effect direction and the paper's U = u causal model. Fig 8B
reproduces at 3.0 bits on the informative-background fixture (now a
paper-reproduction test); `noise_background` reaches the cause side; the
May-locked test was rewritten (it discriminated only after-vs-before, not
fix-vs-integrate); the IIT 3.0 two-OR cause ratio returns to 0, which the
test's own history records as the value its log2(4/3) assertion replaced.
Confirmation question for Larissa remains drafted in the question doc.

## F12/F9 follow-up (2026-08-18): purview-size proxy replaced by the exact S1 rule

The S1 supplement states the distinction-tie rule as "selecting the
[reading] that maximizes the system's structure integrated information Φ",
with congruence as its consequence and larger purviews only as what that
"typically favors". The cascade's purview-size level was therefore a proxy
standing where the sibling resolvers (complex ties, system-state ties)
already compute actual Φ. Resolution now happens jointly in
`Distinctions.resolve_congruence`: congruence filters each distinction's
readings, the Φ-maximal combination is selected over the product of
multi-reading distinctions (analytical Σφ_r; exact up to 4096 combinations,
greedy beyond with a warning), Determinism closes residual Φ-ties. Zoo
values unchanged (proxy and rule agree on every curated fixture — verified
by exhaustive combination enumeration); on random 3-node substrates the
proxy understated Φ in 145 swept cases (up to ~13%), two of which are now
pinned as regression tests. The temporal directed-bipartition schemes were
retired in the same wave (direction was read by no evaluation path;
verified statically and on 28 mirrored pairs).

# Third fix wave (2026-08-19, worktree on the audit branch)

Step-1 decision (user, an author of the paper): the Fig 6D / 7B published-figure
pins are superseded by the S1-faithful values. The published values embed the old
enumeration-order tie resolution (relabeling-dependent, F9/F12) and are
sub-maximal under the S1 supplement's own rule; phi_s and the distinction counts
match the figures exactly. New pins: Fig 6D Phi = 12395 (published 11452);
Fig 7B n_r = 13498, Phi = 19.32 (published 13111, 18.55). A note to Larissa about
the deviation remains to be drafted/sent.

Confirmed-defect fixes, each with a regression test that fails on revert:

- **numpy_aware_eq mapping bug** (`pyphi/models/cmp.py`): dicts compared by
  zipping keys positionally — same keys / different values compared equal.
  Mappings now compare by key set with values compared recursively.
- **TransitionSystem.save() data loss** (`pyphi/actual.py`, `pyphi/serialize/`):
  save was delegated to the underlying System, so load returned a System.
  TransitionSystem now has its own schema and round-trips faithfully.
- **pyphi_config_3.0.yml** unloadable (retired `parallel_concept_evaluation`
  field) and drifted from the preset. Rewritten as an exact mirror of
  `presets.by_name["IIT_3_0"]`; a test loads the file and asserts layer equality.
- **F11** (`analyze()` AttributeError 'normalized_phi' on a validator-approved
  partial IIT 3.0 config): formalisms now declare
  `compatible_sia_tie_strategies`; incompatible SIA tie strategies are rejected
  with a ConfigurationError both eagerly (override/load_yaml constraint) and at
  the IIT3 dispatch boundary (catches per-field-assignment configs). Two test
  files were themselves using the forbidden partial-pin pattern and were caught
  by the new constraint; both now pin the complete preset.
- **Realization enforcement vs the disjunction_conjunction example — already
  resolved on this branch**: the current check enforces exactly the paper's
  Realization axiom, p(after | before) > 0, and no longer requires the
  before-state itself to be reachable (that stricter subsystem-reachability
  check was removed with the AC background rework; see the comment in
  `TransitionSystem.__post_init__`). Verified with both controls: the paper's
  own transitions construct and account (including an unreachable before-state),
  and an impossible transition still raises TransitionUnreachableError. A
  regression test pins this.

Still open from step 2: `config.override()` process-globality (user deciding
between a contextvars fix and a documented limitation).

## config.override() process-globality — RESOLVED (2026-08-19): documented + internal fix

User decision: document the limitation rather than adopt contextvars pre-release
(threaded multi-configuration use is a small slice of users; contextvars would
have to reroute the hottest config read path and explicitly propagate context
into thread-backend workers). The override() docstring and docs/howto/configure.md
now warn that overrides apply process-wide. The one internal misuse is fixed:
macro grain search opened override(parallel=False) inside each parallel worker
(racy on the thread backend); the override is now a single parent-side scope
around the map_reduce dispatch, inherited by process workers via the config
snapshot and read directly by thread workers. Thread-backend regression test
added. Contextvars remains a 2.1 candidate.

# Verification-tier sweep of the high findings — COMPLETE (2026-08-19)

All 123 high-severity raw findings adjudicated. After removing those already
resolved by this branch's earlier work, duplicates, and pure-docs items
(deferred to the step-4 docs sweep), they deduplicated to ~25 distinct claims,
each verified with an executed repro by a dedicated agent. Verdicts: 2 already
fixed on this branch (Relation/RelationFace phi-ordering MRO; noise_background
cause side), 1 refuted in part (Complex serialization claim's dumps half),
1 needs theory adjudication (below), and the rest CONFIRMED and now FIXED, each
with a regression test verified to fail against the unfixed code:

- INTRINSIC_INFORMATION composite: cause-side Bayes normalization (Eq. 11) and
  operand-rank broadcast (wrong ii values, wrong-length specified states).
- Cache layer: @cache() hit-path KeyError race; ByteBoundedStore eviction
  RuntimeError under concurrent hits; clear_all() bypassing store accounting
  (phantom occupancy + permanently dead cache after a latched budget); cgroup
  ancestor limits ignored (Slurm/systemd layouts).
- Relations: RelationFace unpicklable (one repr()/num_faces() call made a whole
  CES unpicklable); NullRelations identity equality (IIT 3.0 CES round-trips
  never compared equal).
- Serialization: results' config degraded to a plain dict on load (diff()
  AttributeError, IIT 3.0 Phi relabeled as phi_s, rerun recipe broken) — now
  rehydrated as a real ConfigSnapshot, every preset field round-trips exactly;
  serializer-registry registration race; MICE purview-tie tuples gaining a
  spurious member on round-trip.
- Model identity: Distinction eq/hash now include the specified purview states
  (structures with different Phi no longer compare equal); MacroSystem no
  longer compares equal to a plain System over its macro substrate. The
  superseded per-distinction resolve_congruence (dead since the joint-Phi
  rework, with a stale import) was removed.
- SIA machinery: single-unit ZeroDivisionError under DIRECTED_BIPARTITION_CUT_ONE;
  EDGE_CUT_BIDIRECTIONAL skipping the Eq. 14 disconnection filter (phi_s = 0
  reported for an irreducible system); single-direction sia() crash on state
  ties; sia.ties inflated with non-tied readings; null-MICE display crashes;
  nondeterministic MIP under shortcircuit_sia=False.
- Config validation: specification_measure validated (eager + dispatch);
  background_conditioning constrained per formalism (IIT 3.0 diagnostic
  fixtures opt out explicitly); version IIT_4_0_2026 now requires a capping
  system measure; partial per-level parallel dicts merge over defaults and
  reject unknown keys; mechanism_partition_scheme gained the last missing
  reactive dispatch guard. The assignment path deliberately remains
  eager-validation-free; every audit-identified silently-wrong field now has a
  dispatch-boundary guard.
- TPM/parallel/timescale: FactoredTPM read-only-view aliasing; row-sum check
  tightened to a purely absolute tolerance; lazy generator consumption below
  the dispatch threshold; thread-backend chunking honored; cross-backend
  parent-pid latch restored after dispatch; run_tpm raises
  ConditionallyDependentError instead of returning wrong multi-step dynamics.

## Open adjudication (theory decisions for the release owner)

1. **CompleteEdgeCut normalization (H075) — RESOLVED (2026-08-24).** The 1/n
   override traced to commit 6f0a800b (2022-08-10, "Simplify & fix general
   system cuts") with no recorded justification. Investigation showed the
   default DIRECTED_SET_PARTITION family already enumerates the strict
   all-singletons directional partition (self-loops intact) with the correct
   Theorem 1 normalization n(n-1) via the general rule, so CompleteEdgeCut's
   unique content is self-loop severance: it represents total unconstraining
   of cause-effect power, is the monad's sole irreducibility test, and is an
   optional MIP candidate otherwise. User decision: the intended meaning of
   "complete" is this total cut; the option stays; normalization now follows
   the uniform severed-connections rule (1/n^2, matching
   num_connections_cut). Monads are unaffected (factor 1 either way), so no
   default-path result changes.
