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
