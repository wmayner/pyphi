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
