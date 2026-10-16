# Metamorphic and Differential Testing Round Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the testing round the spec describes: build the shared fixture bank, brief three checking agents and their verifiers, launch them, and triage what they report.

**Architecture:** Everything before the triage lives in a `round5/` folder in the session scratchpad, outside the repository. A wheel built from `main` is installed into one shared venv. `fixtures.py` builds substrates (with its own numpy permutation, independent of PyPhi's helpers), `screen.py` writes the screened `bank.json`, and a Workflow script runs the three Opus agents, each followed by a Sonnet verifier. Repository changes (fixes and the tests kept from the checks) come after the triage, under their own approval.

**Tech Stack:** Python 3.13, numpy, pytest, the pyphi wheel with all extras, the Workflow tool.

**Spec:** `docs/superpowers/specs/2026-10-15-metamorphic-testing-round-design.md`

## Global Constraints

- Release bar, comparison tolerance, tie rules and the known non-invariances are exactly as in the spec; briefs quote them, not paraphrase them.
- Values agree when `abs(a - b) <= 1e-10 * max(1, abs(a), abs(b))`; save/load round trips must be exact.
- Randomness only through `np.random.default_rng(seed)`; every bank entry records its seed and generator parameters.
- The repository is read-only for every agent; scratch paths do not go into any committed file.
- Fixture sizes: 3 and 4 units; 5 units only where a brief says so.
- Formalism names: `IIT_4_0_2026`, `IIT_4_0_2023`, `IIT_3_0` (presets `pyphi.iit4_2026`, `pyphi.iit4_2023`, `pyphi.iit3`).

## Review Focus

- A permutation helper that is wrong in the little-endian direction would make every A1/B2/C2/C3 check fail at once and look like a library bug: Task 2 differentials it against `FactoredTPM.permute_nodes` and an identity permutation.
- A k-ary generator whose factors do not sum to 1 over the last axis, or that depend on an omitted edge, is rejected by `Substrate`: Task 2 builds 20 seeds of each kind as a test.
- Screening that marks an entry non-trivial on φ_s but not Φ would let structure checks pass vacuously: Task 3 records φ_s and Φ separately and the briefs say which one a relation needs.
- A tie flag computed only from the SIA misses tied distinctions: Task 3 records `effectively_tied` from the SIA and from every distinction.
- Agents that each rebuild the venv waste time and may install a different build: Task 1 records the wheel's hash, and Task 4 has verifiers check it.

---

### Task 1: Wheel and shared venv

**Files:**
- Create: `<scratchpad>/round5/dist/` (wheel), `<scratchpad>/round5/.venv/`, `<scratchpad>/round5/BUILD.txt`

**Interfaces:**
- Produces: `round5/.venv/bin/python` with `pyphi[visualize,emd,mcp,parquet,xarray,cluster]` installed from the wheel, plus pytest; `BUILD.txt` holding the commit hash and the wheel's sha256.

- [ ] **Step 1:** `uv build --wheel -o <scratchpad>/round5/dist` from the repository root (writes only to the scratchpad).
- [ ] **Step 2:** `uv venv -p 3.13 .venv` in `round5/`, then `uv pip install --python .venv/bin/python "pyphi[...] @ file://<wheel>" pytest`.
- [ ] **Step 3:** Verify: `.venv/bin/python -c "import pyphi; print(pyphi.__version__)"` prints a dev version on the recorded commit; write `BUILD.txt`.

### Task 2: `fixtures.py`

**Files:**
- Create: `<scratchpad>/round5/fixtures.py`, `<scratchpad>/round5/test_fixtures.py`

**Interfaces:**
- Produces:
  - `random_substrate(seed: int, n: int, *, p_edge: float = 0.6, alphabet: tuple[int, ...] | None = None, labels: bool = False, extreme: float = 0.0) -> pyphi.Substrate` — each unit's next-state distribution is drawn from a table keyed by the states of its inputs in the connectivity matrix; binary substrates are built from a state-by-node TPM, k-ary ones as `Substrate(marginals=..., state_space=..., cm=..., node_labels=...)` (the form `pyphi.examples.gomez_p53_mdm2_substrate` uses). `extreme` is the fraction of table entries drawn from `[0, 0.05] ∪ [0.95, 1]`.
  - `permute_substrate(substrate, p: tuple[int, ...]) -> pyphi.Substrate` — new unit `i` is old unit `p[i]`; permutes factors' input axes, the connectivity matrix (`cm[np.ix_(p, p)]`), the state space and labels, using numpy only.
  - `permute_state(state, p)`, `permute_units(units, p) -> tuple[int, ...]` (maps old indices to new, sorted), `inverse(p)`.
  - `close(a, b) -> bool` — the spec's tolerance; `None == None` is close.
  - `named() -> dict[str, tuple[Substrate, tuple[int, ...]]]` — the spec's named examples, plus `and_xor`, `xor_and`, `dual_and_xor`, `symmetric_triple` and `noisy_or_background` copied from `test/example_substrates.py`.

- [ ] **Step 1: Write the tests** in `test_fixtures.py`:
  - `test_random_substrates_construct`: 20 seeds each of binary n=3, n=4, k-ary `alphabet=(3, 2, 2)` and `(2, 3, 2, 2)`, labelled; all construct without error.
  - `test_permutation_matches_permute_nodes`: for 10 seeds and a fixed non-identity `p`, `permute_substrate(s, p)`'s factors equal `s.factored_tpm.permute_nodes(p)`'s factors, and its connectivity matrix equals `s.cm[np.ix_(p, p)]`.
  - `test_identity_permutation_is_noop` and `test_permutation_round_trip` (`permute_substrate(permute_substrate(s, p), inverse(p))` equals `s`).
  - `test_permuted_phi_matches`: on `named()["iit4_2023_fig1a"]` under `pyphi.iit4_2026`, φ_s of the permuted substrate in the permuted state equals the original's (0.04049680811800574) within `close`.
- [ ] **Step 2:** Run `.venv/bin/python -m pytest test_fixtures.py -q`; expect failures (module missing).
- [ ] **Step 3:** Implement `fixtures.py` to the interfaces above.
- [ ] **Step 4:** Run the tests; expect all to pass. If `test_permutation_matches_permute_nodes` fails, check the direction convention of both before trusting either.

### Task 3: `screen.py` and `bank.json`

**Files:**
- Create: `<scratchpad>/round5/screen.py`, `<scratchpad>/round5/bank.json`

**Interfaces:**
- Consumes: `random_substrate`, `named` from Task 2.
- Produces: `bank.json`, a list of entries `{id, kind: "binary"|"kary"|"named", params: {seed, n, p_edge, alphabet, labels, extreme} | {name}, state, subset, results: {formalism: {phi_s, big_phi, sia_tied, any_distinction_tied, seconds}}}`, and `load_bank() -> list[dict]` plus `build(entry) -> (Substrate, state, subset)` in `fixtures.py`.

- [ ] **Step 1:** Implement `screen.py`: iterate seeds from 0 with `extreme` in `{0.0, 0.3}`, analyze each candidate (whole system, and for 4-unit binary candidates also the subset of units 0–2) under the three formalisms with a 20-second budget per analysis, and keep entries until the spec's targets are met: ≥20 non-trivial binary 3-unit and ≥15 4-unit per formalism, ≥8 k-ary, ≥5 with a tie, ≥5 subset systems. Non-trivial means φ_s > 1e-10 under that formalism. Write raw results for every candidate tried (kept or not) to `screen_raw.json`.
- [ ] **Step 2:** Run `.venv/bin/python screen.py` in the background; check that `bank.json` meets each target and print the counts. If a target is unmet after 400 seeds, record the shortfall in `bank.json`'s header and in the briefs rather than lowering the bar silently.

### Task 4: Briefs and workflow script

**Files:**
- Create: `<scratchpad>/round5/brief-common.md`, `brief-A.md`, `brief-B.md`, `brief-C.md`; the Workflow script (passed inline; the tool saves it).

**Interfaces:**
- Consumes: `BUILD.txt`, `fixtures.py`, `bank.json`.
- Produces: the workflow, returning per agent `{area, findings: [...], coverage: [...], verdicts: [...]}` with the spec's output schema.

- [ ] **Step 1:** `brief-common.md`: thanks and the why (the spec's Purpose in four sentences); ground rules (repository read-only, own scratch folder `round5/<area>/`, the shared venv, the two environment variables); the release bar, comparison rules, tie table and known non-invariances quoted from the spec; how to use `fixtures.py` and `bank.json`; checks written as one pytest file per relation; the 45-minute time box and stopping rule; permission to report that a relation could not be checked or is wrong.
- [ ] **Step 2:** `brief-A.md`, `brief-B.md`, `brief-C.md`: the area's catalogue entries copied from the spec in priority order, each with what is already tested (from the spec) and which bank entries it needs (φ_s or Φ non-trivial; binary, k-ary, tied or subset).
- [ ] **Step 3:** The workflow: `pipeline` over the three areas; stage 1 an Opus agent with the common and area briefs and the findings/coverage schema; stage 2 (only if findings exist) a Sonnet verifier that checks `BUILD.txt`'s hash against the installed wheel, reruns each reproduction verbatim, and returns `{id, verdict, meets_bar, evidence, corrected}`.
- [ ] **Step 4:** Read each brief once against the spec's catalogue: every relation of every area appears in exactly one brief.

### Task 5: Launch and triage

- [ ] **Step 1:** Launch the workflow. Watch for agent failures; resume a failed agent once with SendMessage, and record if it fails again.
- [ ] **Step 2:** On completion, save results to `round5/results.json`; deduplicate findings across areas; keep confirmed or partly confirmed findings that meet the bar.
- [ ] **Step 3:** Report to the user: the triage list with a recommendation per finding, the coverage table (relations run, non-trivial cases, not run and why), and the checks proposed for the suite. Fixes and the test conversion wait for the user's decision.
