You are implementing **macro sub-project 2 (intrinsic-unit criteria and
bounded grain search)** for PyPhi 2.0, on the `2.0` branch.

The design is fully settled and committed:
`docs/superpowers/specs/2026-06-11-intrinsic-units-criteria-search-design.md`
— read it FIRST and treat it as the source of truth. Your job is
(1) write a detailed implementation plan, (2) get my approval, (3) commit
the plan, (4) execute it task-by-task with a commit per task.

## Process

If the superpowers skills (brainstorming / writing-plans /
executing-plans) are available in this session, use writing-plans and
executing-plans. If they are NOT available, follow this equivalent
discipline:

- Write the plan to
  `docs/superpowers/plans/<today's date>-intrinsic-units-criteria-search.md`.
- The plan is TDD with bite-sized tasks: each task lists exact file
  paths, the complete failing-test code, the complete implementation
  code, the exact run commands with expected outcomes, and a commit
  step. No placeholders ("add validation", "similar to Task N") — every
  code block complete.
- Self-review the plan against the spec (coverage, placeholder scan,
  type/name consistency across tasks) before presenting it.
- **Present the plan and WAIT for my approval before committing it or
  writing any implementation code.** Same for the final results: report
  honestly, including any failures.
- Execute inline, one task at a time: write the failing test, see it
  fail, implement, see it pass, commit.

## Prep reading (before writing the plan)

- The spec (above) — includes the formalism pins (the f(U^J, W^J)
  interpretation, micro-unit exemption, tie semantics), the full API
  sketch, bounds defaults, error cases, and the six test batteries with
  exact anchor values.
- The SP1 machinery you build on: `pyphi/macro/units.py`,
  `pyphi/macro/tpm.py`, `pyphi/macro/system.py`, and the test files
  `test/test_macro_units.py`, `test/test_macro_tpm.py`,
  `test/test_macro_system.py` (note the shared fixtures `CG_TPM`,
  `_bbx_micro_tpm`, `_asymmetric_substrate` — reuse via
  `from test.test_macro_tpm import ...`; `test/__init__.py` exists).
- The SP1 spec for background:
  `docs/superpowers/specs/2026-06-10-intrinsic-units-machinery-design.md`.
- The paper (tracked in the repo):
  `papers/2024__marshall-et-al__intrinsic-units.pdf`, Sec 2.2.2
  (Eqs 15-19, pp. 6-7) and the recursion paragraph (p. 9).

## Facts already established (do not re-derive)

- `config.override(**presets.iit4_2023)` (from `pyphi.conf`) reproduces
  the paper authors' configuration bit-for-bit; all acceptance tests run
  under it. SP1's tests show the pattern.
- `MacroSystem` is hashable (memo key) and `MacroSystem.from_micro`
  reproduces plain `System` results exactly under identity macroing.
- Unit validity (Eqs 15-16) is independent of the candidate's own
  mapping and update grain — it is a property of (V^J, W^J).
- Micro units are axiomatically valid; Eqs 15-16 gate macroing only.
- All inequalities are strict at `config.numerics.precision`; use the
  existing comparison helpers (`pyphi.utils.eq` etc.), not raw `>`.
- The test substrates and their committed anchor values:
  - **Dancing couples (sfn/sfnn/sfs):** 4 units (A,B,C,D); per unit,
    P(ON next) = 0.05 + 0.05*self + 0.6*horizontal + w_v*vertical,
    with neighbor wiring horizontal/vertical per unit index:
    0 -> h=1, v=2; 1 -> h=0, v=3; 2 -> h=3, v=0; 3 -> h=2, v=1.
    State (0,0,0,0). w_v = 0.0 (sfn), 0.01 (sfnn), 0.25 (sfs).
    Anchors (committed by the authors, exact):
    sfn phi_s(A)=0.02363345634846179, phi_s(AC)=0.0;
    sfnn phi_s(A)=0.023640988356789627, phi_s(AC)=0.004863714555961354;
    sfs phi_s(A)=0.02346371771182276, phi_s(AC)=0.16758555077361778,
    phi_s(AB)=0.6728123807299448.
  - **min:** 2 units, state-by-node TPM rows (little-endian states
    00,10,01,11): [0.05,0.05],[0.05,0.06],[0.06,0.05],[0.95,0.95];
    state (0,0). Anchors: phi_s(A)=phi_s(B)=0.0,
    phi_s(AB)=0.005106576483955726; the both-on coarse-grain macro
    unit's one-unit system has phi_s = 0.7883339770634886.
  - **bu:** 3 units, deterministic state-by-node TPM rows (little-endian
    states 000,100,010,110,001,101,011,111):
    [1,1,1],[0,1,0],[0,0,0],[1,1,0],[0,0,1],[0,1,1],[1,0,1],[1,0,0];
    state (0,0,0). Anchors: every 1- and 2-unit subsystem has
    phi_s = 0.0; phi_s(ABC) = 0.8300749985576875.
  - Where the spec says a value is "recorded as a golden at
    implementation time" (min argmax mapping, bu driver verdict),
    compute it during execution, sanity-check it (e.g. the min macro
    argmax should be >= 0.7883339770634886), hard-code it into the
    test with a comment, and CALL IT OUT in your final report.
- The `slow` pytest marker requires the `--slow` flag to run
  (`uv run pytest test/<file> --slow -m slow`).

## Standing constraints (verbatim-in-force)

- NEVER push. NEVER use `--no-verify`. Commit via
  `git -c commit.gpgsign=false commit`.
- The pre-commit hook often reformats files: if a commit prints hook
  output but no commit line, re-`git add` the same files and commit
  again (a failed commit leaves files staged — check `git status`
  before the next add+commit). If `ruff check` FAILS in the hook, read
  the full output and fix the violations; do not retry blindly.
- Targeted `git add <files>` only — the repo root has many untracked
  scratch files; never `git add -A` from the root.
- Ruff rules that have bitten before: no `dict()` calls; no Unicode
  math characters in Python strings or docstrings (no ∪ ×  − – —
  write "union", "x", "-"; Greek letters only in .md files); all
  imports at top of file, tests included (E402/PLC0415), added
  per-task to avoid F401 at commit gates; RUF005 (use `(*tup, x)` not
  `tup + (x,)`); SIM117 (combine nested `with`); RUF043 (regex
  metacharacters in `pytest.raises(match=...)` need raw strings);
  ARG002/ARG003 (no unused method args — drop them or `# noqa`).
- `uv run` for every command. Full verification at the end =
  `uv run pytest` with NO path argument (that includes the doctest
  sweep; pre-deletion baseline today: 1954 passed, 20 skipped,
  1 xfailed, plus 13 slow under `--slow`).
- pyphi states are LITTLE-ENDIAN (`utils.all_states`: first unit
  varies fastest).
- New user-facing work needs a changelog fragment
  (`changelog.d/intrinsic-units-search.feature.md`, content sketched in
  the spec's Files section) and the ROADMAP item-10 SP2 sub-entry
  marked landed (follow the SP1 sub-entry's style).
- Docstrings describe the final state — no design-narrative, no
  migration history, no references to plans/sub-projects/phases.

## Environment setup (container)

If the venv is missing: `uv venv && uv pip install -e ".[dev]"`. Run a
quick `uv run pytest test/test_macro_system.py -q` to confirm the SP1
baseline is green before starting.
