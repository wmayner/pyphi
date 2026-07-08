# Theory narrative (IIT 4.0) — design

The third sub-project of the documentation overhaul
(`docs/superpowers/specs/2026-07-07-documentation-overhaul-design.md` §1, Theory
section). Writes the IIT 4.0 theory narrative — the conceptual content that was
never written for the 2.0 documentation. The section explains what PyPhi
computes and why, mapping every quantity of IIT 4.0 to the runtime type or
function that implements it, grounded in a single worked example that executes
at build time.

## Approach

- **Computation-driven, postulate-grounded.** The section follows the pipeline
  PyPhi actually computes — substrate → system → system integrated information
  (Φ_s) → distinctions → relations → Φ-structure — and at each step names the
  IIT postulate the step embodies, cites the paper equation, and shows the
  PyPhi call and its real output. This matches how a user works and produces
  the paper-to-code map as a natural byproduct.
- **Self-contained, orient-then-map depth.** The section stands on its own: it
  gives enough conceptual explanation for a reader who is doing IIT but not
  steeped in the 4.0 details to understand what each quantity means and why,
  then concentrates its depth on the mapping to code and on running it. It does
  not re-derive the paper's ~40 pages; full derivations are left to the paper,
  which is cited throughout. The reader is also pointed to the IIT wiki
  (<https://iit.wiki>) as an additional orientation resource, but the section
  never depends on it to be understood.
- **One executable example throughout.** A single substrate —
  `pyphi.examples.basic_substrate()`, the three-node example system from the
  IIT 4.0 paper — threads pages 1–5. Each concept shows the actual PyPhi call
  and its build-time-executed output (the Φ_s value, one distinction, one
  relation, the assembled Φ-structure). Because the pages execute (myst-nb),
  the narrative cannot drift from the code. Every example runs well under the
  minutes-per-page build budget.

## Pages

Eight pages under `docs/theory/`, in this order:

1. **Overview — what IIT 4.0 computes.** The `Substrate → System → formalism →
   Φ-structure` layering (this is the architecture guide), the six postulates
   (existence, intrinsicality, information, integration, exclusion,
   composition) named and one-lined, and the worked example introduced. The
   through-line for the rest of the section.
2. **Substrate and system.** The causal model (the transition probability
   matrix), a candidate `System` in a state; the existence and intrinsicality
   postulates; causal marginalization (Albantakis et al. 2023, Eq. 4). Maps to
   `Substrate` and `System`.
3. **System integrated information (Φ_s).** The integration and exclusion
   postulates at the system level: system partitions, the minimum-information
   partition, Φ_s, and finding the complex. Maps to `pyphi.analyze` and the
   system irreducibility analysis.
4. **Distinctions and relations.** The composition postulate at the mechanism
   level: cause and effect repertoires, intrinsic information, the maximally
   irreducible cause–effect distinction (φ_d), then the relations between
   distinction purviews (φ_r). Maps to the distinction and relation types.
5. **The Φ-structure and the paper-to-code map.** Assembles distinctions and
   relations into the Φ-structure, capped by a reference table mapping every
   named quantity in the paper to the runtime type or function that implements
   it — the "mathematician's acceptance test" as a page.
6. **Formalism versions.** IIT 4.0 (2023) vs the 2026 intrinsic-information cap
   vs IIT 3.0 vs actual causation, and how configuration selects among them.
7. **Conditional independence.** Ported from
   `docs/examples/conditional_independence.rst` — the causal-model assumption
   the framework rests on.
8. **IIT 3.0 overview.** Brief, citing Oizumi et al. 2014 for depth.

## Accuracy

Enforced with the rigor of the docstring sweep, because a plausible-but-wrong
theory page is worse than a missing one:

- Every equation, section, theorem, or figure citation is verified against the
  actual paper PDF in `papers/` — never cited from memory. Primary sources:
  `2023__albantakis-et-al__iit-4.0.pdf` and its supplements S1–S4;
  `2026__mayner-et-al__intrinsic-cause-effect-power.pdf` (Mayner, Marshall &
  Tononi, Entropy 28) for the 2026 intrinsic-information cap;
  `2014__oizumi-et-al__iit-3.0.pdf` for the IIT 3.0 page;
  `2019__albantakis-et-al__what-caused-what.pdf` for the actual-causation
  mention.
- Every code claim (a type, a function, an output value, an equation-to-symbol
  mapping) is verified against the implementation, not asserted from the paper
  alone. `graphify-out/bridge-edges.json` provides the concept-to-code starting
  points for the paper-to-code map.
- Where the 2023 paper and the 2026 intrinsic-information cap differ, both are
  cited explicitly rather than conflated.
- Each page passes an independent verification stage — a second reader checking
  the page's claims against the paper and the code, and confirming the executed
  output matches what the prose says — before it is accepted, mirroring the
  two-stage rewrite/verify process of the docstring sweep.

## Success criteria

- All eight pages populate the Theory section and are wired into the site
  navigation.
- The `-W` docs build stays green with every code cell executed; no page
  introduces a build warning.
- Every equation citation is paper-verified and every code claim is
  implementation-verified; each page passed its verification stage with no
  unresolved findings.
- The paper-to-code map covers every named (Greek-letter) quantity in
  Albantakis et al. 2023 and the 2026 cap.
- The carried-through example is consistent across pages: the same substrate,
  the same state, values that agree from one page to the next.
- The reader is pointed to the IIT wiki as an additional resource, and the
  section is fully understandable without it.
