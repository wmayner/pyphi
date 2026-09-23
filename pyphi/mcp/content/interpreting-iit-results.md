# Interpreting an IIT result

How to read an analysis result and explain it in plain language.

## What `analyze` gives back

- `summary.system_phi` — **φₛ**, system integrated information. Positive means
  the system is irreducible: it exists as one integrated whole. 0 means it does
  not: either some partition makes no difference (the system is reducible), or
  the system provides itself no repertoire of alternatives (ii(s) = 0; see
  `requirement_binding` below). `summary.cause_phi` and `summary.effect_phi`
  are the integration on the cause and effect sides; φₛ is the smallest of
  those two and ii(s).
- `summary.mip` — the minimum partition, the system's weakest link.
- `summary.big_phi` — **Φ**, the total structure integrated information (the sum
  over distinctions and relations). It is a different quantity
  from `system_phi`; do not report one for the other. If you find yourself
  reporting the same number for both, you have read the wrong field.
- `summary.formalism` — the version of IIT that produced these numbers,
  recorded for reproducibility.
- `summary.num_distinctions` / `summary.num_relations` and their φ sums —
  the size and weight of the Φ-structure.
- `summary.intrinsic_information` — **ii(s)**, the third term of φₛ.
  `summary.requirement_binding` gives the term and direction when ii(s) is
  what set φₛ (both φ_c and φ_e positive with φₛ = 0 is the usual sign), and
  is `None` when integration set it.
- Purview and mechanism labels are written in the specified *state*:
  uppercase is ON, lowercase is OFF, and a subscript gives the state of a
  unit with more than two states (`A₂`).
- Under the default analytical backend, relation counts and Σφ_r are
  closed-form; individual relations are not enumerated, so `inspect` cannot
  list them. Say so rather than inventing them.
- `card` — a ready-made human-readable summary.

## What `compute="distinctions"` gives back

`analyze(..., compute="distinctions")` returns the distinctions on their own,
skipping the system-partition search that a cause-effect structure runs
before it unfolds anything. Over a sparse substrate that search is most of
the running time, so this is the cheap way to see which mechanisms specify what
when you do not need φₛ, Φ, or the relations.

Read `summary.congruence` first:

- `"resolved"` — the system's specified state was untied, so these are exactly
  the distinctions the Φ-structure has. `num_distinctions` and
  `sum_phi_distinctions` mean what they usually mean.
- `"unresolved"` — the specified state ties, and the tie is broken by the φₛ
  cascade over the tied cause/effect pairs, which needs the search that was
  skipped. The counts come back as `num_distinctions_upper_bound` and
  `sum_phi_distinctions_upper_bound`, because a Φ-structure keeps only the
  distinctions congruent with the system's specified state and that filter can
  remove any number of them, including all. Report these as upper bounds, or
  rerun with `compute="ces"` for the actual set.

## How to narrate it

1. **Lead with the two headline numbers, and keep them distinct.** "This
   system has φₛ ≈ 0.21, so it is irreducible: it exists as one integrated
   whole. Its cause–effect structure has structure integrated information
   Φ ≈ 1.86." The system is a complex, and its cause–effect structure a
   Φ-structure, only if its φₛ is also maximal among the overlapping candidate
   systems. `analyze` does not check this; in Python, `substrate.complexes(state)`
   does. Do not call Φ "the phi value" without saying which one.
2. **Say what the weakest link is.** The minimum partition is where
   the system is closest to falling apart into independent parts. Name it.
3. **Describe the structure concretely.** Use `inspect(result_ref, "ces.distinctions[0]")`
   to read a distinction: which mechanism, which cause and effect purviews,
   which states, what φ_d. Relations bind distinctions that specify the same
   units in the same state.
4. **Name the caveats.** If the substrate is small and deterministic, mention
   that ties may make the structure non-unique (see gotchas). If the result
   was computed under an earlier version of IIT, say so.

## Common misreadings to avoid

- Reporting Φ when asked about φₛ, or vice versa.
- Reading φₛ = 0 as "nothing interesting" rather than "does not exist as one
  whole" (reducible, or ii(s) = 0).
- Treating a small-network result as canonical when a symmetric TPM may have
  produced a tie.
- Comparing φ values computed under different versions of IIT as if they were
  the same quantity. They are defined differently.
- Quoting a `_upper_bound` count from `compute="distinctions"` as the number of
  distinctions the Φ-structure has. Congruence filtering has not run yet.
