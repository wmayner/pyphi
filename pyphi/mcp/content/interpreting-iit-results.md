# Interpreting an IIT result

How to read an analysis result and explain it in plain language.

## What `analyze` gives back

- `summary.system_phi` — **φₛ**, system integrated information. Positive means
  the system exists as one integrated whole; 0 means it is reducible.
  `summary.cause_phi` and `summary.effect_phi` are the two sides; φₛ is the
  smaller of them. Under IIT 3.0 this value is that formalism's Φ.
- `summary.mip` — the minimum information partition, the system's weakest link.
- `summary.big_phi` — **Φ**, the total structure integrated information (the sum
  over distinctions and relations). IIT 4.0 only, and never equal to
  `system_phi` by definition — if you find yourself reporting the same number
  for both, you have read the wrong field.
- `summary.formalism` — which version produced these numbers.
- `summary.num_distinctions` / `summary.num_relations` and their φ sums —
  the size and weight of the Φ-structure.
- `card` — a ready-made human-readable summary.

## What `compute="distinctions"` gives back

`analyze(..., compute="distinctions")` returns the distinctions on their own,
skipping the system-partition search that an IIT 4.0 cause-effect structure
runs before it unfolds anything. Over a sparse substrate that search is most of
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

IIT 3.0 has no congruence filter, so it reports no `congruence` key and its
distinctions are the structure as computed.

## How to narrate it

1. **Lead with the two headline numbers, and keep them distinct.** "This
   system has φₛ ≈ 0.21, so it exists as one integrated whole; its experience
   has structure integrated information Φ ≈ 1.86." Do not call Φ "the phi
   value" without saying which one.
2. **Say what the weakest link is.** The minimum information partition is where
   the system is closest to falling apart into independent parts. Name it.
3. **Describe the structure concretely.** Use `inspect(result_ref, "ces.distinctions[0]")`
   to read a distinction: which mechanism, which cause and effect purviews,
   which states, what φ_d. Relations bind distinctions that specify the same
   units in the same state.
4. **Name the caveats.** If the substrate is small and deterministic, mention
   that ties may make the structure non-unique (see gotchas). If a non-default
   formalism was used, say so.

## Common misreadings to avoid

- Reporting Φ when asked about φₛ, or vice versa.
- Reading Φ = 0 as "nothing interesting" rather than "reducible".
- Treating a small-network result as canonical when a symmetric TPM may have
  produced a tie.
- Comparing φ values across formalism versions as if they were the same
  quantity — they are defined differently.
- Quoting a `_upper_bound` count from `compute="distinctions"` as the number of
  distinctions the Φ-structure has. Congruence filtering has not run yet.
