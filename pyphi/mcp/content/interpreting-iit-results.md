# Interpreting an IIT result

How to read an analysis result and explain it in plain language.

## What `analyze` gives back

- `summary.phi` / `summary.system_phi` — **φₛ**, system integrated information.
  Positive means the system exists as one integrated whole; 0 means it is
  reducible. `summary.cause_phi` and `summary.effect_phi` are the two sides;
  φₛ is the smaller of them.
- `summary.mip` — the minimum information partition, the system's weakest link.
- `summary.big_phi` — **Φ**, the total structure integrated information (the sum
  over distinctions and relations).
- `summary.num_distinctions` / `summary.num_relations` and their φ sums —
  the size and weight of the Φ-structure.
- `card` — a ready-made human-readable summary.

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
