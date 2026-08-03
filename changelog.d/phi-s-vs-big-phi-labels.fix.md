Result cards no longer label φₛ as Φ under IIT 4.0. A `CauseEffectStructure`,
`Analysis`, or `Complex` card now shows `φ_s` for the system irreducibility
value, and the Φ-structure cards additionally show `Φ` — the structure
integrated information, the sum of `Σφ_d` and `Σφ_r` printed beneath it. Under
IIT 3.0, whose system-level value is that formalism's Φ, the label is unchanged.
Added `Analysis.big_phi` (Φ, raising under IIT 3.0) and `big_phi` / `sum_phi_d`
columns to `Analysis.to_pandas()`. The MCP server's result summary drops its
ambiguous `phi` key — a duplicate of `system_phi` — adds a `formalism` key, and
renders the MIP concisely instead of as a full card.
