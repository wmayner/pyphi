Distinction-level normalized φ was not stored on serialization: it was
recomputed from the ambient `distinction_phi_normalization` option at load
time, so a result computed under one formalism and loaded under another
silently changed value (e.g. an IIT 3.0 result reloaded under the 2026
default: 0.5 → 0.1667). The signed normalized φ is now stored in the schema
and restored on load; files written before the field existed keep the old
recompute fallback.
