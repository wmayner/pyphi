# Serialization round-trip completeness — design

## Problem

The serialization layer (`pyphi/serialize/schema.py` + `convert.py`) silently
drops computed fields from most result types: the schemas omit fields the
domain objects carry, and the decoders reconstruct objects whose constructors
then fill the gaps with defaults — or, worse, with freshly fabricated values.
Because the domain equality contracts deliberately exclude diagnostics,
`restored == original` reports `True` throughout, masking every loss. One
related crash makes null actual-causation results unserializable, and the
document-level `format_version` is written but never checked on load.

`IIT4SIASchema` is the one schema that gets all of this right — it carries
`signed_phi`, `reasons`, `config`, `provenance`, and tie peers, with
established helpers (`_enc_reasons`/`_dec_reasons`, `_enc_config`,
`ProvenanceSchema`) and an encode-peers-without-self pattern shared by the
RIA and CausalLink encoders. Every fix below replicates that pattern onto the
schemas that lack it.

### Confirmed findings addressed (whole-library review, 2026-07-13)

1. **`RIASchema` drops `signed_phi`** — the schema stores only the
   `|·|+`-clamped `phi`; the decoder passes no `signed_phi`, so
   `RepertoireIrreducibilityAnalysis.__init__` snapshots it from the clamped
   value. Negative (preventative-cause) integration values round-trip to 0.0,
   destroying `signed_phi` and the derived `signed_normalized_phi`.
2. **`RIASchema` drops `selectivity` and `reasons`** — `selectivity` is set
   on every default-config IIT 4.0 mechanism analysis; null MICE carry
   `NullResultReason` tuples. Both come back `None`.
3. **MICE purview ties are dropped and "not computed" is rewritten as "no
   ties"** — the three MICE schemas carry only `ria` and `purview_margin`;
   `_decode_mice` unconditionally sets `_purview_ties = (instance,)`, so a
   MICE with N tied purviews decodes with `num_purview_ties == 0`, and one
   whose ties were never computed (`None` → NaN) decodes claiming zero ties.
4. **`IIT4SIASchema` and `IIT3SIASchema` drop `runner_up`; `IIT3SIASchema`
   also drops `reasons`** — the main SIA paths set `runner_up` on every
   computed result and `explain()` reports it; round-trip loses the
   runner-up partition and its φ margin.
5. **IIT3 SIA, AcSIA, and CES config/provenance are not serialized** — the
   schemas have no `config`/`provenance` fields, and the domain constructors
   lazily snapshot the *current* global config and capture *fresh*
   `Provenance` when passed `None`. A loaded result silently claims it was
   computed under the loader's formalism, on the loading machine, at load
   time.
6. **AC results lose `node_labels`, `reasons`, and alpha ties** —
   `AcRIASchema` omits `node_labels` and `reasons`; `AcSIASchema` omits
   `reasons` and has no tie-peers field, so labeled AC results come back
   unlabeled, null-result explanations vanish, and `set_ties` records are
   reset to `(self,)`.
7. **`TransitionSchema` drops `noise_background`** — a
   `noise_background=True` transition round-trips as `False` and computes
   different alpha values after reload (executed repro: EFFECT ratio 0.415 →
   0.0). Compounding it, **`Transition.__eq__`/`__hash__` also omit
   `noise_background`**, so behaviorally different transitions compare equal
   and collide under set/dict deduplication — which is exactly why the
   existing round-trip test could not see the loss.
8. **Serializing a null AcSIA crashes** — `_null_ac_sia` constructs
   `account=()`/`partitioned_account=()` as bare tuples although the AcSIA
   contract documents them as `Account`s; `to_schema` has no tuple encoder
   and raises `TypeError`, so reducible/degenerate AC results cannot be
   saved at all.
9. **`format_version` is written but never checked** — `loads()` ignores
   `doc.format_version`, and msgspec's default decoding drops unknown
   fields, so a file written by a future format loads silently with its new
   fields discarded.

## Design

### 1. Schema additions (`pyphi/serialize/schema.py`)

All new fields are optional with defaults, appended after the existing
fields, so files written before this change keep loading (missing fields take
their defaults — the additive-evolution path msgspec is designed for).

| Schema | New fields |
|---|---|
| `RIASchema` | `signed_phi: PhiSchema \| None = None`, `selectivity: float \| None = None`, `reasons: tuple[str, ...] \| None = None` |
| `MICESchema`, `MICECauseSchema`, `MICEEffectSchema` | `purview_tie_peers: tuple[MICEAnySchema, ...] \| None = None` |
| `IIT3SIASchema` | `runner_up: RunnerUpSchema \| None = None`, `reasons: tuple[str, ...] \| None = None`, `config: dict \| None = None`, `provenance: ProvenanceSchema \| None = None` |
| `IIT4SIASchema` (and `NullIIT4SIASchema` via inheritance) | `runner_up: RunnerUpSchema \| None = None` |
| `CESSchema` (and `NullCESSchema` via inheritance) | `config: dict \| None = None`, `provenance: ProvenanceSchema \| None = None` |
| `AcRIASchema` | `node_labels: NodeLabelsSchema \| None = None`, `reasons: tuple[str, ...] \| None = None` |
| `AcSIASchema` | `reasons: tuple[str, ...] \| None = None`, `config: dict \| None = None`, `provenance: ProvenanceSchema \| None = None`, `tie_peers: tuple["AcSIASchema", ...] = ()` |
| `TransitionSchema` | `noise_background: bool = False` |
| **new** `RunnerUpSchema` (tag `"runner_up"`) | `partition: PartitionSchema`, `phi: PhiSchema` |

Notes:

- `signed_normalized_phi` needs no field: `RepertoireIrreducibilityAnalysis`
  derives it in `__init__` from `signed_phi` and the partition's
  normalization factor, so restoring `signed_phi` restores both.
- `config` uses the dict form produced by the existing `_enc_config`
  (msgspec builtins), matching the IIT4 decoder's documented behavior of
  keeping the dict on load.
- `reasons` uses the existing `_enc_reasons`/`_dec_reasons` name-based
  encoding (`NullResultReason` members by name; bare strings pass through).

### 2. Encoder/decoder threading (`pyphi/serialize/convert.py`)

Each new field threads through with the existing helpers. Three structural
pieces:

- **MICE ties are tri-state.** Encode `purview_tie_peers=None` when
  `_purview_ties is None` (never computed); `()` when computed with no ties
  (`_purview_ties == (self,)`); otherwise the peers excluding `self`, each
  encoded with its own tie set suppressed (the `include_peers=False` pattern
  from `_encode_ria`, preventing recursion through the shared tie tuple).
  `_decode_mice` restores accordingly: `None` → `_purview_ties = None`;
  otherwise `(instance, *peers)` shared by reference across all members, as
  `set_purview_ties` does. The current fabricated `(instance,)` is removed.
- **AcSIA ties.** The encoder lambda becomes a `_encode_ac_sia(sia, *,
  include_peers)` function mirroring `_encode_iit3_sia`; the decoder re-links
  the shared tie tuple exactly as the IIT3/IIT4 SIA decoders do.
- **`runner_up`.** Encoded via the new `RunnerUpSchema`; decoded to the
  frozen `RunnerUp` dataclass (`pyphi/models/explanation.py`) and passed to
  the SIA constructors, both of which already accept it.

Everything else is a straight field pass-through: RIA `signed_phi` (raw
value, encoded like the IIT4 SIA's), `selectivity` (plain float), `reasons`;
IIT3 SIA and CES and AcSIA `config=_enc_config(...)` /
`provenance=_enc_optional(...)` on encode and passed through on decode
(suppressing the constructors' lazy fabrication); AcRIA `node_labels` and
`reasons`; Transition `noise_background`.

### 3. Model fixes (root causes)

- **`_null_ac_sia`** (`pyphi/models/actual_causation.py`): construct
  `account=Account(())` and `partitioned_account=Account(())` instead of
  bare tuples. The bare tuple is the type anomaly that crashes `dumps()`;
  `Account` is the documented type, and an empty `Account` is falsy and
  hashes identically to `()`. The existing assertions
  `sia.account == ()` in `test/test_actual.py` flip to asserting an empty
  `Account` (they were pinning the anomaly).
- **`Transition.__eq__`/`__hash__`** (`pyphi/actual.py`): include
  `noise_background` in both. It is a behavioral field — it changes
  background conditioning and therefore alpha — so transitions differing
  only in it must not compare equal or collide in sets/dicts.

### 4. Wire-format policy

`FORMAT_VERSION` stays 1 — this change is purely additive. `loads()` gains
the missing check: if `doc.format_version > FORMAT_VERSION`, raise
`ValueError` naming both the file's version and the library's. Files with
older-or-equal versions load normally.

### 5. Behavior changes for pre-existing files (intentional)

- A pre-fix file containing MICE decodes with `_purview_ties = None`
  (`num_purview_ties` NaN, "not computed") instead of the fabricated
  "no ties" (0). The old file genuinely does not record whether ties were
  computed; NaN is the honest representation.
- Pre-fix IIT3/AC/CES files still fabricate load-time config/provenance —
  there is nothing stored to restore. New files round-trip both faithfully.

## Testing

Field-level round-trip assertions, not `==` — the domain equality contracts
exclude the very fields being fixed, so full-object equality cannot serve as
the oracle.

- **Per-family round-trip tests** in the existing `test/serialize/` modules:
  - RIA: negative `signed_phi` case asserting both `signed_phi` and
    `signed_normalized_phi` survive; `selectivity`; `reasons`.
  - MICE: N-way tie restores `num_purview_ties == N−1` with peer purviews
    intact and the tie tuple shared by reference; `_purview_ties = None`
    round-trips to NaN.
  - IIT3 and IIT4 SIA: `runner_up` partition and φ restored; IIT3 `reasons`.
  - Config/provenance (IIT3 SIA, CES, AcSIA): computed under a pinned
    non-default formalism (`IIT_3_CONFIG` from `test/conftest.py`), loaded
    under the default; assert the loaded config records the pinned version
    and the loaded provenance timestamp equals the saved one.
  - AC: AcRIA `node_labels`/`reasons`; AcSIA `reasons` and alpha-tie set.
  - Transition: `noise_background=True` survives and the reloaded
    transition's alpha-relevant ratio matches the original; plus equality
    and hash now distinguish transitions differing only in
    `noise_background`.
  - Null AcSIA: `dumps`/`loads` round-trips without error and restores the
    empty accounts and `reasons`.
- **Backward-compatibility tests** following the established
  strip-the-fields pattern (`test_iit4_sia_loads_without_margin_fields`):
  remove the new keys from the JSON form and assert the object decodes with
  the documented defaults.
- **`format_version` tests**: a document stamped with a greater version
  raises `ValueError`; the current version loads.
- Completion gate: full pathless `uv run pytest` (doctest sweep included).

## Out of scope

- Cache-safety findings (disk-cache label collision, writable kernel-cache
  arrays, `FactoredTPM` aliasing, ignored `cache_repertoires`) — the second
  Wave-2 sub-unit.
- Structural prevention of schema drift (deriving schemas from models) — a
  serializer redesign, not a bug fix.
- Extending domain `__eq__` beyond `Transition` — tie peers and diagnostics
  compare equal by design; changing that is a formalism-level decision.
