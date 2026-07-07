Added `CauseEffectStructure.meet(other)`: the induced substructure on the
distinctions common to both structures. Structures must share a frame (same
candidate-system node indices and state); mismatches raise `ValueError`
instead of silently returning an empty result.
