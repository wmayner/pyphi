Fixed `CauseEffectStructure.diff()` crashing with `TypeError` whenever the two
structures' relation sets differ: relation gained/lost changes now key on the
relation's mechanisms in sorted order.
