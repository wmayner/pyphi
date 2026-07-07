Added `CauseEffectStructure.relabel(mapping)` (and `pyphi.relabel`):
rewrite a structure through a node-index bijection, reconstructing every
nested result object with mapped indices. φ values are preserved exactly;
tie back-references are dropped; IIT 3.0 SIAs and structure views are not
supported.
