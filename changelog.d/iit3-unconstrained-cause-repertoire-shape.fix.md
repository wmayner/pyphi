Fixed the shape of the unconstrained (empty-mechanism) cause repertoire on
systems with external (background) units. It was constructed over the
system's units instead of the full substrate's, giving it one axis per
system unit rather than one per substrate unit — inconsistent with every
other repertoire, which always carries one axis per substrate unit
(background units as size-1 axes). On candidate subsystems — as visited
during ``complexes`` / ``maximal_complex`` — the missing axis made products
of per-part repertoires misalign; under IIT 3.0 with the EMD measure this
raised ``ValueError: hamming_emd requires distributions of equal shape`` on
densely connected substrates (e.g. ``rule110``). Whole-substrate systems
were unaffected, since their unit set already equals the substrate's.
