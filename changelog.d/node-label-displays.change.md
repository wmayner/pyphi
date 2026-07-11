Result displays and `to_pandas()` tables now render node labels instead of
bare integer indices wherever a substrate's labels are available — relation
relata, distinction and cause-effect-structure purviews, the specified system
state, and the actual-causation account. Node state is encoded the same way
everywhere: state 0 lowercases a label, state 1 uppercases it (the existing
binary convention), and any state ≥ 2 appends the value as a Unicode subscript
(``A₂``, ``A₃``), so the representation is now correct for k-ary units — where
the previous upper/lower casing collapsed every nonzero state to uppercase. The
underlying data attributes (``.mechanism``, ``.purview``, ``.relata``) are
unchanged and still return integer indices.
