Comparison operators on φ-objects and nodes return `NotImplemented` for
foreign types instead of raising `AttributeError`, so equality against
arbitrary objects is `False` and ordering raises the standard `TypeError`.
