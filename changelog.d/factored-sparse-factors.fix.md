Fixed crashes on connectivity-sparse `FactoredTPM`s (factors with size-1
non-input axes): conditioning a size-1 axis at a nonzero state raised
`IndexError` (breaking `condition()`, `subtpm()`, and `System` construction
with nonzero background states); macro TPM construction crashed on the
transition-matrix product because factors were flattened at their stored
shape instead of the full universe grid; and `repr()`, `str()`,
`to_pandas()`, and `to_xarray()` crashed on the same sparse form.
