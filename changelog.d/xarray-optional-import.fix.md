Fixed `import pyphi` failing with `ModuleNotFoundError: No module named
'xarray'` when the optional `xarray` extra is not installed: the xarray
`FactoredTPM` backend is now excluded from the eager submodule walk and is
imported only when `backend="xarray"` is requested.
