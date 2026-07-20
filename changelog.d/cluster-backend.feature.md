Added the `dask` parallel backend: with the `cluster` extra installed
(`pip install "pyphi[cluster]"`) and a `distributed.Client` connected,
`config.parallel_backend = "dask"` distributes PyPhi's parallel levels
across the cluster. Added a how-to guide for running PyPhi on UW–Madison's
CHTC (independent condor jobs, fat-node jobs, and Dask worker pools via
dask-jobqueue).
