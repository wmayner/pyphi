`plot_dynamics` now returns its figure and axes instead of calling
`plt.show()`, and accepts `fig`/`ax`/`figsize`. `plot_tpm` is exported from
`pyphi.visualize`. The auxiliary plot modules (connectivity, distribution,
dynamics, ising) now separate data extraction from figure emission and are
covered by tests.
