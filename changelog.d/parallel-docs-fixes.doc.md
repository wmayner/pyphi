Corrected the parallelization how-to and the configuration references: the
global `parallel` option is a master gate, not a switch — on its own it
enables nothing, since each level of the computation must also be switched on
— and parallelism requires no optional dependency (the previously documented
`pyphi[parallel]` extra does not exist; the process-pool backend is a core
dependency).
