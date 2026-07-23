The generated condor submit file sets `preserve_relative_paths = true`.
Without it, HTCondor flattens the transferred task file to the scratch
root while `run_task.sh` expects it under `tasks/`, so every job on a
real pool failed with `FileNotFoundError` (the local runner bypasses
the submit file and never saw it).
