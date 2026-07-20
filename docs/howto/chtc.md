# Run PyPhi on a cluster (CHTC)

This guide covers running PyPhi on UW–Madison's Center for High-Throughput
Computing (CHTC), and applies with minor changes to any HTCondor pool. It
assumes a CHTC account and basic familiarity with submitting jobs; see the
[CHTC getting-started roadmap](https://chtc.cs.wisc.edu/uw-research-computing/htc-roadmap)
for that groundwork.

## Which CHTC system?

CHTC operates two systems. The **HTC system** (HTCondor) runs many
independent single-node jobs; the **HPC cluster** (Slurm) is for MPI-style
computations that internally span multiple nodes. PyPhi computations are
single-node by construction — parallelism spreads within one machine, or
across machines only via the Dask backend below — and CHTC directs
single-node work to the HTC system. Use HTC.

Three deployment patterns follow, ordered by how well CHTC supports them
today.

## Build the PyPhi container

All three patterns deliver PyPhi to execute nodes as an Apptainer
container. Build a wheel, then the image (on the access point or any Linux
machine with Apptainer):

```bash
# In your PyPhi checkout:
uv build                        # writes dist/pyphi-<version>-py3-none-any.whl
```

`pyphi.def`:

```
Bootstrap: docker
From: python:3.13-bookworm

%files
    dist/pyphi-*.whl /opt/

%post
    pip install --no-cache-dir /opt/pyphi-*.whl

%runscript
    exec python "$@"
```

```bash
apptainer build pyphi.sif pyphi.def
```

Once PyPhi 2.0 is on PyPI the `%files` section can be dropped in favor of
`pip install pyphi` in `%post`.

## Pattern A — many independent runs (fully supported)

The canonical HTC workload: each condor job runs one self-contained PyPhi
computation (one substrate/state/configuration cell), and you collect the
saved results afterwards. Write results with `pyphi.provenance.save_json`
(or `.save()` on result objects) so each output is self-describing.

`run_cell.py` — one cell per job, selected by the process number:

```python
import sys

import pyphi

cell = int(sys.argv[1])

# Define your substrates/states/configs however you like; index them by cell.
substrate = pyphi.examples.basic_substrate()
states = list(pyphi.utils.all_states(substrate.size))
state = states[cell % len(states)]

sia = pyphi.System(substrate, state).sia()
sia.save(f"sia_state{cell}.json.gz")
```

`sweep.sub`:

```
universe = container
container_image = pyphi.sif

executable = run_cell.py
arguments = $(Process)
transfer_executable = false

transfer_input_files = run_cell.py
should_transfer_files = YES
when_to_transfer_output = ON_EXIT

request_cpus = 1
request_memory = 4GB
request_disk = 4GB

log = sweep.log
error = sweep.$(Process).err
output = sweep.$(Process).out

queue 8
```

Submit with `condor_submit sweep.sub`. Jobs have a 72-hour default runtime
limit; keep per-job inputs/outputs under CHTC's file-transfer guidance
(~100 MB per file) or arrange staging with CHTC. For dependent stages
(compute → aggregate), see CHTC's DAGMan guides.

## Pattern B — one big analysis on a fat node (fully supported)

For a single analysis too large for a lab machine, request one many-core,
high-memory slot and let PyPhi's default process backend saturate it:

```
universe = container
container_image = pyphi.sif

executable = analyze.py
transfer_executable = false

transfer_input_files = analyze.py
should_transfer_files = YES
when_to_transfer_output = ON_EXIT

request_cpus = 32
request_memory = 200GB
request_disk = 20GB

log = analyze.log
error = analyze.err
output = analyze.out

queue
```

In `analyze.py`, enable parallelism (`pyphi.config.parallel = True`); the
process backend uses every requested core. See CHTC's high-memory-job
guide for current per-slot limits, and `pyphi.estimate_analysis` for
sizing the workload before submitting.

## Pattern C — distributing one analysis across machines (pilot)

The `dask` backend spreads a single computation's parallel levels
(distinctions, purviews, partitions) across a Dask worker pool. On an
HTCondor pool, [dask-jobqueue](https://jobqueue.dask.org) launches workers
as ordinary condor jobs that connect back to a scheduler in your session
on the access point:

```python
from dask_jobqueue import HTCondorCluster
from distributed import Client

cluster = HTCondorCluster(
    cores=1,
    processes=1,
    memory="4GB",
    disk="4GB",
    job_extra_directives={
        "universe": "container",
        "container_image": "pyphi.sif",
    },
)
cluster.scale(jobs=32)          # 32 single-core workers
client = Client(cluster)

import pyphi

pyphi.config.parallel = True
pyphi.config.parallel_backend = "dask"

substrate = pyphi.examples.basic_substrate()
sia = pyphi.System(substrate, (1, 0, 0)).sia()
```

Notes:

- **Single-threaded workers** (`cores=1, processes=1`): PyPhi's work is
  CPU-bound Python, so extra worker threads do not help.
- **Nesting**: only the outermost parallel level distributes; levels
  reached inside a worker task run within that task.
- **Preemption**: HTC slots can be preempted; Dask reschedules lost tasks
  automatically, at the cost of recomputing them.
- **Dashboard**: forward it over SSH
  (`ssh -L 8787:localhost:8787 <access point>`), then open
  `http://localhost:8787`.

**Support status — read before relying on this pattern.** CHTC does not
currently document or support Dask on the HTC system, and most ports on
CHTC submit and execute nodes are closed. A Dask cluster needs
bidirectional TCP between the scheduler (your access point session) and
the workers (execute nodes). Whether that traffic is permitted from your
access point is a site question. Before adopting this pattern, ask CHTC
facilitation (chtc@cs.wisc.edu):

1. Are inbound connections from execute nodes to a high port on my access
   point permitted (a `dask.distributed` scheduler listening in a user
   session)? If not, is there a designated machine where this is allowed?
2. Is there a policy on long-lived coordinator processes running on access
   points for the duration of a workload?
3. What wall-time and sizing guidance applies to held worker jobs (e.g.
   32 single-core workers held for a few hours)?

If the answers rule this pattern out, Patterns A and B cover sweeps and
single big analyses with fully supported mechanics.
