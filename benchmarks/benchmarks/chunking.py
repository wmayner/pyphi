"""Cost-balanced chunking benchmark (wall-time).

``RelationsParallel`` provides a heterogeneous parallel relations workload that
exercises the size_func cost-balanced packing and the ``num_workers`` count
floor through pyphi's own (importable) relation worker. The cost-balancing win
is read by comparing this benchmark across the B18 commit boundary
(``asv continuous BASE HEAD``), since the relations size_func is always on
post-B18 rather than an in-process toggle.

(The count-floor effect in isolation was validated separately by a one-off
homogeneous-workload de-risk experiment; a synthetic asv benchmark for it is
not viable because loky workers cannot import a function defined in the asv
benchmark module.)
"""

from __future__ import annotations

from pyphi import config

from ._fixtures import FIXTURES_BY_NAME
from ._fixtures import build_system


class RelationsParallel:
    """Heterogeneous parallel relations workload (compare across B18 boundary)."""

    timeout = 600.0
    number = 1

    def setup(self) -> None:
        self.fixture = FIXTURES_BY_NAME["rule110_iit4_2023"]

    def time_relations(self) -> None:
        with self.fixture.config_context(), config.override(
            parallel=True,
            progress_bars=False,
            parallel_relation_evaluation={
                **config.infrastructure.parallel_relation_evaluation,
                "parallel": True,
                "sequential_threshold": 1,
            },
        ):
            build_system(self.fixture).sia()
