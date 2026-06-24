"""Edge benchmarks: parallelism and cache sensitivity (wall-time only)."""

from __future__ import annotations

from pyphi import config

from ._fixtures import FIXTURES_BY_NAME, build_system

# A mid-size fixture: large enough that parallel/cache effects show, small
# enough to stay under the nightly timeout.
_MID = "rule110_iit4_2023"


class ParallelSia:
    params = [True, False]
    param_names = ("parallel",)
    timeout = 600.0
    number = 1

    def setup(self, parallel: bool) -> None:
        self.fixture = FIXTURES_BY_NAME[_MID]

    def time_sia(self, parallel: bool) -> None:
        with self.fixture.config_context(), config.override(parallel=parallel):
            build_system(self.fixture).sia()


class RepertoireCache:
    params = [True, False]
    param_names = ("warm",)
    timeout = 600.0
    number = 1

    def setup(self, warm: bool) -> None:
        self.fixture = FIXTURES_BY_NAME[_MID]

    def time_repertoires(self, warm: bool) -> None:
        with self.fixture.config_context(), config.override(cache_repertoires=True):
            system = build_system(self.fixture)
            if warm:
                system.sia()  # prime the repertoire cache
            for mechanism in system.node_indices:
                for purview in system.node_indices:
                    system.cause_repertoire((mechanism,), (purview,))
                    system.effect_repertoire((mechanism,), (purview,))
