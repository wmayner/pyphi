"""Wall-time benchmarks: every golden fixture x every applicable grain."""

from __future__ import annotations

from ._fixtures import FIXTURES_BY_NAME, GRAINS, applies, run_grain


class Layers:
    params = (sorted(FIXTURES_BY_NAME), list(GRAINS))
    param_names = ("fixture", "grain")
    # SIA on the larger fixtures is slow; give ASV room and few repeats.
    timeout = 600.0
    number = 1
    repeat = (1, 3, 30.0)  # (min_repeat, max_repeat, max_seconds)

    def setup(self, fixture_name: str, grain: str) -> None:
        fixture = FIXTURES_BY_NAME[fixture_name]
        if not applies(fixture, grain):
            # ASV skips the (param) combo when setup raises NotImplementedError.
            raise NotImplementedError
        self.fixture = fixture

    def time_grain(self, fixture_name: str, grain: str) -> None:
        run_grain(self.fixture, grain)
