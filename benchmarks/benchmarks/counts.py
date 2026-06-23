"""Deterministic call-count metrics over the full zoo (ASV track_*).

Counts are exact, so ASV step-detection flags any change without false
positives — full-zoo count-regression coverage with no in-repo pins.
"""

from __future__ import annotations

from functools import partial

from ._fixtures import (
    FIXTURES_BY_NAME,
    FRAMES,
    GRAINS,
    applies,
    count_calls,
    run_grain,
)


class Counts:
    params = (sorted(FIXTURES_BY_NAME), list(GRAINS))
    param_names = ("fixture", "grain")
    timeout = 600.0

    def setup(self, fixture_name: str, grain: str) -> None:
        fixture = FIXTURES_BY_NAME[fixture_name]
        if not applies(fixture, grain):
            raise NotImplementedError
        self.counts = count_calls(partial(run_grain, fixture, grain), FRAMES)

    def track_find_mip(self, fixture_name: str, grain: str) -> int:
        return self.counts["system.py:find_mip"]

    track_find_mip.unit = "calls"  # type: ignore[attr-defined]

    def track_relations(self, fixture_name: str, grain: str) -> int:
        return self.counts["relations.py:relations"]

    track_relations.unit = "calls"  # type: ignore[attr-defined]

    def track_config_override(self, fixture_name: str, grain: str) -> int:
        return self.counts["conf/:override"]

    track_config_override.unit = "calls"  # type: ignore[attr-defined]
