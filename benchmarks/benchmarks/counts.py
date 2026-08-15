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

    def track_cause_repertoire(self, fixture_name: str, grain: str) -> int:
        return self.counts["repertoire_algebra.py:cause_repertoire"]

    track_cause_repertoire.unit = "calls"  # type: ignore[attr-defined]

    def track_effect_repertoire(self, fixture_name: str, grain: str) -> int:
        return self.counts["repertoire_algebra.py:effect_repertoire"]

    track_effect_repertoire.unit = "calls"  # type: ignore[attr-defined]

    def track_find_mip(self, fixture_name: str, grain: str) -> int:
        return self.counts["system.py:find_mip"]

    track_find_mip.unit = "calls"  # type: ignore[attr-defined]

    def track_relations(self, fixture_name: str, grain: str) -> int:
        return self.counts["relations.py:relations"]

    track_relations.unit = "calls"  # type: ignore[attr-defined]

    def track_config_override(self, fixture_name: str, grain: str) -> int:
        return self.counts["conf/:override"]

    track_config_override.unit = "calls"  # type: ignore[attr-defined]

    # Collision-handling frames. These stay near zero while cache-key hashes
    # separate their keys, and grow with the square of the cache's size when
    # one stops; no count of PyPhi operations moves when that happens.
    def track_mapping_eq(self, fixture_name: str, grain: str) -> int:
        return self.counts["_collections_abc:__eq__"]

    track_mapping_eq.unit = "calls"  # type: ignore[attr-defined]

    def track_frozen_map_getitem(self, fixture_name: str, grain: str) -> int:
        return self.counts["frozen_map.py:__getitem__"]

    track_frozen_map_getitem.unit = "calls"  # type: ignore[attr-defined]
