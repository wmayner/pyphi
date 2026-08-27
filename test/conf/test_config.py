"""Behavior tests for the global ``pyphi.config`` singleton.

These cover validation and the override context manager / decorator on the
real layered config. The descriptor / ``Option`` infrastructure that
previously powered the global was deleted in P10b, so only the
integration-level tests remain.
"""

import pytest

from pyphi import config


@config.override()
@pytest.mark.parametrize(
    "name,valid,invalid",
    [
        ("REPR_VERBOSITY", [0, 1, 2, 3, 4], [-1, 5]),
        ("PARALLEL", [True, False], ["True", "False", "no", 0, 1]),
    ],
)
def test_config_validation(name, valid, invalid):
    for value in valid:
        setattr(config, name, value)

    for value in invalid:
        with pytest.raises(ValueError):
            setattr(config, name, value)


class TestParallelKwargsAllowlist:
    """The allowlist must exactly mirror map_reduce's keyword surface: an
    advertised name map_reduce rejects (or an accepted name the allowlist
    silently drops) is a contract violation."""

    def test_allowlist_matches_map_reduce_signature(self):
        import inspect

        from pyphi.conf._helpers import PARALLEL_KWARGS
        from pyphi.parallel import map_reduce

        keyword_params = {
            name
            for name, p in inspect.signature(map_reduce).parameters.items()
            if p.kind is inspect.Parameter.KEYWORD_ONLY
        }
        assert set(PARALLEL_KWARGS) == keyword_params

    def test_user_override_beats_global_gate(self):
        from pyphi.conf import config
        from pyphi.conf._helpers import parallel_kwargs

        with config.override(parallel=False, progress_bars=False):
            kwargs = parallel_kwargs(
                {"parallel": True, "progress": True}, parallel=True, progress=True
            )
        assert kwargs["parallel"] is True
        assert kwargs["progress"] is True
