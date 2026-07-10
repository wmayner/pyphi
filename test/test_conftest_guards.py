"""The conftest guard against silently-skipped slow-lane selections."""

import pytest

import conftest


class _FakeOption:
    def __init__(self, markexpr):
        self.markexpr = markexpr


class _FakeConfig:
    def __init__(self, markexpr, flags):
        self.option = _FakeOption(markexpr)
        self._flags = flags

    def getoption(self, name):
        return self._flags.get(name, False)


@pytest.mark.parametrize(
    "markexpr,flags,should_error",
    [
        ("slow", {}, True),
        ("veryslow", {}, True),
        ("slow or veryslow", {}, True),
        ("macro and slow", {}, True),
        ("slow", {"--slow": True}, False),
        ("veryslow", {"--veryslow": True}, False),
        ("not slow", {}, False),
        ("", {}, False),
        (None, {}, False),
    ],
)
def test_slow_markexpr_requires_flag(markexpr, flags, should_error):
    config = _FakeConfig(markexpr, flags)
    if should_error:
        with pytest.raises(pytest.UsageError):
            conftest.pytest_configure(config)
    else:
        conftest.pytest_configure(config)
