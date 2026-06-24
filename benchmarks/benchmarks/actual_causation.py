"""Actual Causation benchmark: the account of a canonical transition."""

from __future__ import annotations

from pyphi import actual, config, examples
from pyphi.conf import presets


class Account:
    timeout = 600.0
    number = 1

    def setup(self) -> None:
        self.ctx = config.override(**presets.iit3)
        self.ctx.__enter__()
        self.transition = actual.Transition(
            examples.actual_causation_substrate(),
            (1, 0),
            (1, 0),
            (0, 1),
            (0, 1),
        )

    def teardown(self) -> None:
        self.ctx.__exit__(None, None, None)

    def time_account(self) -> None:
        actual.account(self.transition)
