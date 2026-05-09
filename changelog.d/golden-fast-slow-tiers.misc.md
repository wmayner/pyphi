Split the golden regression suite into fast and slow tiers. The fast tier
covers IIT 3.0 + IIT 4.0 (2023) on small substrates and runs in under 10
seconds; the slow tier adds the IIT 4.0 (2026) cap-formalism variants and
large-substrate fixtures (~13 minutes). Slow fixtures carry ``slow=True``
on :class:`GoldenFixture` and inherit the existing ``--slow`` opt-in flag
(see top-level ``conftest.py``). Run ``pytest --slow`` before merging or
regenerating goldens.
