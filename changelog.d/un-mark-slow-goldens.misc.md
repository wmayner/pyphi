Un-mark all golden fixtures as ``slow``. The fast/slow tier split was
introduced earlier today as a workaround for the ~60-300x YAML-write
performance regression on the IIT 4.0 (2026) fixtures. With that bug
fixed (commit ``7c2e2cd2``), the formerly-slow fixtures complete in
1-2 seconds each — the full default golden suite runs in ~20 seconds,
no opt-in required. The ``slow=True`` field on :class:`GoldenFixture`
and the ``--slow`` flag are retained as infrastructure for future
genuinely-slow fixtures; they're just not in use today.
