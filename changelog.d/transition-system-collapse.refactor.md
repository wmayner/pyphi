Collapsed ``TransitionSystem`` (in ``pyphi/actual.py``) onto ``System`` (in ``pyphi/system.py``) via a new optional ``external_indices`` field on ``System``. The field parameterizes which substrate units are conditioned at observed state when computing repertoires; when omitted, ``System`` resolves it to ``substrate - node_indices`` (today's behavior). ``TransitionSystem`` constructs its underlying ``System`` with ``external_indices = substrate - cause_indices``, then delegates the System protocol surface via ``__getattr__``. The class shrinks from ~550 lines to ~150 lines.

The actual-causation pipeline now inherits k-ary substrate support automatically — the last call to ``Substrate._legacy_binary_joint()`` from ``pyphi/actual.py`` is gone, and ``test_kary_account_end_to_end`` (previously xfailed) now passes.

No public API changes. ``TransitionSystem``'s protocol surface and call signatures are unchanged.
