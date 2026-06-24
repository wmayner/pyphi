``pyphi/_conf_legacy.py`` and its type stub are deleted. The descriptor-
based ``Config`` / ``Option`` infrastructure they provided was no longer
used by anything except its own tests after the self-owning
``_GlobalConfig`` cutover; both the module and the descriptor-pattern
tests in ``test/test_config.py`` are removed in this commit.

``pyphi/conf/legacy_global.py`` is renamed to ``pyphi/conf/_global.py``
to reflect that the module no longer wraps a legacy backend. ``Config``,
``Option``, and ``PyphiConfig`` are no longer importable from
``pyphi.conf`` (none had remaining users in the public surface).
