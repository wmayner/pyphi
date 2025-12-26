Caching
~~~~~~~

PyPhi can use Redis as a fast in-memory global LRU cache to store MICE objects,
reducing the memory load on PyPhi processes.

To use this feature, install PyPhi with the extra dependencies ``[caching]``, *e.g*

.. code:: bash

    pip install "pyphi[caching]""

The ``redis.conf`` file provided with PyPhi includes the minimum settings needed
to run Redis as an LRU cache:

.. code:: bash

    redis-server /path/to/pyphi/redis.conf

Once the server is running you can enable Redis caching by setting
``REDIS_CACHE: true`` in your ``pyphi_config.yml``.

**Note:** PyPhi currently flushes the connected Redis database at the start of
every execution. If you are running Redis for another application be sure PyPhi
connects to its own Redis server.


.. |phi| unicode:: U+1D6BD .. mathematical bold capital phi
