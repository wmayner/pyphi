Caching
~~~~~~~

PyPhi can optionally store the results of |Phi| calculations as they're
computed in order to avoid expensive re-computation. These results can be
stored locally on the filesystem (the default setting), or in a full-fledged
database.

Caching is configured either in the ``pyphi_config.yml`` file or at runtime by
modifying ``pyphi.config``. See the `configuration documentation
<http://pyphi.readthedocs.io/en/stable/configuration.html>`_ for more
information.


Caching with MongoDb
````````````````````

Using the default caching system is easier and works out of the box, but using
a database is more robust.

To use the database-backed caching system, you must install `MongoDB
<http://www.mongodb.org/>`_. Please see their `installation guide
<http://docs.mongodb.org/manual/installation/>`_ for instructions.

Once you have MongoDB installed, use ``mongod`` to start the MongoDB server.
Make sure the ``mongod`` configuration matches the PyPhi's database
configuration settings in ``pyphi_config.yml`` (see the `configuration section
<https://pythonhosted.org/pyphi/index.html#configuration>`_ of PyPhi's
documentation).

You can also check out MongoDB's `Getting Started guide
<http://docs.mongodb.org/manual/tutorial/getting-started/>`_ or the full
`manual <http://docs.mongodb.org/manual/>`_.


Caching with Redis
``````````````````

PyPhi can also use Redis as a fast in-memory global LRU cache to store Mice
objects, reducing the memory load on PyPhi processes.

`Install Redis <http://redis.io/download>`_. The `redis.conf` file provided
with PyPhi includes the minimum settings needed to run Redis as an LRU cache:

.. code:: bash

    redis-server /path/to/pyphi/redis.conf

Once the server is running you can enable Redis caching by setting
``REDIS_CACHE: true`` in your ``pyphi_config.yml``.

**Note:** PyPhi currently flushes the connected Redis database at the start of
every execution. If you are running Redis for another application be sure PyPhi
connects to its own Redis server.


.. |phi| unicode:: U+1D6BD .. mathematical bold capital phi
