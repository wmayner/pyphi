`provenance` and the disk result cache no longer stamp the enclosing
repository's commit when pyphi is installed inside another project's git tree:
the discovered repository must actually track the package, else no commit is
recorded (and result caching is keyed accordingly).
