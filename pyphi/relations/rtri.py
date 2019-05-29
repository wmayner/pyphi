@partition_types.register('RTRI')
def rwedge_partitions(mechanism, purview, node_labels=None):
    """Return an iterator over all wedge partitions *with the wedge on the purview side*.

    These are partitions which strictly split the purview and allow a subset
    of the mechanism to be split into a third partition, e.g.::

         A     B     C
        ─── ✕ ─── ✕ ───
         B     C     ∅

    See |PARTITION_TYPE| in |config| for more information.

    Args:
        mechanism (tuple[int]): A mechanism.
        purview (tuple[int]): A purview.

    Yields:
        Tripartition: all unique tripartitions of this mechanism and purview.
    """
    numerators = directed_tripartition(mechanism)
    denominators = bipartition(purview)

    yielded = set()

    def valid(factoring):
        """Return whether the factoring should be considered."""
        # pylint: disable=too-many-boolean-expressions
        numerator, denominator = factoring
        return (
            (numerator[0] or denominator[0]) and
            (numerator[1] or denominator[1]) and
            ((denominator[0] and denominator[1]) or
             not numerator[0] or
             not numerator[1])
        )

    for n, d in filter(valid, product(numerators, denominators)):
        # Normalize order of parts to remove duplicates.
        tripart = Tripartition(
            Part(n[0], d[0]),
            Part(n[1], d[1]),
            Part(n[2], ()),
            node_labels=node_labels
        ).normalize()  # pylint: disable=bad-whitespace

        def nonempty(part):
            """Check that the part is not empty."""
            return part.mechanism or part.purview

        def compressible(tripart):
            """Check if the tripartition can be transformed into a causally
            equivalent partition by combing two of its parts; e.g., A/∅ × B/∅ ×
            ∅/CD is equivalent to AB/∅ × ∅/CD so we don't include it.
            """
            pairs = [
                (tripart[0], tripart[1]),
                (tripart[0], tripart[2]),
                (tripart[1], tripart[2])
            ]
            for x, y in pairs:
                if (nonempty(x) and nonempty(y) and
                        (x.mechanism + y.mechanism == () or
                         x.purview + y.purview == ())):
                    return True
            return False

        if not compressible(tripart) and tripart not in yielded:
            yielded.add(tripart)
            yield tripart
