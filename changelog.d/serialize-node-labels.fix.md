Node labels now survive serialization everywhere they are displayed. Mechanism
and system partitions (`Part`, `JointPartition` and its variants, `NullCut`,
`DirectedBipartition`, `DirectedJointPartition`), state specifications, and a
substrate's factored TPM previously lost their labels on round-trip, so a
reloaded result rendered its MIP and purviews with bare indices while a fresh
one showed labels. Conversely, an object saved with no labels no longer
inherits the document's label frame on load, which could attach another
object's labels to it.
