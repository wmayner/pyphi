`Part` ordering no longer compares `node_labels`, matching its equality and hash semantics. Previously, comparing equal `Part`s that differed only in labels raised `TypeError`.
