# visualize/phi_structure/text.py
"""Utilities for handling text."""

from toolz import identity

from ... import config, utils
from ...models import fmt


def indent(lines, amount=4, char="&nbsp;", newline="<br>"):
    return fmt.indent(lines, amount=amount, char=char, newline=newline)


class Labeler:
    def __init__(self, state, node_labels, postprocessor=None):
        self.state = state
        self.node_labels = node_labels
        self.postprocessor = postprocessor or identity

    def nodes(self, nodes, state=None):
        if state is None:
            state = utils.state_of(nodes, self.state)
        return self.postprocessor(
            "".join(
                n.upper() if state[i] else n.lower()
                for i, n in enumerate(self.node_labels.coerce_to_labels(nodes))
            )
        )

    def units(self, units):
        units = sorted(units)
        indices = [unit.index for unit in units]
        state = [unit.state for unit in units]
        return self.nodes(indices, state=state)

    def mice(self, mice):
        return f"{self.nodes(mice.purview, state=mice.specified_state)}"

    def hover_mice(self, mice):
        return "<br>".join(
            [
                f"Distinction ({mice.direction})",
                indent(
                    "<br>".join(
                        [
                            f"M: {self.nodes(mice.mechanism)}",
                            f"P: {self.nodes(mice.purview, state=mice.specified_state)}",
                            f"φ: {round(mice.phi, config.PRECISION)}",
                            f"S: {','.join(map(str, mice.specified_state))}",
                        ]
                    ),
                ),
            ]
        )

    def hover_relata(self, relata):
        return "<br>".join(map(self.mice, relata))

    def hover_relation(self, relation):
        return f"{len(relation)}-relation<br>" + indent(
            "<br>".join(
                [
                    f"P: {self.units(relation.purview)}",
                    f"φ: {round(relation.phi, config.PRECISION)}",
                    "Relata:",
                    indent(self.relata(relation)),
                ]
            )
        )

    def hover_relation_face(self, face):
        return f"{len(face)}-face<br>" + indent(
            "<br>".join(
                [
                    f"P: {self.units(face.purview)}",
                    f"φ: {round(face.phi, config.PRECISION)}",
                    "Relata:",
                    indent(self.hover_relata(face)),
                ]
            )
        )
