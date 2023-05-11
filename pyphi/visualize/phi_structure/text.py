# visualize/phi_structure/text.py

"""Textual functions for plotting phi structures."""

from ... import config, utils
from ...models import fmt


def indent(lines, amount=2, char="&nbsp;", newline="<br>"):
    return fmt.indent(lines, amount=amount, char=char, newline=newline)


class Labeler:
    def __init__(self, state, node_labels):
        self.state = state
        self.node_labels = node_labels

    def nodes(self, nodes, state=None):
        if state is None:
            state = utils.state_of(nodes, self.state)
        return "".join(
            n.upper() if state[i] else n.lower()
            for i, n in enumerate(self.node_labels.coerce_to_labels(nodes))
        )

    def hover(self, mice):
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

    def mice(self, mice):
        return f"{self.nodes(mice.purview, state=mice.specified_state)}"

    def relata(self, relata):
        return "<br>".join(map(self.mice, relata))

    def relation(self, relation):
        return f"{len(relation)}-relation<br>" + indent(
            "<br>".join(
                [
                    f"P: {self.nodes(relation.purview)}",
                    f"φ: {round(relation.phi, config.PRECISION)}",
                    "Relata:",
                    indent(self.relata(relation.relata)),
                ]
            )
        )
