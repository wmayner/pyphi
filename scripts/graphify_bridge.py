"""Version-control the curated paper-to-code bridge edges separately from the
churny auto-generated graphify graph.

``graphify-out/graph.json`` is large (~11 MB) and regenerates with massive churn,
so it is local-only (gitignored). The hand-built ``implements``/``cites`` edges
linking IIT paper concepts to the code that implements them are the one asset
worth keeping under version control, so they live in a small committed sidecar,
``graphify-out/bridge-edges.json``.

    # after curating bridge edges, refresh the committed sidecar:
    uv run python scripts/graphify_bridge.py extract

    # after `graphify update .` regenerates the local graph, restore the edges:
    uv run python scripts/graphify_bridge.py inject

``extract`` is reviewed in the diff like a golden; ``inject`` is idempotent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_OUT = Path(__file__).resolve().parents[1] / "graphify-out"
_GRAPH = _OUT / "graph.json"
_SIDECAR = _OUT / "bridge-edges.json"

_BRIDGE_RELATIONS = {"implements", "cites"}


def _is_bridge(link: dict) -> bool:
    return link.get("relation") in _BRIDGE_RELATIONS


def extract() -> None:
    """Write the bridge edges and their endpoint nodes to the sidecar."""
    graph = json.loads(_GRAPH.read_text())
    edges = sorted(
        (link for link in graph["links"] if _is_bridge(link)),
        key=lambda e: (e["source"], e["target"], e["relation"]),
    )
    endpoint_ids = {e["source"] for e in edges} | {e["target"] for e in edges}
    nodes = sorted(
        (n for n in graph["nodes"] if n["id"] in endpoint_ids),
        key=lambda n: n["id"],
    )
    _SIDECAR.write_text(
        json.dumps({"edges": edges, "nodes": nodes}, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {_SIDECAR}: {len(edges)} edges, {len(nodes)} endpoint nodes")  # noqa: T201


def inject() -> None:
    """Merge the sidecar edges (and any missing endpoint nodes) into the graph."""
    graph = json.loads(_GRAPH.read_text())
    sidecar = json.loads(_SIDECAR.read_text())

    node_ids = {n["id"] for n in graph["nodes"]}
    added_nodes = 0
    for node in sidecar["nodes"]:
        if node["id"] not in node_ids:
            graph["nodes"].append(node)
            node_ids.add(node["id"])
            added_nodes += 1

    edge_keys = {
        (link["source"], link["target"], link.get("relation")) for link in graph["links"]
    }
    added_edges = 0
    for edge in sidecar["edges"]:
        key = (edge["source"], edge["target"], edge.get("relation"))
        if key not in edge_keys:
            graph["links"].append(edge)
            edge_keys.add(key)
            added_edges += 1

    _GRAPH.write_text(json.dumps(graph, indent=2, sort_keys=True) + "\n")
    print(f"injected into {_GRAPH}: +{added_edges} edges, +{added_nodes} nodes")  # noqa: T201


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("extract", help="write graph.json bridge edges to the sidecar")
    sub.add_parser("inject", help="merge the sidecar bridge edges into graph.json")
    args = parser.parse_args()
    if args.command == "extract":
        extract()
    else:
        inject()


if __name__ == "__main__":
    main()
