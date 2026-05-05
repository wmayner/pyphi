"""Compute the full set of values to capture in a golden fixture.

Three layers, each independently useful as a regression target:

- Layer 1: Repertoires for every (mechanism, purview) pair, plus unconstrained
  repertoires for every purview. These are the foundation; if these change,
  everything downstream is wrong.

- Layer 2: Mechanism-level analysis — cause and effect MIPs (phi values,
  partitions, partitioned repertoires, specified states for IIT 4.0).

- Layer 3: System-level analysis — the full SystemIrreducibilityAnalysis
  including system phi, system partition, and (for IIT 4.0) the full
  PhiStructure with distinctions and relations.

Each layer can be skipped via ``GoldenFixture.skip_layers`` for fixtures
where it doesn't apply (e.g., IIT 3.0 doesn't have PhiStructure).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pyphi import Direction
from pyphi import Subsystem
from pyphi import utils as pyphi_utils

from .canonicalize import canonical_mechanism
from .canonicalize import canonical_partition
from .canonicalize import canonical_purview
from .canonicalize import canonical_state_set
from .fixture import GoldenFixture
from .fixture import array_ref
from .fixture import network_hash


def compute_all_layers(
    fixture: GoldenFixture,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Compute every value to be captured by the fixture.

    Runs inside the fixture's config context. Returns ``(structured, arrays)``
    suitable for ``store_fixture()``.
    """
    network = fixture.build_network()
    nodes = fixture.node_indices or network.node_indices
    subsystem = Subsystem(network, fixture.state, nodes)

    structured: dict[str, Any] = {
        "network_hash": network_hash(np.asarray(network.tpm), np.asarray(network.cm)),
    }
    arrays: dict[str, np.ndarray] = {}
    array_counter = [0]

    def stash(arr: np.ndarray) -> str:
        """Store an array in the npz, return its reference key."""
        key = f"a{array_counter[0]}"
        array_counter[0] += 1
        arrays[key] = np.ascontiguousarray(arr, dtype=np.float64)
        return array_ref(key)

    if "repertoires" not in fixture.skip_layers:
        structured["repertoires"] = _compute_repertoires(subsystem, stash)

    if "mechanism_mips" not in fixture.skip_layers:
        structured["mechanism_mips"] = _compute_mechanism_mips(subsystem, stash)

    if "sia" not in fixture.skip_layers:
        structured["sia"] = _compute_sia(subsystem, stash)

    if "phi_structure" not in fixture.skip_layers:
        structured["phi_structure"] = _compute_phi_structure(subsystem, stash)

    return structured, arrays


# ============== Layer 1: Repertoires ==============


def _compute_repertoires(subsystem: Subsystem, stash: Any) -> list[dict[str, Any]]:
    """For every (mechanism, purview) pair, capture the cause and effect repertoires.

    Also captures the unconstrained repertoires for each purview (which depend
    only on the purview, not the mechanism).
    """
    nodes = subsystem.node_indices
    out: list[dict[str, Any]] = []

    for mechanism in pyphi_utils.powerset(nodes):
        for purview in pyphi_utils.powerset(nodes):
            cause_rep = subsystem.cause_repertoire(mechanism, purview)
            effect_rep = subsystem.effect_repertoire(mechanism, purview)

            entry: dict[str, Any] = {
                "mechanism": canonical_mechanism(mechanism),
                "purview": canonical_purview(purview),
                "cause_repertoire": stash(cause_rep),
                "effect_repertoire": stash(effect_rep),
            }
            out.append(entry)

    return out


# ============== Layer 2: Mechanism MIPs ==============


def _compute_mechanism_mips(subsystem: Subsystem, stash: Any) -> list[dict[str, Any]]:
    """For every (mechanism, purview), capture the MIP analysis."""
    nodes = subsystem.node_indices
    out: list[dict[str, Any]] = []

    for mechanism in pyphi_utils.powerset(nodes, nonempty=True):
        for purview in pyphi_utils.powerset(nodes, nonempty=True):
            for direction in [Direction.CAUSE, Direction.EFFECT]:
                try:
                    mip = subsystem.find_mip(direction, mechanism, purview)
                except Exception as e:
                    out.append(
                        {
                            "mechanism": canonical_mechanism(mechanism),
                            "purview": canonical_purview(purview),
                            "direction": direction.name,
                            "error": f"{type(e).__name__}: {e}",
                        }
                    )
                    continue

                entry: dict[str, Any] = {
                    "mechanism": canonical_mechanism(mechanism),
                    "purview": canonical_purview(purview),
                    "direction": direction.name,
                    "phi": float(mip.phi) if mip.phi is not None else None,
                    "partition": canonical_partition(mip.partition),
                }

                # Specified states (IIT 4.0). Sort lexicographically when ties.
                if hasattr(mip, "ties") and mip.ties is not None:
                    entry["specified_states"] = canonical_state_set(
                        ria.specified_state.state
                        for ria in mip.ties
                        if hasattr(ria, "specified_state")
                        and ria.specified_state is not None
                    )

                # Repertoires (some IIT 3.0 paths leave these unset)
                if hasattr(mip, "repertoire") and mip.repertoire is not None:
                    entry["repertoire"] = stash(np.asarray(mip.repertoire))
                if (
                    hasattr(mip, "partitioned_repertoire")
                    and mip.partitioned_repertoire is not None
                ):
                    entry["partitioned_repertoire"] = stash(
                        np.asarray(mip.partitioned_repertoire)
                    )

                out.append(entry)

    return out


# ============== Layer 3: SIA ==============


def _compute_sia(subsystem: Subsystem, stash: Any) -> dict[str, Any]:
    """Capture the system-level irreducibility analysis."""
    try:
        sia = subsystem.sia()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    out: dict[str, Any] = {
        "phi": float(sia.phi) if sia.phi is not None else None,
        "partition": canonical_partition(getattr(sia, "partition", None)),
    }

    # IIT 4.0: cut + partition + system_state + ties
    if hasattr(sia, "cut") and sia.cut is not None:
        out["cut"] = canonical_partition(sia.cut)

    if hasattr(sia, "system_state") and sia.system_state is not None:
        ss = sia.system_state
        ss_dict: dict[str, Any] = {}
        for direction in ("cause", "effect"):
            spec = getattr(ss, direction, None)
            if spec is not None and hasattr(spec, "ties"):
                ss_dict[direction] = canonical_state_set(s.state for s in spec.ties)
        if ss_dict:
            out["system_state"] = ss_dict

    return out


# ============== Layer 3b: Phi-structure (IIT 4.0 only) ==============


def _compute_phi_structure(subsystem: Subsystem, stash: Any) -> dict[str, Any]:
    """Capture the IIT 4.0 PhiStructure."""
    try:
        from pyphi.new_big_phi import phi_structure
    except ImportError:
        return {"error": "phi_structure not available"}

    try:
        ps = phi_structure(subsystem)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    out: dict[str, Any] = {
        "big_phi": float(ps.big_phi) if ps.big_phi is not None else None,
    }

    # Distinctions, sorted by mechanism for determinism
    distinctions = sorted(
        ps.distinctions,
        key=lambda d: tuple(d.mechanism),
    )
    out["distinctions"] = [
        {
            "mechanism": canonical_mechanism(d.mechanism),
            "phi": float(d.phi),
            "cause_purview": canonical_purview(d.cause.purview)
            if hasattr(d, "cause") and d.cause is not None
            else None,
            "effect_purview": canonical_purview(d.effect.purview)
            if hasattr(d, "effect") and d.effect is not None
            else None,
        }
        for d in distinctions
    ]
    out["num_distinctions"] = len(distinctions)

    # Relations: count and total phi (full enumeration is expensive for large nets)
    relations = ps.relations
    if hasattr(relations, "num_relations"):
        out["num_relations"] = int(relations.num_relations())
    if hasattr(relations, "sum_phi"):
        out["sum_phi_relations"] = float(relations.sum_phi())

    return out
