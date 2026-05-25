"""Compute the full set of values to capture in a golden fixture.

Three layers, each independently useful as a regression target:

- Layer 1: Repertoires for every (mechanism, purview) pair, plus unconstrained
  repertoires for every purview. These are the foundation; if these change,
  everything downstream is wrong.

- Layer 2: Mechanism-level analysis — cause and effect MIPs (phi values,
  partitions, partitioned repertoires, specified states for IIT 4.0).

- Layer 3: System-level analysis — the full SystemIrreducibilityAnalysis
  including system phi, system partition, and (for IIT 4.0) the full
  CauseEffectStructure with distinctions and relations.

Each layer can be skipped via ``GoldenFixture.skip_layers`` for fixtures
where it doesn't apply (e.g., IIT 3.0 doesn't have CauseEffectStructure).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pyphi import Direction
from pyphi import System
from pyphi import utils as pyphi_utils
from pyphi.measures.distribution import resolve_mechanism_measure
from pyphi.measures.distribution import resolve_system_measure

from .canonicalize import canonical_mechanism
from .canonicalize import canonical_partition
from .canonicalize import canonical_purview
from .canonicalize import canonical_state_set
from .fixture import GoldenFixture
from .fixture import array_ref
from .fixture import substrate_hash


def compute_all_layers(
    fixture: GoldenFixture,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Compute every value to be captured by the fixture.

    Runs inside the fixture's config context. Returns ``(structured, arrays)``
    suitable for ``store_fixture()``.
    """
    substrate = fixture.build_substrate()
    nodes = fixture.node_indices or substrate.node_indices
    system = System(substrate, fixture.state, nodes)

    structured: dict[str, Any] = {
        "substrate_hash": substrate_hash(
            # Hash the legacy binary joint shape so fixtures stay stable
            # across the Substrate.joint_tpm() shape unification. All golden
            # fixtures are binary substrates.
            np.asarray(substrate._legacy_binary_joint()),
            np.asarray(substrate.cm),
        ),
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
        structured["repertoires"] = _compute_repertoires(system, stash)

    if "mechanism_mips" not in fixture.skip_layers:
        structured["mechanism_mips"] = _compute_mechanism_mips(system, stash)

    if "sia" not in fixture.skip_layers:
        # Detect IIT version from the nested iit config override (if present),
        # falling back to the flat FORMALISM key for older-style fixtures.
        iit_override = fixture.config_overrides.get("iit")
        if iit_override is not None and hasattr(iit_override, "version"):
            iit_version = 3.0 if iit_override.version == "IIT_3_0" else 4.0
        else:
            formalism_name = fixture.config_overrides.get("FORMALISM", "IIT_4_0_2023")
            iit_version = 3.0 if formalism_name == "IIT_3_0" else 4.0
        structured["sia"] = _compute_sia(system, stash, iit_version)

    if "phi_structure" not in fixture.skip_layers:
        structured["phi_structure"] = _compute_ces(system, stash)

    return structured, arrays


# ============== Layer 1: Repertoires ==============


def _compute_repertoires(system: System, stash: Any) -> list[dict[str, Any]]:
    """For every (mechanism, purview) pair, capture the cause and effect repertoires.

    Also captures the unconstrained repertoires for each purview (which depend
    only on the purview, not the mechanism).
    """
    nodes = system.node_indices
    out: list[dict[str, Any]] = []

    for mechanism in pyphi_utils.powerset(nodes):
        for purview in pyphi_utils.powerset(nodes):
            cause_rep = system.cause_repertoire(mechanism, purview)
            effect_rep = system.effect_repertoire(mechanism, purview)

            entry: dict[str, Any] = {
                "mechanism": canonical_mechanism(mechanism),
                "purview": canonical_purview(purview),
                "cause_repertoire": stash(cause_rep),
                "effect_repertoire": stash(effect_rep),
            }
            out.append(entry)

    return out


# ============== Layer 2: Mechanism MIPs ==============


def _compute_mechanism_mips(system: System, stash: Any) -> list[dict[str, Any]]:
    """For every (mechanism, purview), capture the MIP analysis."""
    nodes = system.node_indices
    out: list[dict[str, Any]] = []

    for mechanism in pyphi_utils.powerset(nodes, nonempty=True):
        for purview in pyphi_utils.powerset(nodes, nonempty=True):
            for direction in [Direction.CAUSE, Direction.EFFECT]:
                try:
                    mip = system.find_mip(direction, mechanism, purview)
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


def _compute_sia(system: System, stash: Any, iit_version: float) -> dict[str, Any]:
    """Capture the system-level irreducibility analysis.

    Dispatches on IIT version because ``System.sia()`` is hardcoded to call
    ``new_big_phi.sia()`` regardless of ``config.IIT_VERSION``
    (``pyphi/system.py:1391``). The genuine IIT 3.0 SIA path is reachable
    only via ``pyphi.formalism.iit3.sia(s)``. Without this dispatch, IIT 3.0
    fixtures would silently test the IIT 4.0 SIA framework with EMD as the
    measure — not the actual IIT 3.0 SIA algorithm. P4 (formalism split) is
    expected to make the entry-point dispatch consistent; until then we route
    explicitly here so the harness covers both code paths.
    """
    try:
        if iit_version == 3.0:
            from pyphi.formalism import iit3 as _iit3

            sia = _iit3.sia(system)
        else:
            sia = system.sia()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    out: dict[str, Any] = {
        "phi": float(sia.phi) if sia.phi is not None else None,
    }

    # ``signed_phi`` is the IIT 4.0 SIA's pre-clamp value; for systems where
    # PyPhi computes negative integration ("preventative cause"), this
    # differs from ``phi``. IIT 3.0 SIA doesn't expose it.
    if hasattr(sia, "signed_phi") and sia.signed_phi is not None:
        out["signed_phi"] = float(sia.signed_phi)

    # IIT 3.0 SIA cut capture is skipped: the IIT 3.0 SIA selects a MIP via
    # ``MapReduce(reduce_func=min, ...)`` over ``OrderableByPhi`` analyses,
    # and ``min()`` breaks ties by first-occurrence. When multiple cuts hit
    # the same minimum phi (a frequent case on small substrates), which one
    # ``min()`` picks depends on the iteration order of ``sia_partitions``
    # — and that interacts with cross-test state in a way that makes the
    # picked cut order-dependent across fixture runs (verified empirically:
    # running ``basic_iit3_emd_tri`` before ``basic_iit3_emd`` produces a
    # different cut for ``basic_iit3_emd`` than running it alone). The phi
    # value itself is deterministic, as are the CES summary stats below.
    #
    # Resolving this requires either (a) extending the IIT 3.0 SIA to
    # capture all tied minima as a set, or (b) adding a structural
    # tie-breaker to ``OrderableByPhi`` so ``min()`` picks a canonical
    # winner. Both are code changes outside P6's type-system cleanup
    # scope; tracked for a follow-up. IIT 4.0 SIAs always expose
    # ``partition`` (a DirectedSetPartition / EdgeCut), captured below.
    if iit_version != 3.0 and hasattr(sia, "partition") and sia.partition is not None:
        out["partition"] = canonical_partition(sia.partition)

    if hasattr(sia, "system_state") and sia.system_state is not None:
        ss = sia.system_state
        ss_dict: dict[str, Any] = {}
        for direction in ("cause", "effect"):
            spec = getattr(ss, direction, None)
            if spec is not None and hasattr(spec, "ties"):
                ss_dict[direction] = canonical_state_set(s.state for s in spec.ties)
        if ss_dict:
            out["system_state"] = ss_dict

    if (
        hasattr(sia, "partitioned_distinctions")
        and sia.partitioned_distinctions is not None
    ):
        out["partitioned_distinctions_size"] = len(sia.partitioned_distinctions)
        out["partitioned_distinctions_phi_sum"] = float(
            sum(d.phi for d in sia.partitioned_distinctions)
        )

    return out


# ============== Layer 3b: Phi-structure (IIT 4.0 only) ==============


def _compute_ces(system: System, stash: Any) -> dict[str, Any]:
    """Capture the IIT 4.0 CauseEffectStructure."""
    try:
        from pyphi import config as _config
        from pyphi.formalism.iit4 import ces
    except ImportError:
        return {"error": "phi_structure not available"}

    try:
        ps = ces(
            system,
            system_measure=resolve_system_measure(
                _config.formalism.iit.system_phi_measure
            ),
            specification_measure=resolve_mechanism_measure(
                _config.formalism.iit.specification_measure
            ),
        )
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
