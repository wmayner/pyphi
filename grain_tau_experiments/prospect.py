#!/usr/bin/env python
"""Prospect for substrates where a temporal-grain (τ≥2) unit wins the search.

Hunts two channels suggested by theory:
(A) micro-forfeit — the overlapping micro system's state is unreachable
    under its own marginalized TPM, so the grain-2 system wins by forfeit;
(B) tie escalation — grain-2 ties the micro incumbent at φₛ and the
    Composition (big Φ) escalation resolves in its favor.

Each run records the winner, every grain-2 record, and tie/forfeit
markers. Raw per-candidate records are saved alongside the summary.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

import pyphi
from pyphi import config
from pyphi.conf import presets
from pyphi.macro.search import SearchBounds
from pyphi.macro.search import complexes

pyphi.config.progress_bars = False

OUTDIR = Path(__file__).parent


def tpm_from_fn(f, n):
    tpm = np.zeros((2**n, n))
    for row in range(2**n):
        nxt = f(row)
        tpm[row] = [(nxt >> k) & 1 for k in range(n)]
    return tpm


def noised(tpm, eps):
    return tpm * (1 - eps) + eps * 0.5


def consecutive_pair(fn_table, n, start=1, burn=8):
    """Two consecutive states on the deterministic skeleton's trajectory."""
    s = start % (2**n)
    for _ in range(burn):
        s = fn_table[s]
    t = fn_table[s]
    return [tuple((s >> k) & 1 for k in range(n)), tuple((t >> k) & 1 for k in range(n))]


def run_one(label, tpm, history, bounds, cap):
    sub = pyphi.Substrate(tpm)
    est = bounds.estimate(sub)
    if est.distinct_systems_upper_bound > cap:
        return {
            "label": label,
            "skipped": f"estimate {est.distinct_systems_upper_bound} > cap",
        }
    with config.override(**presets.iit4_2023):
        result = complexes(sub, history, bounds)
        records = [
            {
                "units": [
                    [list(u.micro_constituents), u.micro_grain] for u in r.system.units
                ],
                "phi": float(r.phi),
            }
            for r in result.records
        ]
        winners = [
            {
                "footprint": list(c.node_indices),
                "phi": float(c.phi),
                "grains": [u.micro_grain for u in (c.units or ())],
                "margin": c.exclusion_margin,
                "effectively_tied": c.effectively_tied,
            }
            for c in result.complexes
        ]
        num_ties = len(result.ties)
        g2 = [r for r in records if any(g > 1 for _, g in r["units"])]
        top_g2 = max((r["phi"] for r in g2), default=None)
        top_all = max((r["phi"] for r in records), default=None)
        hit = any(any(g > 1 for g in w["grains"]) for w in winners)
        near = (
            top_g2 is not None
            and top_all is not None
            and top_g2 > 0
            and abs(top_g2 - top_all) < 1e-10
        )
        return {
            "label": label,
            "tpm": np.asarray(tpm).tolist(),
            "history": [list(s) for s in history],
            "estimate": est.distinct_systems_upper_bound,
            "winners": winners,
            "num_ties": num_ties,
            "num_grain2_records": len(g2),
            "top_grain2_phi": top_g2,
            "top_phi": top_all,
            "HIT": hit,
            "NEAR_TIE": bool(near and not hit),
            "records": records,
        }


def structured_family(n):
    """Deterministic skeleton functions keyed by name, for universe size n."""
    fams = {}
    if n == 3:
        # swap pair + driven third unit variants
        def swap_drive_copy(row):
            a, b = row & 1, (row >> 1) & 1
            return b | (a << 1) | (a << 2)  # C copies A

        def swap_drive_xor(row):
            a, b, c = row & 1, (row >> 1) & 1, (row >> 2) & 1
            return b | ((a ^ c) << 1) | (a << 2)  # B reads A xor C; C copies A

        def ring3(row):
            a, b, c = row & 1, (row >> 1) & 1, (row >> 2) & 1
            return c | (a << 1) | (b << 2)

        def swapnot_drive(row):
            a, b = row & 1, (row >> 1) & 1
            return (1 - b) | (a << 1) | ((a ^ b) << 2)

        fams = {
            "swap_drive_copy": swap_drive_copy,
            "swap_drive_xor": swap_drive_xor,
            "ring3": ring3,
            "swapnot_drive": swapnot_drive,
        }
    return fams


def random_function(rng, n):
    return [rng.randrange(2**n) for _ in range(2**n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--eps", type=float, nargs="*", default=[0.0, 0.1])
    ap.add_argument("--cap", type=int, default=1500)
    ap.add_argument("--max-constituents", type=int, default=2)
    ap.add_argument("--run-label", default="")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    bounds = SearchBounds(max_update_grain=2, max_constituents=args.max_constituents)
    results = []
    hits = []

    for name, fn in structured_family(args.n).items():
        table = [fn(r) for r in range(2**args.n)]
        for eps in args.eps:
            tpm = noised(tpm_from_fn(lambda r, table=table: table[r], args.n), eps)
            history = consecutive_pair(table, args.n)
            out = run_one(f"struct:{name}:eps={eps}", tpm, history, bounds, args.cap)
            results.append(out)
            if out.get("HIT"):
                hits.append(out["label"])
            print(
                out["label"],
                "HIT" if out.get("HIT") else "",
                out.get("winners", out.get("skipped")),
                flush=True,
            )

    for trial in range(args.trials):
        table = random_function(rng, args.n)
        for eps in args.eps:
            tpm = noised(tpm_from_fn(lambda r, table=table: table[r], args.n), eps)
            history = consecutive_pair(table, args.n, start=rng.randrange(2**args.n))
            out = run_one(f"rand:{trial}:eps={eps}", tpm, history, bounds, args.cap)
            out["fn_table"] = table
            results.append(out)
            if out.get("HIT"):
                hits.append(out["label"])
            marker = (
                "HIT!" if out.get("HIT") else ("near-tie" if out.get("NEAR_TIE") else "")
            )
            print(
                out["label"],
                marker,
                f"top_g2={out.get('top_grain2_phi')}",
                f"top={out.get('top_phi')}",
                flush=True,
            )

    stem = f"prospect_n{args.n}_seed{args.seed}_trials{args.trials}_mc{args.max_constituents}"
    if args.run_label:
        stem += f"_{args.run_label}"
    path = OUTDIR / f"{stem}.json"
    v = 2
    while path.exists():
        path = OUTDIR / f"{stem}_v{v}.json"
        v += 1
    path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "n": args.n,
                "trials": args.trials,
                "eps": args.eps,
                "max_constituents": args.max_constituents,
                "preset": "iit4_2023",
                "hits": hits,
                "near_ties": [r["label"] for r in results if r.get("NEAR_TIE")],
                "results": results,
            },
            indent=1,
        )
    )
    print(
        f"\nWROTE {path}  hits={len(hits)} near_ties={sum(1 for r in results if r.get('NEAR_TIE'))}"
    )


if __name__ == "__main__":
    main()
