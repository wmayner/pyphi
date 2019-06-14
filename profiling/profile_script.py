#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Profiling tool for PyPhi."""

import argparse
import cProfile
from pathlib import Path
from time import time

import pyphi


def profile_file(path):
    """Profile the execution of the script at ``path``."""
    start = time()
    pr = cProfile.Profile()
    pr.enable()

    exec(path.read_text())

    pr.disable()
    end = time()

    pstatsfile = path.with_suffix(".pstats")
    pr.dump_stats(pstatsfile)

    elapsed = round(end - start, 2)
    print(f"Finished in {elapsed} seconds.")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        description=(
            "Program to profile PyPhi on sample networks. \n\n"
            "After running this code, either\n"
            " - Use `python -m pstats [file.pstats]` for an interactive "
            "pstats prompt.\n"
            " - Use `loadprofile.sh [file.pstats] && print_stats.py` to print "
            "the most offensive functions.\n"
            " - Use the `makecallgraph` script to visualize the call graph.\n\n"
            "For the most descriptive results, disable any parallelization in "
            "PyPhi."
        )
    )
    parser.add_argument("script", nargs="?", help="the script to profile.")
    args = parser.parse_args()

    profile_file(Path(args.script))
