#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prints the pstats results for the functions in which the most time was spent.
"""

import argparse
import pstats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pstats_file", help="the profile file to print")
    args = parser.parse_args()

    p = pstats.Stats(args.pstats_file)
    p.strip_dirs()
    p.sort_stats("tottime", "calls", "name")
    p.print_stats(25)
