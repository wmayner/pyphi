#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Profiling tool for PyPhi.
"""

import argparse
import cProfile
import json
import logging
import os
import pickle
import sys
from time import time

from joblib import Parallel, delayed

import pyphi

pyphidir = os.path.abspath('..')
if pyphidir not in sys.path:
    sys.path.insert(0, pyphidir)


formatter = logging.Formatter(
    fmt='%(asctime)s [%(name)s] %(levelname)s: %(message)s')

PSTATS = 'pstats'
LOGS = 'logs'
RESULTS = 'results'
NETWORKS = 'networks'


def load_json_network(json_dict):
    """Load a network from a json file"""
    network = pyphi.Network.from_json(json_dict['network'])
    state = json_dict['state']
    return (network, state)


def all_network_files():
    """All network files"""
    # TODO: list explicitly since some are missing?
    network_types = [
        'AND-circle',
        'MAJ-specialized',
        'MAJ-complete',
        'iit-3.0-modular'
    ]
    network_sizes = range(5, 8)
    network_files = []
    for n in network_sizes:
        for t in network_types:
            network_files.append('{}-{}'.format(n, t))
    return network_files


def profile_network(filename):
    """Profile a network.

    Saves PyPhi results, pstats, and logs to respective directories.
    """
    log = logging.getLogger(filename)
    logfile = os.path.join(LOGS, filename + '.log')
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    handler = logging.FileHandler(logfile)
    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.setLevel(logging.INFO)

    try:
        with open(os.path.join(NETWORKS, filename + '.json')) as f:

            network, state = load_json_network(json.load(f))

            log.info('Profiling %s...', filename)
            log.info('PyPhi configuration:\n%s',
                     pyphi.config.get_config_string())

            start = time()
            pr = cProfile.Profile()
            pr.enable()

            results = tuple(pyphi.compute.complexes(network, state))

            pr.disable()
            end = time()

            pstatsfile = os.path.join(PSTATS, filename + '.pstats')
            os.makedirs(os.path.dirname(pstatsfile), exist_ok=True)
            pr.dump_stats(pstatsfile)

            log.info('Finished in %i seconds.', end - start)

            resultfile = os.path.join(RESULTS, filename + '-results.pkl')
            os.makedirs(os.path.dirname(resultfile), exist_ok=True)
            with open(resultfile, 'wb') as f:
                pickle.dump(results, f)
    except Exception as e:
        log.error(e)
        raise e


if __name__ == "__main__":
    # Setup directories
    os.makedirs(PSTATS, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)

    # Parse arguments
    parser = argparse.ArgumentParser(description=(
        "Program to profile PyPhi on sample networks. \n\n"
        "After running this code, either\n"
        " - Use `python -m pstats [file.pstats]` for an interactive "
        "pstats prompt.\n"
        " - Use `loadprofile.sh [file.pstats] && print_stats.py` to print "
        "the most offensive functions.\n"
        " - Use the `makecallgraph` script to visualize the call graph.\n\n"
        "For the most descriptive results, disable any parallelization in "
        "PyPhi."))
    parser.add_argument('network_file', nargs='?', help=(
        "The network to profile, e.g. '5-AND-circle'."
        "Defaults to all networks."))
    parser.add_argument('-p', nargs='?',
                        help=('Profile networks in parallel.'),
                        default=False)
    args = parser.parse_args()

    # Network to profile
    if args.network_file:
        network_files = [args.network_file]
    else:
        network_files = all_network_files()

    # Do it
    if len(network_files) > 1:
        Parallel(n_jobs=(-5), verbose=20)(
            delayed(profile_network)(filename) for filename in network_files)
    else:
        for filename in network_files:
            profile_network(filename)
