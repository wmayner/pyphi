#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
After running this code, either
- Use `python -m pstats [file.pstats]` for an interactive pstats prompt.
- Use the `makecallgraph` script to visualize the call graph.
"""

import logging
import os
import sys
import json
import pickle
import cProfile
import argparse
from time import time
pyphidir = os.path.abspath('..')
if pyphidir not in sys.path:
    sys.path.insert(0, pyphidir)
import pyphi
from joblib import Parallel, delayed


formatter = logging.Formatter(
    fmt='%(asctime)s [%(name)s] %(levelname)s: %(message)s')

PSTATS = 'pstats'
LOGS = 'logs'
RESULTS = 'results'
NETWORKS = 'networks'


def json2pyphi(network):
    """Load a network from a json file"""
    tpm = network['tpm']
    current_state = network['currentState']
    cm = network['connectivityMatrix']
    network = pyphi.Network(tpm, connectivity_matrix=cm)
    return (network, current_state)


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

    Saves PyPhi results, pstats, and logs to respective directories
    """
    log = logging.getLogger(filename)
    handler = logging.FileHandler(LOGS + '/' + filename + '.log')
    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.setLevel(logging.INFO)

    try:
        with open(NETWORKS + '/' + filename + '.json') as f:

            (network, state) = json2pyphi(json.load(f))

            print(''.center(72, '-'))
            log.info('Profiling {}...'.format(filename))
            log.info('PyPhi configuration:\n' +
                     pyphi.config.get_config_string())

            start = time()
            pr = cProfile.Profile()
            pr.enable()

            results = tuple(pyphi.compute.complexes(network, state))

            pr.disable()
            end = time()

            pr.dump_stats(PSTATS + '/' + filename + '.pstats')

            log.info('Finished in {} seconds.'.format(end - start))
            with open(RESULTS + '/' + filename + '-results.pkl', 'wb') as f:
                pickle.dump(results, f)
    except Exception as e:
        log.error(e)
        pass


def ensure_dir(dirname):
    """Make a directory if it does not already exist"""
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


if __name__ == "__main__":
    # Setup directories
    ensure_dir(PSTATS)
    ensure_dir(LOGS)
    ensure_dir(RESULTS)

    # Parse arguments
    parser = argparse.ArgumentParser(description=(
        "Program to profile PyPhi on sample networks. \n\n"
        "After running this code, either\n"
        " - Use `python -m pstats [file.pstats]` for an interactive "
        "pstats prompt.\n"
        " - Use the `makecallgraph` script to visualize the call graph. "),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('network_file', nargs='*', help=(
        "The networks to profile, e.g. '5-AND-circle'. "
        "Defaults to all networks."))
    args = parser.parse_args()

    # Networks to profile
    if args.network_file:
        network_files = args.network_file
    else:
        network_files = all_network_files()

    # Do it
    Parallel(n_jobs=(-5), verbose=20)(
        delayed(profile_network)(filename) for filename in network_files)
