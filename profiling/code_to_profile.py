#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import sys
import json
import pickle
import cProfile
from time import time
pyphidir = os.path.abspath('..')
if pyphidir not in sys.path:
    sys.path.insert(0, pyphidir)
import pyphi
from joblib import Parallel, delayed


formatter = logging.Formatter(
    fmt='%(asctime)s [%(name)s] %(levelname)s: %(message)s')


def json2pyphi(network):
    tpm = network['tpm']
    current_state = network['currentState']
    past_state = network['pastState']
    cm = network['connectivityMatrix']
    network = pyphi.Network(tpm, current_state, past_state,
                            connectivity_matrix=cm)
    return network


network_types = [
    'AND-circle',
    'MAJ-specialized',
    'MAJ-complete',
    'iit-3.0-modular'
]
network_sizes = range(5, 9)
network_files = []
for n in network_sizes:
    for t in network_types:
        network_files.append('{}-{}'.format(n, t))


def profile_network(filename):
    log = logging.getLogger(filename)
    handler = logging.FileHandler('logs/' + filename + '.log')
    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.setLevel(logging.INFO)

    try:
        with open('networks/' + filename + '.json') as f:

            net = json2pyphi(json.load(f))

            print(''.center(72, '-'))
            log.info('Profiling {}...'.format(filename))
            log.info('PyPhi configuration:\n' +
                     pyphi.config.get_config_string())

            start = time()
            pr = cProfile.Profile()
            pr.enable()

            results = tuple(pyphi.compute.complexes(net))

            pr.disable()
            end = time()

            pr.dump_stats('pstats/' + filename + '.pstats')

            log.info('Finished in {} seconds.'.format(end - start))
            with open('pyphi_results/' + filename + '-results.pkl', 'wb') as f:
                pickle.dump(results, f)
    except Exception as e:
        log.error(e)
        pass

Parallel(n_jobs=(-5), verbose=20)(
    delayed(profile_network)(filename) for filename in network_files)
