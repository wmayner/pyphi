# -*- coding: utf-8 -*-
# network_generator/unit_functions.py

from . import utils


def logical_or_function(element, weights, state):
    return utils.input_weight(element, weights, state) >= 1


def logical_and_function(element, weights, state):
    # Convention: i,j means i -> j
    num_inputs = (weights[:, element] > 0).sum()
    return utils.input_weight(element, weights, state) >= num_inputs


def logical_parity_function(element, weights, state):
    return utils.input_weight(element, weights, state) % 2 >= 1


def logical_nor_function(element, weights, state):
    return not (logical_or_function(element, weights, state))


def logical_nand_function(element, weights, state):
    return not (logical_and_function(element, weights, state))


def logical_nparity_function(element, weights, state):
    return not (logical_parity_function(element, weights, state))
