#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pyphi

tpm = np.array([[1, 0, 0],
                [1, 1, 0],
                [0, 0, 0],
                [0, 1, 1],
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 1]])

cm = np.array([[0, 1, 1],
               [1, 0, 1],
               [1, 1, 0]])

pv = np.array([0.75, 0.75, 0.25])

current_state = (0, 1, 0)
past_state = (0, 0, 1)

network = pyphi.Network(tpm, current_state, past_state, connectivity_matrix=cm,
                        perturb_vector=pv)

subsystem = pyphi.Subsystem(range(network.size), network)

A = (subsystem.nodes[0],)
B = (subsystem.nodes[1],)
C = (subsystem.nodes[2],)
AB = subsystem.nodes[0:2]
AC = subsystem.indices2nodes([0, 2])
BC = subsystem.nodes[1:3]
ABC = subsystem.nodes[:]


def test_perturb_unconstrained_cause_repertoire():
    assert np.all(subsystem.cause_repertoire((), ABC).flatten(order='F')
                  == np.array([3, 9, 9, 27, 1, 3, 3, 9]) / 64)

    assert np.all(subsystem.cause_repertoire((), AB).flatten(order='F')
                  == np.array([1, 3, 3, 9]) / 16)

    assert np.all(subsystem.cause_repertoire((), AC).flatten(order='F')
                  == np.array([3, 9, 1, 3]) / 16)

    assert np.all(subsystem.cause_repertoire((), BC).flatten(order='F')
                  == np.array([3, 9, 1, 3]) / 16)

    assert np.all(subsystem.cause_repertoire((), A).flatten(order='F')
                  == np.array([1, 3]) / 4)

    assert np.all(subsystem.cause_repertoire((), B).flatten(order='F')
                  == np.array([1, 3]) / 4)

    assert np.all(subsystem.cause_repertoire((), C).flatten(order='F')
                  == np.array([3, 1]) / 4)


def test_perturb_unconstrained_effect_repertoire():
    assert np.all(
        subsystem.effect_repertoire((), ABC).flatten(order='F')
        == np.array([273, 63, 1183, 273, 351, 81, 1521, 351]) / 16 ** 3)

    assert np.all(subsystem.effect_repertoire((), AB).flatten(order='F')
                  == np.array([39, 9, 169, 39]) / 16 ** 2)

    assert np.all(subsystem.effect_repertoire((), AC).flatten(order='F')
                  == np.array([91, 21, 117, 27]) / 16 ** 2)

    assert np.all(subsystem.effect_repertoire((), BC).flatten(order='F')
                  == np.array([21, 91, 27, 117]) / 16 ** 2)

    assert np.all(subsystem.effect_repertoire((), A).flatten(order='F')
                  == np.array([13, 3]) / 16)

    assert np.all(subsystem.effect_repertoire((), B).flatten(order='F')
                  == np.array([3, 13]) / 16)

    assert np.all(subsystem.effect_repertoire((), C).flatten(order='F')
                  == np.array([7, 9]) / 16)


def test_perturb_constrained_effect_repertoire_A():
    assert np.all(subsystem.effect_repertoire(A, ABC).flatten(order='F')
                  == np.array([39, 9, 13, 3, 0, 0, 0, 0]) / 64)

    assert np.all(subsystem.effect_repertoire(A, AB).flatten(order='F')
                  == np.array([39, 9, 13, 3]) / 64)

    assert np.all(subsystem.effect_repertoire(A, AC).flatten(order='F')
                  == np.array([13, 3, 0, 0]) / 16)

    assert np.all(subsystem.effect_repertoire(A, BC).flatten(order='F')
                  == np.array([3, 1, 0, 0]) / 4)

    assert np.all(subsystem.effect_repertoire(A, A).flatten(order='F')
                  == np.array([13, 3]) / 16)

    assert np.all(subsystem.effect_repertoire(A, B).flatten(order='F')
                  == np.array([3, 1]) / 4)

    assert np.all(subsystem.effect_repertoire(A, C).flatten(order='F')
                  == np.array([1, 0]))


def test_perturb_constrained_effect_repertoire_B():
    assert np.all(subsystem.effect_repertoire(B, ABC).flatten(order='F')
                  == np.array([3, 0, 13, 0, 9, 0, 39, 0]) / 64)

    assert np.all(subsystem.effect_repertoire(B, AB).flatten(order='F')
                  == np.array([3, 0, 13, 0]) / 16)

    assert np.all(subsystem.effect_repertoire(B, AC).flatten(order='F')
                  == np.array([1, 0, 3, 0]) / 4)

    assert np.all(subsystem.effect_repertoire(B, BC).flatten(order='F')
                  == np.array([3, 13, 9, 39]) / 64)

    assert np.all(subsystem.effect_repertoire(B, A).flatten(order='F')
                  == np.array([1, 0]) / 1)

    assert np.all(subsystem.effect_repertoire(B, B).flatten(order='F')
                  == np.array([3, 13]) / 16)

    assert np.all(subsystem.effect_repertoire(B, C).flatten(order='F')
                  == np.array([1, 3]) / 4)


def test_perturb_constrained_effect_repertoire_C():
    assert np.all(subsystem.effect_repertoire(C, ABC).flatten(order='F')
                  == np.array([21, 7, 63, 21, 27, 9, 81, 27]) / 256)

    assert np.all(subsystem.effect_repertoire(C, AB).flatten(order='F')
                  == np.array([3, 1, 9, 3]) / 16)

    assert np.all(subsystem.effect_repertoire(C, AC).flatten(order='F')
                  == np.array([21, 7, 27, 9]) / 64)

    assert np.all(subsystem.effect_repertoire(C, BC).flatten(order='F')
                  == np.array([7, 21, 9, 27]) / 64)

    assert np.all(subsystem.effect_repertoire(C, A).flatten(order='F')
                  == np.array([3, 1]) / 4)

    assert np.all(subsystem.effect_repertoire(C, B).flatten(order='F')
                  == np.array([1, 3]) / 4)

    assert np.all(subsystem.effect_repertoire(C, C).flatten(order='F')
                  == np.array([7, 9]) / 16)


def test_perturb_constrained_effect_repertoire_AB():
    assert np.all(subsystem.effect_repertoire(AB, ABC).flatten(order='F')
                  == np.array([3, 0, 1, 0, 0, 0, 0, 0]) / 4)

    assert np.all(subsystem.effect_repertoire(AB, AB).flatten(order='F')
                  == np.array([3, 0, 1, 0]) / 4)

    assert np.all(subsystem.effect_repertoire(AB, AC).flatten(order='F')
                  == np.array([1, 0, 0, 0]))

    assert np.all(subsystem.effect_repertoire(AB, BC).flatten(order='F')
                  == np.array([3, 1, 0, 0]) / 4)

    assert np.all(subsystem.effect_repertoire(AB, A).flatten(order='F')
                  == np.array([1, 0]))

    assert np.all(subsystem.effect_repertoire(AB, B).flatten(order='F')
                  == np.array([3, 1]) / 4)

    assert np.all(subsystem.effect_repertoire(AB, C).flatten(order='F')
                  == np.array([1, 0]))


def test_perturb_constrained_effect_repertoire_AC():
    assert np.all(subsystem.effect_repertoire(AC, ABC).flatten(order='F')
                  == np.array([3, 1, 0, 0, 0, 0, 0, 0]) / 4)

    assert np.all(subsystem.effect_repertoire(AC, AB).flatten(order='F')
                  == np.array([3, 1, 0, 0]) / 4)

    assert np.all(subsystem.effect_repertoire(AC, AC).flatten(order='F')
                  == np.array([3, 1, 0, 0]) / 4)

    assert np.all(subsystem.effect_repertoire(AC, BC).flatten(order='F')
                  == np.array([1, 0, 0, 0]))

    assert np.all(subsystem.effect_repertoire(AC, A).flatten(order='F')
                  == np.array([3, 1]) / 4)

    assert np.all(subsystem.effect_repertoire(AC, B).flatten(order='F')
                  == np.array([1, 0]))

    assert np.all(subsystem.effect_repertoire(AC, C).flatten(order='F')
                  == np.array([1, 0]))


def test_perturb_constrained_effect_repertoire_BC():
    assert np.all(subsystem.effect_repertoire(BC, ABC).flatten(order='F')
                  == np.array([1, 0, 3, 0, 3, 0, 9, 0]) / 16)

    assert np.all(subsystem.effect_repertoire(BC, AB).flatten(order='F')
                  == np.array([1, 0, 3, 0]) / 4)

    assert np.all(subsystem.effect_repertoire(BC, AC).flatten(order='F')
                  == np.array([1, 0, 3, 0]) / 4)

    assert np.all(subsystem.effect_repertoire(BC, BC).flatten(order='F')
                  == np.array([1, 3, 3, 9]) / 16)

    assert np.all(subsystem.effect_repertoire(BC, A).flatten(order='F')
                  == np.array([1, 0]))

    assert np.all(subsystem.effect_repertoire(BC, B).flatten(order='F')
                  == np.array([1, 3]) / 4)

    assert np.all(subsystem.effect_repertoire(BC, C).flatten(order='F')
                  == np.array([1, 3]) / 4)


def test_perturb_constrained_effect_repertoire_ABC():
    assert np.all(subsystem.effect_repertoire(ABC, ABC).flatten(order='F')
                  == np.array([1, 0, 0, 0, 0, 0, 0, 0]))

    assert np.all(subsystem.effect_repertoire(ABC, AB).flatten(order='F')
                  == np.array([1, 0, 0, 0]))

    assert np.all(subsystem.effect_repertoire(ABC, AC).flatten(order='F')
                  == np.array([1, 0, 0, 0]))

    assert np.all(subsystem.effect_repertoire(ABC, BC).flatten(order='F')
                  == np.array([1, 0, 0, 0]))

    assert np.all(subsystem.effect_repertoire(ABC, A).flatten(order='F')
                  == np.array([1, 0]))

    assert np.all(subsystem.effect_repertoire(ABC, B).flatten(order='F')
                  == np.array([1, 0]))

    assert np.all(subsystem.effect_repertoire(ABC, C).flatten(order='F')
                  == np.array([1, 0]))


def test_perturb_constrained_cause_repertoire_A():
    assert np.all(subsystem.cause_repertoire(A, ABC).flatten(order='F')
                  == np.array([0, 0, 9, 27, 1, 3, 3, 9]) / 52)

    assert np.all(subsystem.cause_repertoire(A, AB).flatten(order='F')
                  == np.array([1, 3, 12, 36]) / 52)

    assert np.all(subsystem.cause_repertoire(A, AC).flatten(order='F')
                  == np.array([9, 27, 4, 12]) / 52)

    assert np.all(subsystem.cause_repertoire(A, BC).flatten(order='F')
                  == np.array([0, 36, 4, 12]) / 52)

    assert np.all(subsystem.cause_repertoire(A, A).flatten(order='F')
                  == np.array([13, 39]) / 52)

    assert np.all(subsystem.cause_repertoire(A, B).flatten(order='F')
                  == np.array([4, 48]) / 52)

    assert np.all(subsystem.cause_repertoire(A, C).flatten(order='F')
                  == np.array([36, 16]) / 52)


def test_perturb_constrained_cause_repertoire_B():
    assert np.all(subsystem.cause_repertoire(B, ABC).flatten(order='F')
                  == np.array([0, 9, 0, 27, 1, 3, 3, 9]) / 52)

    assert np.all(subsystem.cause_repertoire(B, AB).flatten(order='F')
                  == np.array([1, 12, 3, 36]) / 52)

    assert np.all(subsystem.cause_repertoire(B, AC).flatten(order='F')
                  == np.array([0, 36, 4, 12]) / 52)

    assert np.all(subsystem.cause_repertoire(B, BC).flatten(order='F')
                  == np.array([9, 27, 4, 12]) / 52)

    assert np.all(subsystem.cause_repertoire(B, A).flatten(order='F')
                  == np.array([4, 48]) / 52)

    assert np.all(subsystem.cause_repertoire(B, B).flatten(order='F')
                  == np.array([13, 39]) / 52)

    assert np.all(subsystem.cause_repertoire(B, C).flatten(order='F')
                  == np.array([36, 16]) / 52)


def test_perturb_constrained_cause_repertoire_C():
    assert np.all(subsystem.cause_repertoire(C, ABC).flatten(order='F')
                  == np.array([3, 9, 9, 0, 1, 3, 3, 0]) / 28)

    assert np.all(subsystem.cause_repertoire(C, AB).flatten(order='F')
                  == np.array([4, 12, 12, 0]) / 28)

    assert np.all(subsystem.cause_repertoire(C, AC).flatten(order='F')
                  == np.array([12, 9, 4, 3]) / 28)

    assert np.all(subsystem.cause_repertoire(C, BC).flatten(order='F')
                  == np.array([12, 9, 4, 3]) / 28)

    assert np.all(subsystem.cause_repertoire(C, A).flatten(order='F')
                  == np.array([16, 12]) / 28)

    assert np.all(subsystem.cause_repertoire(C, B).flatten(order='F')
                  == np.array([16, 12]) / 28)

    assert np.all(subsystem.cause_repertoire(C, C).flatten(order='F')
                  == np.array([21, 7]) / 28)


def test_perturb_constrained_cause_repertoire_AB():
    assert np.all(subsystem.cause_repertoire(AB, ABC).flatten(order='F')
                  == np.array([0, 0, 0, 27, 1, 3, 3, 9]) / 43)

    assert np.all(subsystem.cause_repertoire(AB, AB).flatten(order='F')
                  == np.array([1, 12, 12, 144]) / 169)

    assert np.all(subsystem.cause_repertoire(AB, AC).flatten(order='F')
                  == np.array([0, 108, 16, 48]) / 172)

    assert np.all(subsystem.cause_repertoire(AB, BC).flatten(order='F')
                  == np.array([0, 108, 16, 48]) / 172)

    assert np.all(subsystem.cause_repertoire(AB, A).flatten(order='F')
                  == np.array([52, 624]) / 676)

    assert np.all(subsystem.cause_repertoire(AB, B).flatten(order='F')
                  == np.array([52, 624]) / 676)

    assert np.all(subsystem.cause_repertoire(AB, C).flatten(order='F')
                  == np.array([432, 256]) / 688)


def test_perturb_constrained_cause_repertoire_AC():
    assert np.all(subsystem.cause_repertoire(AC, ABC).flatten(order='F')
                  == np.array([0, 0, 9, 0, 1, 3, 3, 0]) / 16)

    assert np.all(subsystem.cause_repertoire(AC, AB).flatten(order='F')
                  == np.array([4, 12, 48, 0]) / 64)

    assert np.all(subsystem.cause_repertoire(AC, AC).flatten(order='F')
                  == np.array([36, 27, 16, 12]) / 91)

    assert np.all(subsystem.cause_repertoire(AC, BC).flatten(order='F')
                  == np.array([0, 36, 16, 12]) / 64)

    assert np.all(subsystem.cause_repertoire(AC, A).flatten(order='F')
                  == np.array([208, 156]) / 364)

    assert np.all(subsystem.cause_repertoire(AC, B).flatten(order='F')
                  == np.array([64, 192]) / 256)

    assert np.all(subsystem.cause_repertoire(AC, C).flatten(order='F')
                  == np.array([252, 112]) / 364)


def test_perturb_constrained_cause_repertoire_BC():
    assert np.all(subsystem.cause_repertoire(BC, ABC).flatten(order='F')
                  == np.array([0, 9, 0, 0, 1, 3, 3, 0]) / 16)

    assert np.all(subsystem.cause_repertoire(BC, AB).flatten(order='F')
                  == np.array([4, 48, 12, 0]) / 64)

    assert np.all(subsystem.cause_repertoire(BC, AC).flatten(order='F')
                  == np.array([0, 36, 16, 12]) / 64)

    assert np.all(subsystem.cause_repertoire(BC, BC).flatten(order='F')
                  == np.array([36, 27, 16, 12]) / 91)

    assert np.all(subsystem.cause_repertoire(BC, A).flatten(order='F')
                  == np.array([64, 192]) / 256)

    assert np.all(subsystem.cause_repertoire(BC, B).flatten(order='F')
                  == np.array([208, 156]) / 364)

    assert np.all(subsystem.cause_repertoire(BC, C).flatten(order='F')
                  == np.array([252, 112]) / 364)


def test_perturb_constrained_cause_repertoire_ABC():
    assert np.all(subsystem.cause_repertoire(ABC, ABC).flatten(order='F')
                  == np.array([0, 0, 0, 0, 1, 3, 3, 0]) / 7)

    assert np.all(subsystem.cause_repertoire(ABC, AB).flatten(order='F')
                  == np.array([4, 48, 48, 0]) / 100)

    assert np.all(subsystem.cause_repertoire(ABC, AC).flatten(order='F')
                  == np.array([0, 108, 64, 48]) / 220)

    assert np.all(subsystem.cause_repertoire(ABC, BC).flatten(order='F')
                  == np.array([0, 108, 64, 48]) / 220)

    assert np.all(subsystem.cause_repertoire(ABC, A).flatten(order='F')
                  == np.array([832, 2496]) / 3328)

    assert np.all(subsystem.cause_repertoire(ABC, B).flatten(order='F')
                  == np.array([832, 2496]) / 3328)

    assert np.all(subsystem.cause_repertoire(ABC, C).flatten(order='F')
                  == np.array([3024, 1792]) / 4816)
