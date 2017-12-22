#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test/test_distribution.py

import numpy as np

from pyphi import distribution
from pyphi.utils import powerset


def test_normalize():
    x = np.array([2, 4, 2])
    assert np.array_equal(distribution.normalize(x),
                          np.array([.25, .5, .25]))
    x = np.array([[0, 4], [2, 2]])
    assert np.array_equal(distribution.normalize(x),
                          np.array([[0, .5], [.25, .25]]))
    x = np.array([0, 0])
    assert np.array_equal(distribution.normalize(x),
                          np.array([0, 0]))


def test_uniform_distribution():
    assert np.array_equal(distribution.uniform_distribution(3),
                          (np.ones(8) / 8).reshape([2] * 3))


def test_purview_max_entropy_distribution():
    max_ent = distribution.max_entropy_distribution((0, 1), 3)
    assert max_ent.shape == (2, 2, 1)
    assert np.array_equal(max_ent,
                          (np.ones(4) / 4).reshape((2, 2, 1)))
    assert max_ent[0][1][0] == 0.25


def test_marginal_zero():
    repertoire = np.array([
        [[0., 0.],
         [0., 0.5]],
        [[0., 0.],
         [0., 0.5]]])
    assert distribution.marginal_zero(repertoire, 0) == 0.5
    assert distribution.marginal_zero(repertoire, 1) == 0
    assert distribution.marginal_zero(repertoire, 2) == 0


def test_marginal():
    repertoire = np.array([
        [[0., 0.],
         [0., 0.5]],
        [[0., 0.],
         [0., 0.5]]])
    assert np.array_equal(
        distribution.marginal(repertoire, 0), np.array([[[0.5]], [[0.5]]]))
    assert np.array_equal(
        distribution.marginal(repertoire, 1), np.array([[[0], [1]]]))
    assert np.array_equal(
        distribution.marginal(repertoire, 2), np.array([[[0, 1]]]))


def test_independent():
    repertoire = np.array([
        [[0.25],
         [0.25]],
        [[0.25],
         [0.25]]])
    assert distribution.independent(repertoire)

    repertoire = np.array([
        [[0.5],
         [0.0]],
        [[0.0],
         [0.5]]])
    assert not distribution.independent(repertoire)


def test_purview_size(s):
    mechanisms = powerset(s.node_indices)
    purviews = powerset(s.node_indices)

    for mechanism, purview in zip(mechanisms, purviews):
        repertoire = s.cause_repertoire(mechanism, purview)
        assert distribution.purview_size(repertoire) == len(purview)


def test_purview(s):
    mechanisms = powerset(s.node_indices)
    purviews = powerset(s.node_indices)

    for mechanism, purview in zip(mechanisms, purviews):
        repertoire = s.cause_repertoire(mechanism, purview)
        assert distribution.purview(repertoire) == purview

    assert distribution.purview(None) is None


def test_repertoire_shape():
    N = 3
    assert distribution.repertoire_shape((), N) == [1, 1, 1]
    assert distribution.repertoire_shape((1, 2), N) == [1, 2, 2]
    assert distribution.repertoire_shape((0, 2), N) == [2, 1, 2]


def test_flatten():
    repertoire = np.array([
        [[0.1, 0.0]],
        [[0.2, 0.7]]])

    assert np.array_equal(distribution.flatten(repertoire),
                          [0.1, 0.2, 0.0, 0.7])
    assert np.array_equal(distribution.flatten(repertoire, big_endian=True),
                          [0.1, 0.0, 0.2, 0.7])
    assert distribution.flatten(None) is None
