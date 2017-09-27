#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# constants.py

'''
Package-wide constants.
'''

import pickle
from enum import Enum

import joblib

from . import config


class Direction(Enum):
    '''Constants that parametrize cause and effect methods.

    Accessed using ``Direction.PAST`` and ``Direction.FUTURE``, etc.
    '''
    PAST = 0
    FUTURE = 1
    BIDIRECTIONAL = 2

    def to_json(self):
        return {'direction': self.value}

    @classmethod
    def from_json(cls, dct):
        return cls(dct['direction'])

    def order(self, mechanism, purview):
        '''Order the mechanism and purview in time.

        If the direction is ``PAST``, then the ``purview`` is at |t-1| and the
        ``mechanism`` is at time |t|. If the direction is ``FUTURE``, then
        the ``mechanism`` is at time |t| and the purview is at |t+1|.
        '''
        if self is self.PAST:
            return purview, mechanism
        elif self is self.FUTURE:
            return mechanism, purview

        from . import validate
        validate.direction(self)


#: The threshold below which we consider differences in phi values to be zero.
EPSILON = 10 ** - config.PRECISION

#: Label for the filesystem cache backend.
FILESYSTEM = 'fs'

#: Label for the MongoDB cache backend.
DATABASE = 'db'

#: The protocol used for pickling objects.
PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL

#: The joblib ``Memory`` object for persistent caching without a database.
joblib_memory = joblib.Memory(cachedir=config.FS_CACHE_DIRECTORY,
                              verbose=config.FS_CACHE_VERBOSITY)

#: Node states
OFF = (0,)
ON = (1,)
