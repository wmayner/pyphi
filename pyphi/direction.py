#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# direction.py

'''
Causal directions.
'''

from enum import Enum


class Direction(Enum):
    '''Constant that parametrizes cause and effect methods.

    Accessed using ``Direction.PAST`` and ``Direction.FUTURE``, etc.
    '''
    PAST = 0
    FUTURE = 1
    BIDIRECTIONAL = 2

    def __str__(self):
        if self is Direction.PAST:
            return 'PAST'
        elif self is Direction.FUTURE:
            return 'FUTURE'
        elif self is Direction.BIDIRECTIONAL:
            return 'BIDIRECTIONAL'

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
        if self is Direction.PAST:
            return purview, mechanism
        elif self is Direction.FUTURE:
            return mechanism, purview

        from . import validate
        validate.direction(self)
