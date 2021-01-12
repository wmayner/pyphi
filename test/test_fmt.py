#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test/test_fmt.py

from pyphi import Direction


def test_fmt_mice(s):
    mice = s.find_mice(Direction.CAUSE, (2,))
    repr(mice)
    str(mice)
