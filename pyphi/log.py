#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# log.py

"""Utilities for logging and progress bars."""

import logging

from tqdm import tqdm


class TqdmHandler(logging.StreamHandler):
    """Logging handler that writes through ``tqdm`` in order to not break
    progress bars.
    """
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=self.stream, end=self.terminator)
            self.flush()
        except Exception:  # pylint: disable=broad-except
            self.handleError(record)
