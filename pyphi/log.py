#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# logging.py

'''Utilities for logging and progress bars.'''

import logging
import threading

import tqdm


# pylint: disable=arguments-differ
class ProgressBar(tqdm.tqdm):
    '''Thread safe progress-bar wrapper around ``tqdm``.'''

    _lock = threading.RLock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            return super().__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        with self._lock:
            super().__init__(*args, **kwargs)

    @classmethod
    def write(cls, *args, **kwargs):
        with cls._lock:
            super().write(*args, **kwargs)

    def update(self, *args, **kwargs):
        with self._lock:
            super().update(*args, **kwargs)

    def close(self):
        with self._lock:
            super().close()


class ProgressBarHandler(logging.StreamHandler):
    '''Logging handler that writes through ``tqdm`` in order to not break
    progress bars.
    '''
    def emit(self, record):
        try:
            msg = self.format(record)
            ProgressBar.write(msg, file=self.stream)
            self.flush()
        except Exception:  # pylint: disable=broad-except
            self.handleError(record)
