import tqdm
import sys
import logging


class TqdmHandler(logging.StreamHandler):
    """
    Logging handler which uses the tqdm write function to not break progress
    bars.
    """
    def emit (self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg, file=self.stream)
            self.flush()
        except Exception:
            self.handleError(record)
