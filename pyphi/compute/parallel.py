#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# compute/parallel.py

import logging
import multiprocessing
import threading

from tqdm import tqdm

from .. import config



def get_num_processes():
    """Return the number of processes to use in parallel."""
    cpu_count = multiprocessing.cpu_count()

    if config.NUMBER_OF_CORES == 0:
        raise ValueError(
            'Invalid NUMBER_OF_CORES; value may not be 0.')

    if config.NUMBER_OF_CORES > cpu_count:
        raise ValueError(
            'Invalid NUMBER_OF_CORES; value must be less than or '
            'equal to the available number of cores ({} for this '
            'system).'.format(cpu_count))

    if config.NUMBER_OF_CORES < 0:
        num = cpu_count + config.NUMBER_OF_CORES + 1
        if num <= 0:
            raise ValueError(
                'Invalid NUMBER_OF_CORES; negative value is too negative: '
                'requesting {} cores, {} available.'.format(num, cpu_count))

        return num

    return config.NUMBER_OF_CORES



# The worker configuration is done at the start of the worker process run.
def configure_worker(queue):
    config_worker = {
        'version': 1,
        'disable_existing_loggers': False,
        'handlers': {
            'queue': {
                'class': 'logging.handlers.QueueHandler',
                'queue': queue,
            },
        },
        'root': {
            'level': 'DEBUG',
            'handlers': ['queue']
        },
    }
    logging.config.dictConfig(config_worker)


def logger_thread(q):
    while True:
        record = q.get()
        if record is None:
            break
        logger = logging.getLogger(record.name)
        logger.handle(record)


POISON_PILL = None


class MapReduce:
    """
    Perform a computation over an iterable.

    This is similar to ``multiprocessing.Pool``, but allows computations to
    shortcircuit.

    Supports both parallel and sequential computations.
    """
    # *args are (subsystem, unpartitioned_constellation)
    def __init__(self, iterable, default_result, *context):
        self.iterable = iterable
        self.default_result = default_result
        self.context = context

    # TODO: should this not be a method? Is there a performance cost to
    # using a bound method as a Process?
    def worker(self, in_queue, out_queue, log_queue, *context):
        """Worker process."""
        configure_worker(log_queue)
        while True:
            obj = in_queue.get()
            if obj is POISON_PILL:
                break
            out_queue.put(self.compute(obj, *context))
        out_queue.put(POISON_PILL)

    def compute(self, obj, *context):
        """Computation handler.

        This method is given an obj from ``self.iterable`` to perform a
        computation over.
        """
        raise NotImplementedError

    def process_result(self, new_result, old_result):
        """Result handler.

        Every time a new result is computed by ``compute``, this method is
        called with this result and the previous (accumulated) result. This
        method compares or collates these two values, returning the new result.

        Setting ``self.working`` to ``False`` in this method will abort the
        remainder of the comnputation, returning this last result.
        """
        raise NotImplementedError

    def init_parallel(self):
        self.number_of_processes = get_num_processes()

        self.in_queue = multiprocessing.Queue()
        self.out_queue = multiprocessing.Queue()
        self.log_queue = multiprocessing.Queue()

        # Load all objects to perform the computation over
        for obj in self.iterable:
            self.in_queue.put(obj)

        for i in range(self.number_of_processes):
            self.in_queue.put(POISON_PILL)

        args = (self.in_queue, self.out_queue, self.log_queue) + self.context
        self.processes = [
            multiprocessing.Process(target=self.worker, args=args)
            for i in range(self.number_of_processes)]

        self.log_thread = threading.Thread(target=logger_thread,
                                           args=(self.log_queue,))

        # Initialize progress bar
        self.progress = tqdm(total=len(self.iterable), leave=False,
                        disable=(not config.PROGRESS_BARS),
                        desc='Evaluating \u03D5 cuts')

    def start_parallel(self):
        """Start all processses and the logger thread."""
        for process in self.processes:
            process.start()

        self.log_thread.start()

    def finish_parallel(self):
        """Terminate all processes."""
        # Remove the progress bar
        self.progress.close()

        for process in self.processes:
            process.terminate()

        # Shutdown the log thread
        self.log_queue.put(POISON_PILL)
        self.log_thread.join()

    def run_parallel(self):
        """Perform the computation in parallel, reading results from the output
        queue and passing them to ``process_result``.
        """
        self.init_parallel()
        self.start_parallel()
        self.working = True
        result = self.default_result

        while self.working:
            r = self.out_queue.get()
            if r is POISON_PILL:
                self.number_of_processes -= 1
                if self.number_of_processes == 0:
                    break
            else:
                result = self.process_result(r, result)
                self.progress.update(1)

        self.finish_parallel()

        return result

    def run_sequential(self):
        """Perform the computation sequentially, only holding two computed
        objects in memory at a time.
        """
        self.working = True
        result = self.default_result
        for cut in self.iterable:
            r = self.compute(cut, *self.context)
            result = self.process_result(r, result)
            self.progress.update(1)

            # Short-circuited?
            if not self.working:
                break

        return result

    def run(self, parallel=True):
        if parallel:
            return self.run_parallel()
        else:
            return self.run_sequential()
