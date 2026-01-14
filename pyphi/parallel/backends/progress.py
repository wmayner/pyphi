# parallel/backends/progress.py
"""Thread-safe progress tracking for parallel computation."""

from __future__ import annotations

import contextlib
import multiprocessing
import threading
from typing import TYPE_CHECKING
from typing import Any

from tqdm.auto import tqdm

from pyphi.conf import fallback

if TYPE_CHECKING:
    from multiprocessing import Queue


class LocalProgressBar:
    """Thread-safe progress tracking using multiprocessing Queue.

    Uses a multiprocessing-safe queue for progress updates from worker
    processes. The progress bar runs in a background thread that consumes
    updates from the queue, allowing worker processes to update progress
    without blocking.
    """

    def __init__(self, total: int | None = None, desc: str = ""):
        """Initialize the progress bar.

        Args:
            total: Total number of items to process (None for unknown).
            desc: Description to display on the progress bar.
        """
        self._manager = multiprocessing.Manager()
        # Manager().Queue() returns a proxy, not a real Queue, but has the same interface
        self._queue: Queue[Any] = self._manager.Queue()  # pyright: ignore[reportAttributeAccessIssue]
        self._total = total
        self._desc = desc
        self._bar: tqdm | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._started = False
        self._closed = False

        # Start the progress bar
        self._start()

    @property
    def queue(self) -> Queue:
        """Return the progress queue for worker processes."""
        return self._queue

    def _start(self) -> None:
        """Start the progress bar and background update thread."""
        if self._started:
            return

        self._bar = tqdm(total=self._total, desc=self._desc)
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()
        self._started = True

    def _update_loop(self) -> None:
        """Background thread that consumes updates from the queue."""
        total = fallback(self._total, float("inf"))
        counter = 0

        while not self._stop_event.is_set():
            try:
                # Non-blocking get with timeout to check stop event
                delta = self._queue.get(timeout=0.1)
                if delta == "FINISH":
                    break
                counter += delta
                if self._bar is not None:
                    self._bar.update(delta)
                if counter >= total:
                    break
            except Exception:
                # Queue empty or other error, continue
                continue

    def update(self, n: int = 1) -> None:
        """Update progress by n items.

        Thread-safe method that can be called from any thread or process.

        Args:
            n: Number of items completed.
        """
        if not self._closed:
            with contextlib.suppress(Exception):
                self._queue.put_nowait(n)

    def close(self) -> None:
        """Close the progress bar and clean up resources."""
        if self._closed:
            return

        self._closed = True
        self._stop_event.set()

        # Send finish signal
        with contextlib.suppress(Exception):
            self._queue.put_nowait("FINISH")

        # Wait for thread to finish
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        # Close the progress bar
        if self._bar is not None:
            self._bar.close()

        # Shutdown the manager
        with contextlib.suppress(Exception):
            self._manager.shutdown()

    def __enter__(self) -> LocalProgressBar:
        """Context manager entry."""
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.close()
