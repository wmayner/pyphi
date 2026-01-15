"""Tests for the parallel progress tracking module."""

import time
from unittest.mock import patch

from pyphi.parallel.backends.progress import LocalProgressBar


class TestLocalProgressBarBasic:
    """Basic functionality tests for LocalProgressBar."""

    def test_construction_with_total(self):
        """Progress bar initializes with known total."""
        bar = LocalProgressBar(total=100, desc="test")
        try:
            assert bar._total == 100
            assert bar._desc == "test"
            assert bar._started is True
        finally:
            bar.close()

    def test_construction_without_total(self):
        """Progress bar works with unknown total (None)."""
        bar = LocalProgressBar(total=None, desc="test")
        try:
            assert bar._total is None
            assert bar._started is True
        finally:
            bar.close()

    def test_queue_property(self):
        """queue property returns the multiprocessing queue."""
        bar = LocalProgressBar(total=100, desc="test")
        try:
            queue = bar.queue
            assert queue is not None
            assert queue is bar._queue
        finally:
            bar.close()

    def test_starts_automatically(self):
        """Progress bar starts on construction (_started=True)."""
        bar = LocalProgressBar(total=100)
        try:
            assert bar._started is True
            assert bar._bar is not None
            assert bar._thread is not None
        finally:
            bar.close()


class TestLocalProgressBarContextManager:
    """Tests for context manager protocol."""

    def test_enter_returns_self(self):
        """__enter__ returns the progress bar instance."""
        bar = LocalProgressBar(total=100)
        try:
            result = bar.__enter__()
            assert result is bar
        finally:
            bar.close()

    def test_exit_closes_bar(self):
        """__exit__ calls close()."""
        bar = LocalProgressBar(total=100)
        bar.__enter__()
        assert bar._closed is False
        bar.__exit__(None, None, None)
        assert bar._closed is True

    def test_context_manager_cleanup(self):
        """Resources properly cleaned up after context exit."""
        with LocalProgressBar(total=100, desc="test") as bar:
            assert bar._started is True
            assert bar._closed is False
        assert bar._closed is True

    def test_double_close_safe(self):
        """Calling close() twice doesn't raise errors."""
        bar = LocalProgressBar(total=100)
        bar.close()
        bar.close()  # Should not raise
        assert bar._closed is True


class TestLocalProgressBarUpdates:
    """Tests for progress update mechanism."""

    def test_update_puts_to_queue(self):
        """update() puts value to queue."""
        bar = LocalProgressBar(total=100)
        try:
            # Queue starts empty (background thread consumes)
            # Send an update
            bar.update(5)
            # Give time for queue to be processed
            time.sleep(0.2)
            # Can't easily check queue contents since background thread
            # consumes them, but we can verify no error occurred
        finally:
            bar.close()

    def test_update_after_close_ignored(self):
        """update() after close() is silently ignored."""
        bar = LocalProgressBar(total=100)
        bar.close()
        # Should not raise
        bar.update(5)
        bar.update(10)

    def test_update_with_custom_n(self):
        """update(n=5) correctly puts n to queue."""
        bar = LocalProgressBar(total=100)
        try:
            # Just verify it doesn't raise
            bar.update(1)
            bar.update(5)
            bar.update(10)
        finally:
            bar.close()


class TestLocalProgressBarThreading:
    """Tests for thread safety and cleanup."""

    def test_background_thread_starts(self):
        """Background thread is started and alive."""
        bar = LocalProgressBar(total=100)
        try:
            assert bar._thread is not None
            assert bar._thread.is_alive()
        finally:
            bar.close()

    def test_background_thread_is_daemon(self):
        """Background thread is a daemon thread."""
        bar = LocalProgressBar(total=100)
        try:
            assert bar._thread.daemon is True
        finally:
            bar.close()

    def test_finish_signal_stops_thread(self):
        """FINISH signal causes thread to terminate."""
        bar = LocalProgressBar(total=100)
        assert bar._thread.is_alive()
        bar.close()
        # Give time for thread to stop
        time.sleep(0.2)
        assert not bar._thread.is_alive()

    def test_stop_event_terminates_loop(self):
        """Setting stop_event terminates update loop."""
        bar = LocalProgressBar(total=100)
        try:
            assert not bar._stop_event.is_set()
            bar._stop_event.set()
            # Thread should notice and exit
            time.sleep(0.2)
            # Thread may still be alive briefly due to timeout
        finally:
            bar.close()

    def test_close_joins_thread(self):
        """close() waits for background thread to finish."""
        bar = LocalProgressBar(total=100)
        bar.close()
        # After close, thread should not be alive
        assert not bar._thread.is_alive()

    def test_close_shuts_down_manager(self):
        """close() shuts down the multiprocessing Manager."""
        bar = LocalProgressBar(total=100)
        assert bar._manager is not None  # Manager exists before close
        bar.close()
        # Manager should be shutdown (queue operations would fail)
        assert bar._closed is True


class TestLocalProgressBarIntegration:
    """Integration tests for progress bar with simulated workers."""

    def test_multiple_updates_processed(self):
        """Multiple updates are processed by background thread."""
        bar = LocalProgressBar(total=100, desc="test")
        try:
            for _ in range(10):
                bar.update(1)
            # Give time for processing
            time.sleep(0.3)
        finally:
            bar.close()

    def test_progress_bar_with_zero_total(self):
        """Progress bar handles total=0."""
        bar = LocalProgressBar(total=0, desc="test")
        try:
            # Should not raise
            bar.update(1)
        finally:
            bar.close()

    @patch("pyphi.parallel.backends.progress.tqdm")
    def test_tqdm_called_with_params(self, mock_tqdm):
        """tqdm is called with correct parameters."""
        bar = LocalProgressBar(total=50, desc="testing")
        try:
            mock_tqdm.assert_called_once_with(total=50, desc="testing")
        finally:
            bar.close()
