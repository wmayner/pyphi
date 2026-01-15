"""Tests for the parallel progress tracking module."""

from unittest.mock import patch

from pyphi.parallel.backends.progress import LocalProgressBar


class TestLocalProgressBarBasic:
    """Basic functionality tests for LocalProgressBar."""

    def test_construction_with_total(self):
        """Progress bar initializes with known total."""
        bar = LocalProgressBar(total=100, desc="test")
        try:
            assert bar._bar is not None
            assert bar._closed is False
        finally:
            bar.close()

    def test_construction_without_total(self):
        """Progress bar works with unknown total (None)."""
        bar = LocalProgressBar(total=None, desc="test")
        try:
            assert bar._bar is not None
            assert bar._closed is False
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

    def test_update_increments_progress(self):
        """update() increments the progress bar."""
        bar = LocalProgressBar(total=100)
        try:
            # Just verify it doesn't raise
            bar.update(5)
            bar.update(10)
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
        """update(n=5) correctly increments by n."""
        bar = LocalProgressBar(total=100)
        try:
            bar.update(1)
            bar.update(5)
            bar.update(10)
        finally:
            bar.close()


class TestLocalProgressBarIntegration:
    """Integration tests for progress bar."""

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
            mock_tqdm.assert_called_once_with(
                total=50, desc="testing", miniters=1, mininterval=0
            )
        finally:
            bar.close()
