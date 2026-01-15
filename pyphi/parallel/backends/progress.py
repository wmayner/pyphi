# parallel/backends/progress.py
"""Progress tracking for parallel computation."""

from __future__ import annotations

from tqdm.auto import tqdm


class LocalProgressBar:
    """Simple progress bar wrapper around tqdm.

    Provides a consistent interface for progress tracking that works
    in both terminal and Jupyter notebook environments.
    """

    def __init__(self, total: int | None = None, desc: str = ""):
        """Initialize the progress bar.

        Args:
            total: Total number of items to process (None for unknown).
            desc: Description to display on the progress bar.
        """
        # miniters=1 ensures every update is displayed (no skipping)
        # mininterval=0 allows immediate refresh
        self._bar: tqdm | None = tqdm(total=total, desc=desc, miniters=1, mininterval=0)
        self._closed = False

    def update(self, n: int = 1) -> None:
        """Update progress by n items.

        Args:
            n: Number of items completed.
        """
        if not self._closed and self._bar is not None:
            self._bar.update(n)
            self._bar.refresh()  # Force immediate display update

    def close(self) -> None:
        """Close the progress bar."""
        if self._closed:
            return
        self._closed = True
        if self._bar is not None:
            self._bar.close()

    def __enter__(self) -> LocalProgressBar:
        """Context manager entry."""
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.close()
