# parallel/backends/__init__.py
"""Backend implementations for parallel computation."""

from .local_process import LocalMapReduce

__all__ = ["LocalMapReduce"]
