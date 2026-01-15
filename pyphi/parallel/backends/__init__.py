# parallel/backends/__init__.py
"""Backend implementations for parallel computation."""

from .local import LocalMapReduce

__all__ = ["LocalMapReduce"]
