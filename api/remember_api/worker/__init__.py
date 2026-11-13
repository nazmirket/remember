"""
Background task processing for the local daemon.

This package provides:
- WorkerPoller: Polls PostgreSQL for pending tasks and executes them
"""

from .poller import WorkerPoller

__all__ = ["WorkerPoller"]
