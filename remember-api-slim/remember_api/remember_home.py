"""Resolve the local Remember data directory."""

import os
from pathlib import Path


def remember_home() -> Path:
    override = os.environ.get("REMEMBER_HOME", "").strip()
    return Path(override).resolve() if override else Path.home() / ".remember"
