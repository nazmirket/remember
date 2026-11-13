"""Common configuration for the built-in memory services."""


class Extension:
    """Hold the configuration supplied to a built-in service."""

    def __init__(self, config: dict[str, str]):
        self.config = config
