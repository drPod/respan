"""Instrumentation protocol for Respan plugins."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Instrumentation(Protocol):
    """Protocol that all Respan instrumentation plugins must implement.

    Plugins are discovered via the ``respan.instrumentations`` entry-point
    group and activated by the ``Respan`` class at startup.
    """

    name: str

    def activate(self) -> None:
        """Start intercepting spans."""
        ...

    def deactivate(self) -> None:
        """Stop intercepting spans and clean up resources."""
        ...
