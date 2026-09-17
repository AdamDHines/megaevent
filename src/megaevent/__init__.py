"""MegaEvent event-based place recognition."""

__version__ = "0.2.0"


def retrieve(reference, query, **kwargs):
    """Retrieve reference matches; see :func:`megaevent.api.retrieve`."""
    from .eval import retrieve as run

    return run(reference, query, **kwargs)
