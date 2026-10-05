"""Shared network/hub exception helpers for the production RAG pipeline.

Provides a lazily-built tuple of exception types that must be caught for graceful
degradation when the network or HuggingFace Hub is unavailable.

The tuple is built lazily at runtime so that:
  1. We never hard-import httpx / huggingface_hub at module-load time (avoiding
     ImportError from accidentally breaking unrelated code paths).
  2. An ImportError during tuple construction is itself swallowed — the tuple
     degrades to the minimum viable set (OSError only).

Runtime verification (installed versions):
  - HfHubHTTPError.__mro__ = (HfHubHTTPError, httpx.HTTPError, OSError, …)
    → HfHubHTTPError already inherits OSError via httpx.HTTPError, so its
      explicit inclusion in the tuple is redundant but kept for defensive
      safety (installed huggingface_hub version may differ).
  - httpx.ProxyError.__mro__ = (ProxyError, httpx.HTTPError, Exception, …)
    → ProxyError does NOT inherit OSError — this is the real gap the fix closes.
"""

from __future__ import annotations


def get_network_exceptions() -> tuple[type, ...]:
    """Return the tuple of exception types to catch for network/hub failures.

    Includes:
      - OSError            : built-in I/O errors, file-not-found on model cache.
      - httpx.HTTPError    : all httpx network errors (e.g. ProxyError,
                             ConnectError, ReadTimeout — none of which inherit
                             OSError in httpx 0.28.x).
      - HfHubHTTPError     : HuggingFace Hub HTTP errors. Kept for defensive
                              safety even though it inherits httpx.HTTPError
                              (which inherits OSError), as the installed
                              huggingface_hub version may differ.

    Returns
    -------
    tuple[type, ...]
        A tuple of exception classes (e.g. ``(OSError, httpx.HTTPError, …)``).
    """
    exc_types: list[type] = [OSError]

    try:
        import httpx as _httpx

        exc_types.append(_httpx.HTTPError)
    except ImportError:
        pass

    try:
        from huggingface_hub.errors import HfHubHTTPError as _HfHubHTTPError

        exc_types.append(_HfHubHTTPError)
    except ImportError:
        pass

    return tuple(exc_types)
