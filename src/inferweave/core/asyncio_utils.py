"""Helpers for bridging synchronous entry points onto async implementations."""

import asyncio


def ensure_no_running_loop(sync_api: str, async_api: str) -> None:
    """Rejects a synchronous API call made from inside a running event loop.

    Blocking a running loop (or hopping to a worker thread to dodge ``asyncio.run``'s
    nested-loop guard) stalls every other task on it, so fail loudly instead.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError(
        f"{sync_api} cannot be called from a running event loop; await {async_api} instead."
    )
