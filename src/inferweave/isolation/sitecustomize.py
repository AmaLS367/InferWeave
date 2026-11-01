"""Compatibility hooks for isolated provider worker subprocesses."""

from __future__ import annotations

import contextlib
import os


def _patch_fastapi_prefix_for_skypilot() -> None:
    # SkyPilot 0.13.0 instantiates fastapi.FastAPI(prefix='/api/v1', ...).
    # The 'prefix' kwarg was tolerated in FastAPI <= 0.144.0 but removed in FastAPI >= 0.145.0.
    with contextlib.suppress(Exception):
        import fastapi

        orig_init = fastapi.FastAPI.__init__

        def _init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            kwargs.pop("prefix", None)
            return orig_init(self, *args, **kwargs)

        fastapi.FastAPI.__init__ = _init  # type: ignore[method-assign]


if os.environ.get("IS_SKYPILOT_SERVER") == "true":
    _patch_fastapi_prefix_for_skypilot()
