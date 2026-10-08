"""Nonblocking process locks; lock files are never removed while other processes may use them."""

import contextlib
import importlib
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any


@contextlib.contextmanager
def file_lock(path: Path, *, blocking: bool = False) -> Iterator[bool]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0)
        if sys.platform == "win32":
            import msvcrt

            # Windows byte-range locks need an existing byte.
            if path.stat().st_size == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(
                    handle.fileno(), msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK, 1
                )
            except OSError:
                yield False
                return
            try:
                yield True
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl: Any = importlib.import_module("fcntl")

            try:
                fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
            except BlockingIOError:
                yield False
                return
            try:
                yield True
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
