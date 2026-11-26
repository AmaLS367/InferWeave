"""Persistent key for owner fingerprints.

An owner fingerprint binds a deployment record to the control-plane keys that created it. It is
persisted, so it has to be comparable in every later process. CredWeave fingerprints use a key
that is random per process and never leaves memory, so they cannot be compared after a restart.
InferWeave therefore signs owner fingerprints with its own key, kept in a file under the
accounts ``state_dir`` (owner read/write only).

The fingerprint is a keyed HMAC, not a bare hash: control-plane keys can be low-entropy, and an
unkeyed digest stored in a database could be checked offline against guesses.
"""

import hashlib
import hmac
import os
import secrets
import tempfile
import threading
from collections.abc import Mapping
from pathlib import Path

from inferweave.core.exceptions import AccountConfigurationError

OWNER_FINGERPRINT_PREFIX = "v2:"
"""Marks a fingerprint signed with the persistent owner key (InferWeave >= 0.4.2)."""

KEY_FILE_NAME = "owner-fingerprint.key"
_KEY_BYTES = 32


def is_persistent_owner_fingerprint(value: str) -> bool:
    """True for a fingerprint that later processes can verify.

    Records written by InferWeave < 0.4.2 carry a CredWeave fingerprint whose key died with
    the process that computed it.
    """
    return value.startswith(OWNER_FINGERPRINT_PREFIX)


class OwnerKey:
    """Loads the owner-fingerprint key from ``state_dir``, creating it on first use."""

    def __init__(self, state_dir: str | Path) -> None:
        self._path = Path(state_dir) / KEY_FILE_NAME
        self._value: bytes | None = None
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def fingerprint(self, provider: str, account_id: str, secrets_by_name: Mapping[str, str]) -> str:
        """Keyed digest of an account's control-plane secrets, stable across processes."""
        mac = hmac.new(self._load(), digestmod=hashlib.sha256)
        for part in (provider.lower(), account_id):
            mac.update(part.encode("utf-8"))
            mac.update(b"\x00")
        for name in sorted(secrets_by_name):
            mac.update(name.encode("utf-8"))
            mac.update(b"\x00")
            mac.update(secrets_by_name[name].encode("utf-8"))
            mac.update(b"\x00")
        return OWNER_FINGERPRINT_PREFIX + mac.hexdigest()

    def _load(self) -> bytes:
        with self._lock:
            if self._value is None:
                self._value = self._read_or_create()
            return self._value

    def _read_or_create(self) -> bytes:
        if not self._path.exists():
            self._create()
        try:
            value = self._path.read_bytes()
        except OSError as err:
            raise AccountConfigurationError(
                f"Cannot read the InferWeave owner key {self._path}: {err.strerror}."
            ) from None
        if len(value) != _KEY_BYTES:
            raise AccountConfigurationError(
                f"{self._path} is not a valid InferWeave owner key ({len(value)} bytes, expected "
                f"{_KEY_BYTES}). Restore it from a backup: deployments recorded with it cannot be "
                "operated without it."
            )
        return value

    def _create(self) -> None:
        """Writes a new key atomically; a key another process created first wins."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd, tmp = tempfile.mkstemp(prefix=".owner-key-", dir=self._path.parent)
        except OSError as err:
            raise AccountConfigurationError(
                f"Cannot create the InferWeave owner key in {self._path.parent}: {err.strerror}."
            ) from None
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(secrets.token_bytes(_KEY_BYTES))
                handle.flush()
                os.fsync(handle.fileno())
            try:
                # A hard link publishes the complete file or fails if one already exists, so a
                # concurrent process can never read a partly written key.
                os.link(tmp, self._path)
            except FileExistsError:
                pass
        finally:
            Path(tmp).unlink(missing_ok=True)

    def __repr__(self) -> str:
        return f"OwnerKey(path={str(self._path)!r})"
