"""Owner fingerprints survive restarts; legacy process-local ones are bound by account id."""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fakes import MODAL_ENV, contains_sentinel, modal_specs

from inferweave.accounts import AccountManager, AccountsConfig, ProviderAccounts
from inferweave.accounts.owner_key import (
    KEY_FILE_NAME,
    OWNER_FINGERPRINT_PREFIX,
    OwnerKey,
)
from inferweave.core.exceptions import (
    AccountConfigurationError,
    AccountUnavailableError,
)
from inferweave.domain.deployment_record import DeploymentRecord

_CHILD = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from fakes import modal_specs
from inferweave.accounts import AccountManager, AccountsConfig, ProviderAccounts
config = AccountsConfig({"modal": ProviderAccounts(accounts=modal_specs())}, state_dir=Path(sys.argv[1]))
environ = json.loads(sys.stdin.read())
print(AccountManager(config, environ=environ).resolve("modal", "a").owner_fingerprint)
"""


def _manager(state_dir: Path, environ: dict[str, str] | None = None) -> AccountManager:
    config = AccountsConfig({"modal": ProviderAccounts(accounts=modal_specs())}, state_dir=state_dir)
    return AccountManager(config, environ=environ if environ is not None else dict(MODAL_ENV))


def _record(fingerprint: str | None, account: str = "a") -> DeploymentRecord:
    return DeploymentRecord(
        id="dep-1", model="m", provider="modal", account=account, owner_fingerprint=fingerprint
    )


def _fingerprint_in_new_process(state_dir: Path) -> str:
    result = subprocess.run(
        [sys.executable, "-c", _CHILD, str(state_dir), str(Path(__file__).parent)],
        input=json.dumps(dict(MODAL_ENV)),
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return result.stdout.strip()


def test_fingerprint_from_another_process_still_resolves(tmp_path):
    """The bug: a fingerprint recorded before a restart must still match after it."""
    recorded = _fingerprint_in_new_process(tmp_path)
    assert recorded.startswith(OWNER_FINGERPRINT_PREFIX)
    assert _fingerprint_in_new_process(tmp_path) == recorded

    record = _record(recorded)
    account = _manager(tmp_path).resolve_owner(record)

    assert account.id == "a"
    assert record.owner_fingerprint == recorded


def test_rotated_control_keys_are_still_refused(tmp_path):
    record = _record(_manager(tmp_path).resolve("modal", "a").owner_fingerprint)
    rotated = dict(MODAL_ENV, IW_MODAL_A_TOKEN_SECRET="SENTINEL-another-cloud-identity")

    with pytest.raises(AccountUnavailableError):
        _manager(tmp_path, rotated).resolve_owner(record)


def test_other_account_fingerprint_is_refused(tmp_path):
    manager = _manager(tmp_path)
    record = _record(manager.resolve("modal", "b").owner_fingerprint, account="a")

    with pytest.raises(AccountUnavailableError):
        manager.resolve_owner(record)


def test_legacy_fingerprint_is_bound_by_account_id_and_restamped(tmp_path, caplog):
    manager = _manager(tmp_path)
    record = _record("7cd8114d453231f9267320917fa4b47b239245e5192af49d1101052f40ab7798")

    with caplog.at_level(logging.WARNING, logger="inferweave.accounts.manager"):
        account = manager.resolve_owner(record)

    assert account.id == "a"
    assert record.owner_fingerprint == manager.resolve("modal", "a").owner_fingerprint
    assert "InferWeave < 0.4.2" in caplog.text
    assert not contains_sentinel(caplog.text)
    # Once restamped, the record is held to the strict check again.
    rotated = dict(MODAL_ENV, IW_MODAL_A_TOKEN_SECRET="SENTINEL-another-cloud-identity")
    with pytest.raises(AccountUnavailableError):
        _manager(tmp_path, rotated).resolve_owner(record)


def test_legacy_fingerprint_of_unconfigured_account_is_refused(tmp_path):
    with pytest.raises(AccountUnavailableError):
        _manager(tmp_path).resolve_owner(_record("legacy-digest", account="missing"))


def test_key_file_is_private_and_reused(tmp_path):
    key = OwnerKey(tmp_path / "state")
    first = key.fingerprint("modal", "a", {"token_id": "x", "token_secret": "y"})
    path = tmp_path / "state" / KEY_FILE_NAME

    assert len(path.read_bytes()) == 32
    if os.name == "posix":
        assert path.stat().st_mode & 0o777 == 0o600
    assert OwnerKey(tmp_path / "state").fingerprint(
        "modal", "a", {"token_id": "x", "token_secret": "y"}
    ) == first
    assert [p.name for p in path.parent.iterdir()] == [KEY_FILE_NAME]
    assert "y" not in first


def test_existing_key_is_never_overwritten(tmp_path):
    path = tmp_path / KEY_FILE_NAME
    path.write_bytes(b"k" * 32)
    before = OwnerKey(tmp_path).fingerprint("modal", "a", {"token_secret": "y"})

    assert path.read_bytes() == b"k" * 32
    assert OwnerKey(tmp_path).fingerprint("modal", "a", {"token_secret": "y"}) == before


def test_corrupt_key_file_fails_loudly(tmp_path):
    (tmp_path / KEY_FILE_NAME).write_bytes(b"short")

    with pytest.raises(AccountConfigurationError, match="not a valid InferWeave owner key"):
        OwnerKey(tmp_path).fingerprint("modal", "a", {"token_secret": "y"})


def test_ambient_and_unpooled_accounts_create_no_key(tmp_path):
    manager = AccountManager(AccountsConfig(state_dir=tmp_path))
    assert manager.resolve_owner(_record(None, account="ambient")).is_ambient
    assert not (tmp_path / KEY_FILE_NAME).exists()
