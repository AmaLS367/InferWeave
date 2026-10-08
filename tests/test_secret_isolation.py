"""Sentinel secrets never escape into reprs, storage, logs, errors, runtimes or os.environ."""

import asyncio
import logging
import os
import sqlite3
import traceback
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fakes import (
    ALL_ENV,
    MODAL_ENV,
    FakeProvider,
    accounts_config,
    contains_sentinel,
    crash,
    deploy,
    fail,
    make_weave,
)

from inferweave.accounts import AccountManager
from inferweave.adapters.healthcheck.mock_probe import MockHealthcheckProbeAdapter
from inferweave.adapters.lifecycle.json_repository import JsonDeploymentRepository
from inferweave.core.exceptions import (
    AccountUnavailableError,
    NoAccountAvailableError,
    ProviderAuthError,
    ProviderOperationError,
    ProvisioningUncertainError,
)
from inferweave.core.failures import FailureKind
from inferweave.domain.options import DeploymentOptions, ProviderOptions
from inferweave.models.deployment import DeploymentRequest
from inferweave.providers.lightning_provider import LightningProvider
from inferweave.providers.skypilot import SkyPilotProvider
from inferweave.registry.base import ModelRegistry
from inferweave.runtimes.templates import get_runtime_template
from inferweave.services.provisioning_service import (
    PLATFORM_CREDENTIAL_ENV,
    guard_runtime,
)

pytestmark = pytest.mark.asyncio


def assert_clean(*texts: Any) -> None:
    for text in texts:
        found = contains_sentinel(text if isinstance(text, bytes | str) else repr(text))
        assert not found, f"secret(s) {found} leaked into: {text!r}"[:500]


def _db_bytes(db: Path) -> bytes:
    data = b""
    for candidate in (
        db,
        db.with_name(db.name + "-wal"),
        db.with_name(db.name + "-shm"),
    ):
        if candidate.exists():
            data += candidate.read_bytes()
    return data


async def _cycle(weave, fake: FakeProvider) -> list:
    """Deploy, fail over, stop, an uncertain failure and an unavailable account."""
    secret = MODAL_ENV["IW_MODAL_A_TOKEN_SECRET"]
    fake.script("provision", crash(RuntimeError(f"denied: token_secret={secret}")))
    errors: list[BaseException] = []
    try:
        await deploy(weave)
    except ProvisioningUncertainError as err:
        errors.append(err)
        await weave.reconcile(min_age_seconds=0, confirmed_settled=(err.deployment_id,))
    d1 = await deploy(weave)
    d2 = await deploy(weave, env={"APP_MODE": "prod"})
    await weave.get_status(d1.id)
    await weave.stop(d1.id)
    fake.script(
        "provision",
        crash(RuntimeError(f"Modal-Key {MODAL_ENV['IW_MODAL_B_PROXY_ID']}")),
        crash(ConnectionError(f"x-api-key: {MODAL_ENV['IW_MODAL_A_PROXY_SECRET']}")),
    )
    try:
        await deploy(weave)
    except (NoAccountAvailableError, ProvisioningUncertainError) as err:
        errors.append(err)
    fake.script(
        "provision", fail(FailureKind.TRANSIENT, create=True, message="timeout")
    )
    fake.script("exists", crash(RuntimeError(f"lookup failed {secret}")))
    try:
        await deploy(weave, account="a")
    except ProvisioningUncertainError as err:
        errors.append(err)
    return [d1, d2, *errors]


# --- reprs ---------------------------------------------------------------------------------------


async def test_reprs_contain_no_secrets(tmp_path: Path) -> None:
    fake = FakeProvider()
    config = accounts_config(("modal", "lightning", "runpod", "vast"))
    manager = AccountManager(config, environ=dict(ALL_ENV))
    weave = make_weave(tmp_path / "d.db", manager, None, [fake])
    d = await deploy(weave)
    lease = await manager.acquire("modal")
    objects = [
        manager,
        config,
        lease,
        lease.account,
        d,
        d.status,
        manager.health(),
        weave.endpoint_auth,
        *[manager.resolve(p, "a") for p in ("modal", "lightning", "runpod", "vast")],
        *config.providers.values(),
    ]
    for obj in objects:
        assert_clean(repr(obj), str(obj))
    assert_clean(d.status.model_dump_json())
    await lease.report_success()


# --- storage -------------------------------------------------------------------------------------


async def test_sqlite_rows_and_file_bytes_are_clean(tmp_path: Path) -> None:
    db = tmp_path / "d.db"
    fake = FakeProvider()
    weave = make_weave(db, accounts_config(cooldown_seconds=0), dict(MODAL_ENV), [fake])
    await _cycle(weave, fake)
    await weave.close()
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT * FROM deployments").fetchall()
    assert len(rows) >= 4
    for row in rows:
        assert_clean(" ".join(str(col) for col in row))
    assert_clean(_db_bytes(db))


async def test_json_repository_file_is_clean(tmp_path: Path) -> None:
    path = tmp_path / "deployments.json"
    fake = FakeProvider()
    weave = make_weave(
        tmp_path / "unused.db",
        accounts_config(cooldown_seconds=0),
        dict(MODAL_ENV),
        [fake],
        repository=JsonDeploymentRepository(path),
    )
    await _cycle(weave, fake)
    content = path.read_bytes()
    assert b'"account"' in content
    assert_clean(content)


# --- logs & exceptions ---------------------------------------------------------------------------


async def test_logs_and_exceptions_are_clean(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    fake = FakeProvider()
    env = dict(MODAL_ENV)
    weave = make_weave(
        tmp_path / "d.db", accounts_config(cooldown_seconds=0), env, [fake]
    )
    with caplog.at_level(logging.DEBUG):
        results = await _cycle(weave, fake)
        owner = results[1].account
        del env[f"IW_MODAL_{owner.upper()}_TOKEN_ID"]
        try:
            await weave.stop(results[1].id)  # owned by the now-removed account
        except AccountUnavailableError as err:
            results.append(err)
    errors = [r for r in results if isinstance(r, BaseException)]
    assert {type(e) for e in errors} == {
        ProvisioningUncertainError,
        AccountUnavailableError,
    }
    for err in errors:
        assert_clean(str(err), repr(err), "".join(traceback.format_exception(err)))
    assert caplog.records
    assert_clean(caplog.text)
    for record in caplog.records:
        assert_clean(record.getMessage(), str(record.exc_text or ""))


async def test_unclassified_stop_error_text_is_not_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    fake = FakeProvider()
    weave = make_weave(tmp_path / "d.db", accounts_config(), dict(MODAL_ENV), [fake])
    d = await deploy(weave)
    fake.script(
        "stop", crash(RuntimeError(f"bad token {MODAL_ENV['IW_MODAL_A_TOKEN_SECRET']}"))
    )
    with caplog.at_level(logging.DEBUG), pytest.raises(ProviderOperationError):
        await weave.stop(d.id)
    assert_clean(caplog.text)


# --- runtime guard -------------------------------------------------------------------------------


def _runtime(**request_kwargs: Any):
    profile = ModelRegistry().get("fish-s2-pro")
    request = DeploymentRequest(model=profile.id, provider="modal", **request_kwargs)
    return get_runtime_template(profile.default_runtime).render(
        profile, request
    ), request


@pytest.mark.parametrize("env_name", sorted(PLATFORM_CREDENTIAL_ENV))
async def test_guard_rejects_platform_credential_names(env_name: str) -> None:
    runtime, request = _runtime(env={env_name: "not-a-real-value"})
    with pytest.raises(ProviderAuthError, match="must not be injected"):
        guard_runtime(runtime, request.options, ())
    lower, _ = _runtime(env={env_name.lower(): "x"})
    with pytest.raises(ProviderAuthError):
        guard_runtime(lower, None, ())


async def test_guard_rejects_known_secret_values_anywhere() -> None:
    secret = MODAL_ENV["IW_MODAL_B_TOKEN_SECRET"]
    secrets = tuple(MODAL_ENV.values())
    in_env, _ = _runtime(env={"HF_TOKEN": f"prefix-{secret}"})
    with pytest.raises(ProviderAuthError) as info:
        guard_runtime(in_env, None, secrets)
    assert_clean(str(info.value))

    clean, request = _runtime(env={"APP_MODE": "prod"})
    guard_runtime(clean, request.options, secrets)  # nothing to refuse

    in_args = clean.model_copy(
        update={"run_args": [*clean.run_args, "--token", secret]}
    )
    with pytest.raises(ProviderAuthError):
        guard_runtime(in_args, None, secrets)

    options = DeploymentOptions(
        provider=ProviderOptions(extra_provider_args={"note": secret})
    )
    with pytest.raises(ProviderAuthError):
        guard_runtime(clean, options, secrets)

    # Values shorter than 4 characters are not treated as secrets.
    guard_runtime(clean, None, ("--", "a"))


async def test_guard_rejects_ambient_platform_secret_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "SENTINEL-ambient-runpod-key")
    runtime, _ = _runtime(env={"SOME_SETTING": "SENTINEL-ambient-runpod-key"})
    with pytest.raises(ProviderAuthError):
        guard_runtime(runtime, None, ())


async def test_deploy_refuses_secret_in_env_before_any_remote_call(
    tmp_path: Path,
) -> None:
    fake = FakeProvider()
    weave = make_weave(tmp_path / "d.db", accounts_config(), dict(MODAL_ENV), [fake])
    with pytest.raises(ProviderAuthError):
        await deploy(weave, env={"HF_TOKEN": MODAL_ENV["IW_MODAL_B_PROXY_SECRET"]})
    with pytest.raises(ProviderAuthError):
        await deploy(weave, env={"MODAL_TOKEN_ID": "anything"})
    assert fake.ops("provision") == []
    assert await weave.list_records() == []
    # A refused request is not the account's fault.
    assert all(
        h.state == "available" and h.in_flight == 0 for h in weave.account_health()
    )


# --- endpoint headers ----------------------------------------------------------------------------


class RecordingProbe(MockHealthcheckProbeAdapter):
    def __init__(self) -> None:
        super().__init__(default_healthy=True)
        self.headers: list[tuple[str, dict[str, str]]] = []

    async def probe(
        self,
        url: str,
        method: str = "GET",
        timeout_seconds: float = 5.0,
        headers=None,
        expected_status_codes=None,
    ):
        self.headers.append((url, dict(headers or {})))
        return await super().probe(
            url, method, timeout_seconds, headers, expected_status_codes
        )


async def test_endpoint_headers_come_from_owning_account_only(tmp_path: Path) -> None:
    probe = RecordingProbe()
    fake = FakeProvider()
    weave = make_weave(
        tmp_path / "d.db", accounts_config(), dict(MODAL_ENV), [fake], probe=probe
    )
    d_a, d_b = await deploy(weave), await deploy(weave)
    expected = {
        d_a.id: {
            "Modal-Key": MODAL_ENV["IW_MODAL_A_PROXY_ID"],
            "Modal-Secret": MODAL_ENV["IW_MODAL_A_PROXY_SECRET"],
        },
        d_b.id: {
            "Modal-Key": MODAL_ENV["IW_MODAL_B_PROXY_ID"],
            "Modal-Secret": MODAL_ENV["IW_MODAL_B_PROXY_SECRET"],
        },
    }
    for d in (d_a, d_b):
        assert weave._auth_headers_for(d) == expected[d.id]
        record = await weave.lifecycle_service.get_record(d.id)
        assert (
            weave.lifecycle_service.endpoint_headers(record, d.endpoint_url)
            == expected[d.id]
        )
        await weave.get_status(d.id)
        await d.check_health()
    for url, headers in probe.headers:
        dep_id = next(i for i in expected if i in url)
        assert headers == expected[dep_id]
    assert {next(i for i in expected if i in url) for url, _ in probe.headers} == set(
        expected
    )


async def test_account_without_proxy_tokens_sends_no_ambient_tokens(
    tmp_path: Path,
) -> None:
    from fakes import modal_spec

    from inferweave.accounts import AccountsConfig, ProviderAccounts

    config = AccountsConfig(
        {"modal": ProviderAccounts(accounts=[modal_spec("a", proxy=False)])}
    )
    fake = FakeProvider()
    weave = make_weave(tmp_path / "d.db", config, dict(MODAL_ENV), [fake])
    d = await deploy(weave)
    # The ambient MODAL_PROXY_TOKEN_* (conftest) belong to another workspace: never sent.
    assert weave._auth_headers_for(d) == {}


# --- os.environ ----------------------------------------------------------------------------------


class FakeRunner:
    """Answers worker operations for the Lightning and SkyPilot providers."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.gone: set[str] = set()
        self.created: set[str] = set()

    async def run(
        self,
        operation,
        payload,
        *,
        env,
        secrets,
        timeout,
        deployment_id=None,
        account_id="",
    ):
        self.calls.append(
            {
                "op": operation,
                "payload": dict(payload),
                "env": dict(env),
                "secrets": dict(secrets or {}),
                "account": account_id,
            }
        )
        await asyncio.sleep(0)
        target = payload.get("target") or payload.get("cluster")
        if operation == "whoami":
            return {"auth_type": "user"}
        if operation == "start":
            self.created.add(payload["name"])
            return {
                "urls": [f"https://8080-{payload['name']}.cloudspaces.litng.ai"],
                "resource_id": payload["name"],
            }
        if operation == "launch":
            return {"endpoint": f"http://{payload['cluster']}.example.test:8080"}
        if operation == "inspect":
            return (
                None
                if target in self.gone or target not in self.created
                else {"desired_state": "RUNNING", "status": {}}
            )
        if operation == "status":
            return {"exists": target not in self.gone, "status": "UP"}
        if operation in ("delete", "down", "stop"):
            self.gone.add(target)
        return {}


async def test_os_environ_untouched_by_concurrent_multi_provider_deploys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace
    monkeypatch.setattr("inferweave.services.routing_service.sys", SimpleNamespace(platform="linux"))
    modal = FakeProvider()
    lightning_runner, runpod_runner, vast_runner = (
        FakeRunner(),
        FakeRunner(),
        FakeRunner(),
    )
    providers = [
        modal,
        LightningProvider(state_dir=tmp_path / "state", runner=lightning_runner),  # type: ignore[arg-type]
        SkyPilotProvider("runpod", state_dir=tmp_path / "state", runner=runpod_runner),  # type: ignore[arg-type]
        SkyPilotProvider("vast", state_dir=tmp_path / "state", runner=vast_runner),  # type: ignore[arg-type]
    ]
    config = accounts_config(
        ("modal", "lightning", "runpod", "vast"), state_dir=tmp_path / "state"
    )
    weave = make_weave(tmp_path / "d.db", config, dict(ALL_ENV), providers)
    before = dict(os.environ)
    with patch.object(
        SkyPilotProvider, "_ensure_supported_platform", return_value=None
    ):
        deployments = await asyncio.gather(
            *(
                deploy(weave, provider=name, gpu_type="L4")
                for name in ("modal", "lightning", "runpod", "vast")
                for _ in range(2)
            )
        )
        await asyncio.gather(*(weave.stop(d.id) for d in deployments))
    assert dict(os.environ) == before
    assert {(d.provider, d.account) for d in deployments} == {
        (p, a) for p in ("modal", "lightning", "runpod", "vast") for a in ("a", "b")
    }
    for runner, provider in (
        (lightning_runner, "lightning"),
        (runpod_runner, "runpod"),
        (vast_runner, "vast"),
    ):
        assert runner.calls
        for call in runner.calls:
            # Secrets travel only in the dedicated channel, and only the caller's own account's.
            assert_clean(repr(call["env"]), repr(call["payload"]))
            own = {
                v
                for k, v in ALL_ENV.items()
                if k.startswith(f"IW_{provider.upper()}_{call['account'].upper()}_")
            }
            assert set(call["secrets"].values()) == own
            assert "MODAL_PROXY_TOKEN_SECRET" not in call["env"]
            assert call["env"]["HOME"].startswith(str(tmp_path / "state"))
    assert modal.live == {}
