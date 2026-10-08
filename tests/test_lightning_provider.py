"""Offline Lightning provider tests; every worker operation goes through a fake WorkerRunner.

No test here starts a subprocess or contacts Lightning. The fake runner records each call's
``op``/``payload``/``env``/``secrets`` so isolation and account affinity can be asserted.
"""

import asyncio
import hashlib
import io
import json
import os
import shlex
import sqlite3
import sys
import wave
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from support import AUDIO_MODEL, FAST_RETRIES, IMAGE_MODEL

from inferweave import (
    DeploymentState,
    InferWeave,
    LightningEndpointAuth,
    LightningOptions,
)
from inferweave.accounts import (
    AccountManager,
    AccountsConfig,
    ProviderAccount,
    ProviderAccounts,
    lightning_account,
)
from inferweave.adapters.auth import default_endpoint_auth
from inferweave.adapters.healthcheck.mock_probe import MockHealthcheckProbeAdapter
from inferweave.adapters.lifecycle.sqlite_repository import SqliteDeploymentRepository
from inferweave.adapters.lifecycle.watchdog import MockWatchdogAdapter
from inferweave.core.exceptions import (
    AccountUnavailableError,
    HealthcheckTimeoutError,
    InsufficientVramError,
    ProviderAuthError,
    ProviderOperationError,
    ProviderPlatformError,
    ProvisioningUncertainError,
)
from inferweave.core.failures import FailureKind
from inferweave.domain.deployment_record import DeploymentRecord
from inferweave.domain.lifecycle import AutostopAction
from inferweave.isolation.runner import _PASSTHROUGH_ENV
from inferweave.models.deployment import DeploymentRequest
from inferweave.models.profile import HealthcheckConfig
from inferweave.providers import lightning_provider as lp
from inferweave.providers.lightning_provider import LightningProvider, runtime_command
from inferweave.providers.router import ProviderRouter
from inferweave.registry.base import ModelRegistry
from inferweave.runtimes.base import RuntimeSpec
from inferweave.runtimes.templates import get_runtime_template
from inferweave.services.healthcheck_service import HealthcheckService
from inferweave.services.lifecycle_service import LifecycleService
from inferweave.services.provisioning_service import guard_runtime

KEY = "test-lightning-secret"
HOST_SUFFIX = ".cloudspaces.litng.ai"
ENDPOINT = "https://8080-dep-test-d.cloudspaces.litng.ai"
COLLISION_MESSAGE = "Lightning resource name collision; the existing resource was preserved."

# Sentinel credentials of two pooled accounts, resolved via AccountManager(environ=...).
ACCOUNT_ENV = {
    "ACCT_A_USER": "SENTINEL-user-a",
    "ACCT_A_KEY": "SENTINEL-key-a",
    "ACCT_B_USER": "SENTINEL-user-b",
    "ACCT_B_KEY": "SENTINEL-key-b",
}
ACCOUNT_SECRETS = {
    "acct-a": {"LIGHTNING_USER_ID": "SENTINEL-user-a", "LIGHTNING_API_KEY": "SENTINEL-key-a"},
    "acct-b": {"LIGHTNING_USER_ID": "SENTINEL-user-b", "LIGHTNING_API_KEY": "SENTINEL-key-b"},
}
ACCOUNT_TEAMSPACE = {"acct-a": "own-a/ts-a", "acct-b": "own-b/ts-b"}
AMBIENT_POISON = {
    "LIGHTNING_USER_ID": "AMBIENT-user",
    "LIGHTNING_API_KEY": "AMBIENT-key",
    "LIGHTNING_AUTH_TOKEN": "AMBIENT-token",
    "LIGHTNING_TEAMSPACE": "ambient/space",
    "RUNPOD_API_KEY": "AMBIENT-runpod",
    "VAST_API_KEY": "AMBIENT-vast",
    "MODAL_TOKEN_ID": "AMBIENT-modal-id",
    "MODAL_TOKEN_SECRET": "AMBIENT-modal-secret",
    "HF_TOKEN": "AMBIENT-hf",
    "AWS_SECRET_ACCESS_KEY": "AMBIENT-aws",
}


# --- fake worker runner ------------------------------------------------------------------------


@dataclass
class Call:
    op: str
    payload: dict[str, Any]
    env: dict[str, str]
    secrets: dict[str, str]
    timeout: float
    deployment_id: str | None
    account_id: str


@dataclass
class FakeRunner:
    """Stands in for ``WorkerRunner``: an in-memory Lightning control plane."""

    calls: list[Call] = field(default_factory=list)
    resources: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    identity: dict[str, str] = field(default_factory=dict)  # account id -> auth_type
    whoami_error: ProviderOperationError | None = None
    start_error: BaseException | None = None  # raised after the resource was created
    collide: bool = False  # a foreign resource already owns the name
    urls: list[str] | None = None  # None -> one per-resource https URL
    delete_error: ProviderOperationError | None = None
    delete_lag: int = 0  # inspect calls that still see a deleted resource
    lingering: dict[tuple[str, str], int] = field(default_factory=dict)
    start_gate: asyncio.Event | None = None
    start_entered: asyncio.Event = field(default_factory=asyncio.Event)

    async def run(
        self,
        op: str,
        payload: Any,
        *,
        env: Any,
        secrets: Any = None,
        timeout: float,
        deployment_id: str | None = None,
        account_id: str = "",
    ) -> Any:
        call = Call(
            op, dict(payload), dict(env), dict(secrets or {}), timeout, deployment_id, account_id
        )
        self.calls.append(call)
        if op == "start":
            return await self._start(call)
        return getattr(self, f"_{op}")(call)

    # --- operations ---

    def _whoami(self, call: Call) -> dict[str, Any]:
        if self.whoami_error is not None:
            raise self.whoami_error
        return {"auth_type": self.identity.get(call.account_id, "user")}

    async def _start(self, call: Call) -> dict[str, Any]:
        key = (call.payload["teamspace"], call.payload["name"])
        if self.collide:
            self.resources[key] = self._data(call.payload["name"], foreign=True)
            raise ProviderOperationError(
                COLLISION_MESSAGE, FailureKind.INVALID_REQUEST, resource_may_exist=False
            )
        self.resources[key] = self._data(call.payload["name"])
        self.start_entered.set()
        if self.start_gate is not None:
            await self.start_gate.wait()
        if self.start_error is not None:
            raise self.start_error
        name = call.payload["name"]
        urls = self.urls if self.urls is not None else [f"https://8080-{name}{HOST_SUFFIX}"]
        return {"resource_id": "dep_" + name, "urls": urls}

    def _find(self, payload: dict[str, Any]) -> tuple[str, str] | None:
        for key, data in self.resources.items():
            if key[0] == payload["teamspace"] and payload["target"] in (data["id"], key[1]):
                return key
        return None

    def _inspect(self, call: Call) -> dict[str, Any] | None:
        key = self._find(call.payload)
        if key is None:
            return None
        if key in self.lingering:
            remaining = self.lingering[key]
            if remaining <= 0:
                del self.resources[key], self.lingering[key]
                return None
            self.lingering[key] = remaining - 1
        return dict(self.resources[key])

    def _delete(self, call: Call) -> dict[str, bool]:
        if self.delete_error is not None:
            raise self.delete_error
        key = self._find(call.payload)
        if key is None:
            return {"deleted": False}
        if self.delete_lag:
            self.lingering[key] = self.delete_lag
        else:
            del self.resources[key]
        return {"deleted": True}

    # --- helpers ---

    @staticmethod
    def _data(name: str, **extra: Any) -> dict[str, Any]:
        return {
            "id": "dep_" + name,
            "name": name,
            "desired_state": "RUNNING",
            "status": {"ready_replicas": 1, "pending_replicas": 0, "failing_replicas": 0},
            **extra,
        }

    def add_resource(self, teamspace: str, name: str, **extra: Any) -> dict[str, Any]:
        self.resources[(teamspace, name)] = self._data(name, **extra)
        return self.resources[(teamspace, name)]

    def of(self, op: str) -> list[Call]:
        return [c for c in self.calls if c.op == op]

    def ops(self) -> list[str]:
        return [c.op for c in self.calls]


# --- fixtures and helpers ------------------------------------------------------------------------


@pytest.fixture
def ambient_env(monkeypatch):
    monkeypatch.setenv("LIGHTNING_USER_ID", "test-user")
    monkeypatch.setenv("LIGHTNING_API_KEY", KEY)
    monkeypatch.setenv("LIGHTNING_TEAMSPACE", "owner/tests")
    monkeypatch.delenv("LIGHTNING_ORG", raising=False)


@pytest.fixture
def runner():
    return FakeRunner()


@pytest.fixture
def provider(runner, tmp_path):
    return LightningProvider(runner=runner, state_dir=tmp_path / "accounts")


def recipe(model="fish-s2-pro", **kwargs):
    profile = ModelRegistry().get(model)
    request = DeploymentRequest(model=model, provider="lightning", **kwargs)
    return request, profile, get_runtime_template(profile.default_runtime).render(profile, request)


def pool_config(tmp_path, *, strategy="round_robin", **overrides):
    specs = [
        lightning_account(
            "acct-a",
            user_id_env="ACCT_A_USER",
            api_key_env="ACCT_A_KEY",
            teamspace=ACCOUNT_TEAMSPACE["acct-a"],
            priority=0,
        ),
        lightning_account(
            "acct-b",
            user_id_env="ACCT_B_USER",
            api_key_env="ACCT_B_KEY",
            teamspace=ACCOUNT_TEAMSPACE["acct-b"],
            priority=1,
        ),
    ]
    return AccountsConfig(
        {"lightning": ProviderAccounts(accounts=specs, strategy=strategy)},
        state_dir=tmp_path / "accounts",
        **overrides,
    )


def pool_manager(tmp_path, **kwargs):
    return AccountManager(pool_config(tmp_path, **kwargs), environ=ACCOUNT_ENV)


def account_of(manager: AccountManager, account_id: str) -> ProviderAccount:
    return manager.resolve("lightning", account_id)


def new_record(
    provider: LightningProvider, account: ProviderAccount, **kwargs: Any
) -> tuple[DeploymentRecord, DeploymentRequest, Any, RuntimeSpec]:
    request, profile, runtime = recipe(**kwargs)
    deployment_id = provider.new_deployment_id(profile)
    record = DeploymentRecord(
        id=deployment_id,
        model=profile.id,
        provider="lightning",
        account=account.id,
        resource=provider.resource_ref(deployment_id, request, account),
    )
    return record, request, profile, runtime


async def provision(provider, account, **kwargs):
    record, request, profile, runtime = new_record(provider, account, **kwargs)
    result = await provider.provision(record, request, profile, runtime, account)
    return record, result


def home_for(provider: LightningProvider, account_id: str) -> Path:
    account = account_of(pool_manager(provider.state_dir.parent), account_id)
    digest = hashlib.sha256(f"{account.id}/{account.owner_fingerprint}".encode()).hexdigest()[:16]
    return provider.state_dir / "lightning" / digest


# --- machine mapping -----------------------------------------------------------------------------


@pytest.mark.parametrize("gpu,count,machine", [
    ("L4", 1, "L4"), ("L40S", 2, "L40S_X_2"),
    ("A100-40GB", 1, "A100_40GB"), ("A100-80GB", 4, "A100_80GB_X_4"),
    ("H100", 1, "H100"), ("H200", 1, "H200"),
    ("H200", 2, "H200_X_2"), ("H200", 4, "H200_X_4"),
    ("H200", 8, "H200_X_8"), ("B200", 1, "B200"), ("B200", 8, "B200_X_8"),
])
def test_machine_mapping(gpu, count, machine):
    request, profile, _ = recipe(gpu_type=gpu, num_gpus=count)
    assert LightningProvider.machine_name(request, profile) == machine


@pytest.mark.parametrize("gpu,count", [
    ("RTX4090", 1), ("A10G", 1), ("L4", 3), ("B200", 2), ("B200", 4),
])
def test_unsupported_machine_is_typed(gpu, count):
    request, profile, _ = recipe(gpu_type=gpu, num_gpus=count)
    with pytest.raises(ProviderPlatformError, match="cannot be represented"):
        LightningProvider.machine_name(request, profile)


def test_t4_is_too_small_for_fish():
    with pytest.raises(InsufficientVramError):
        LightningProvider.machine_name(*recipe(gpu_type="T4")[:2])


# --- teamspace resolution ------------------------------------------------------------------------


def test_ambient_teamspace_from_environment(provider, ambient_env):
    assert provider.teamspace(recipe()[0], ProviderAccount.ambient("lightning")) == "owner/tests"


def test_ambient_org_and_teamspace_configuration(provider, ambient_env, monkeypatch):
    monkeypatch.setenv("LIGHTNING_TEAMSPACE", "tests")
    monkeypatch.setenv("LIGHTNING_ORG", "org")
    assert provider.teamspace(recipe()[0], ProviderAccount.ambient("lightning")) == "org/tests"


def test_explicit_teamspace_overrides_environment(provider, ambient_env):
    request = recipe(custom_args={"lightning": {"teamspace": "other/space"}})[0]
    assert provider.teamspace(request, ProviderAccount.ambient("lightning")) == "other/space"


def test_pooled_account_uses_its_metadata_teamspace(provider, tmp_path, monkeypatch):
    monkeypatch.setenv("LIGHTNING_TEAMSPACE", "ambient/space")
    manager = pool_manager(tmp_path)
    request = recipe()[0]
    for account_id, teamspace in ACCOUNT_TEAMSPACE.items():
        assert provider.teamspace(request, account_of(manager, account_id)) == teamspace


def test_per_deploy_teamspace_overrides_account_teamspace(provider, tmp_path):
    account = account_of(pool_manager(tmp_path), "acct-a")
    request = recipe(custom_args={"lightning": {"teamspace": "other/space"}})[0]
    assert provider.teamspace(request, account) == "other/space"


def test_pooled_account_without_teamspace_ignores_ambient_env(provider, monkeypatch):
    monkeypatch.setenv("LIGHTNING_TEAMSPACE", "ambient/space")
    config = AccountsConfig({"lightning": ProviderAccounts(accounts=[
        lightning_account("bare", user_id_env="ACCT_A_USER", api_key_env="ACCT_A_KEY"),
    ])})
    account = AccountManager(config, environ=ACCOUNT_ENV).resolve("lightning", "bare")
    with pytest.raises(ProviderAuthError, match="teamspace"):
        provider.teamspace(recipe()[0], account)


def test_missing_teamspace_is_auth_error_but_dry_run_has_placeholder(provider, monkeypatch):
    monkeypatch.delenv("LIGHTNING_TEAMSPACE", raising=False)
    ambient = ProviderAccount.ambient("lightning")
    with pytest.raises(ProviderAuthError, match="teamspace"):
        provider.teamspace(recipe()[0], ambient)
    assert provider.teamspace(recipe(dry_run=True)[0], ambient) == "dryrun/teamspace"


def test_resource_ref_is_unowned_until_name_is_confirmed_free(provider, ambient_env):
    ref = provider.resource_ref("iw-lightning-x", recipe()[0], ProviderAccount.ambient("lightning"))
    assert (ref.name, ref.scope, ref.owned) == ("iw-lightning-x", "owner/tests", False)


def test_resource_prefix_is_validated():
    with pytest.raises(ValueError, match="iw-lightning"):
        LightningProvider(resource_prefix="other")
    with pytest.raises(ValueError, match="letters/digits/hyphens"):
        LightningProvider(resource_prefix="iw-lightning_bad")


# --- credential guards ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["LIGHTNING_API_KEY", "LIGHTNING_USER_ID"])
def test_missing_ambient_credentials_fail_preflight(provider, runner, ambient_env, monkeypatch, key):
    monkeypatch.delenv(key)
    request, profile, runtime = recipe()
    with pytest.raises(ProviderAuthError, match="LIGHTNING_USER_ID"):
        provider.preflight(request, profile, runtime, ProviderAccount.ambient("lightning"))
    assert not runner.calls


def test_preflight_requires_endpoint_auth_for_readiness(tmp_path, runner, ambient_env):
    class NoAuth(type(default_endpoint_auth())):
        def headers_for(self, provider, endpoint_url, account=None):
            return {}

    provider = LightningProvider(endpoint_auth=NoAuth(), runner=runner, state_dir=tmp_path)
    request, profile, runtime = recipe()
    with pytest.raises(ProviderAuthError, match="endpoint auth resolver"):
        provider.preflight(request, profile, runtime, ProviderAccount.ambient("lightning"))


def test_preflight_rejects_provider_arg_passthrough(provider, ambient_env):
    request, profile, runtime = recipe(custom_args={"provider_args": {"anything": 1}})
    with pytest.raises(ProviderPlatformError, match="pass-through"):
        provider.preflight(request, profile, runtime, ProviderAccount.ambient("lightning"))


def test_dry_run_preflight_needs_no_credentials(provider, runner, monkeypatch):
    for name in ("LIGHTNING_API_KEY", "LIGHTNING_USER_ID", "LIGHTNING_TEAMSPACE"):
        monkeypatch.delenv(name, raising=False)
    request, profile, runtime = recipe(dry_run=True)
    provider.preflight(request, profile, runtime, ProviderAccount.ambient("lightning"))
    assert not runner.calls


def test_pooled_preflight_uses_the_accounts_own_endpoint_key(provider, tmp_path, monkeypatch):
    for name in ("LIGHTNING_API_KEY", "LIGHTNING_USER_ID", "LIGHTNING_TEAMSPACE"):
        monkeypatch.delenv(name, raising=False)
    request, profile, runtime = recipe()
    provider.preflight(request, profile, runtime, account_of(pool_manager(tmp_path), "acct-a"))


@pytest.mark.parametrize("env_var", [
    "LIGHTNING_API_KEY", "LIGHTNING_USER_ID", "LIGHTNING_AUTH_TOKEN",
])
@pytest.mark.parametrize("location", ["cli", "engine", "options_env", "runtime_env", "setup"])
def test_embedded_credentials_rejected_before_persistence(monkeypatch, env_var, location):
    values = {"LIGHTNING_API_KEY": KEY, "LIGHTNING_USER_ID": "test-user",
              "LIGHTNING_AUTH_TOKEN": "test-auth-token"}
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    credential = values[env_var]
    value = f"prefix-{credential}-suffix"
    custom_args = {
        "cli": {"extra_cli_args": [f"--token={value}"]},
        "engine": {"engine_args": {"token": f"Bearer {value}"}},
        "options_env": {"extra_env": {"OTHER_NAME": f"Bearer {value}"}},
    }.get(location, {})
    request, _, runtime = recipe(custom_args=custom_args)
    if location == "runtime_env":
        runtime.env_vars["OTHER_NAME"] = f"Bearer {value}"
    elif location == "setup":
        runtime.setup_commands.append(f"echo {value}")
    with pytest.raises(ProviderAuthError, match="must not be injected") as error:
        guard_runtime(runtime, request.options, ())
    assert credential not in str(error.value)


@pytest.mark.parametrize("custom_args", [
    {"engine_args": {"LIGHTNING_API_KEY": KEY}},
    {"extra_env": {"LIGHTNING_USER_ID": "x"}},
])
def test_platform_credential_names_rejected_even_without_ambient_values(custom_args):
    request, _, runtime = recipe(custom_args=custom_args)
    with pytest.raises(ProviderAuthError, match="must not be injected"):
        guard_runtime(runtime, request.options, ())


def test_pooled_account_secrets_cannot_reach_a_runtime(tmp_path):
    manager = pool_manager(tmp_path)
    request, _, runtime = recipe(env={"OTHER_NAME": "Bearer SENTINEL-key-b"})
    with pytest.raises(ProviderAuthError, match="must not be injected"):
        guard_runtime(runtime, request.options, manager.secret_values())


def test_unrelated_runtime_secrets_are_allowed(tmp_path):
    request, _, runtime = recipe(env={"MODEL_TOKEN": "unrelated-model-secret"})
    guard_runtime(runtime, request.options, pool_manager(tmp_path).secret_values())


# --- verify_credentials --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_credentials_accepts_user_key(provider, runner, ambient_env):
    await provider.verify_credentials(ProviderAccount.ambient("lightning"))
    assert runner.ops() == ["whoami"]


@pytest.mark.parametrize("identity", ["scoped-api-key", "service", ""])
@pytest.mark.asyncio
async def test_scoped_key_is_permission_error_without_resource(provider, runner, ambient_env, identity):
    runner.identity["ambient"] = identity
    with pytest.raises(ProviderOperationError) as caught:
        await provider.verify_credentials(ProviderAccount.ambient("lightning"))
    assert caught.value.kind is FailureKind.PERMISSION
    assert caught.value.resource_may_exist is False


@pytest.mark.asyncio
async def test_non_dict_identity_is_permission_error(provider, runner, ambient_env, monkeypatch):
    async def run(*args, **kwargs):
        return None

    monkeypatch.setattr(runner, "run", run)
    with pytest.raises(ProviderOperationError) as caught:
        await provider.verify_credentials(ProviderAccount.ambient("lightning"))
    assert caught.value.kind is FailureKind.PERMISSION


@pytest.mark.asyncio
async def test_scoped_key_blocks_provisioning_before_any_resource(provider, runner, ambient_env):
    runner.identity["ambient"] = "scoped-api-key"
    record, request, profile, runtime = new_record(provider, ProviderAccount.ambient("lightning"))
    with pytest.raises(ProviderOperationError) as caught:
        await provider.provision(record, request, profile, runtime, ProviderAccount.ambient("lightning"))
    assert caught.value.kind is FailureKind.PERMISSION
    assert caught.value.resource_may_exist is False
    assert runner.ops() == ["whoami"]
    assert record.resource.owned is False


@pytest.mark.parametrize("kind", [FailureKind.AUTH, FailureKind.TRANSIENT, FailureKind.RATE_LIMIT])
@pytest.mark.asyncio
async def test_verify_failure_is_marked_resource_may_exist_false(provider, runner, ambient_env, kind):
    # Even a runner-level failure that claims the resource may exist (e.g. a worker crash)
    # cannot have created anything: nothing was started yet.
    runner.whoami_error = ProviderOperationError("boom", kind, resource_may_exist=True)
    account = ProviderAccount.ambient("lightning")
    record, request, profile, runtime = new_record(provider, account)
    with pytest.raises(ProviderOperationError) as caught:
        await provider.provision(record, request, profile, runtime, account)
    assert caught.value.kind is kind
    assert caught.value.resource_may_exist is False
    assert runner.ops() == ["whoami"]
    assert record.resource.owned is False


# --- provision -----------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provision_sends_shared_runtime_recipe_to_worker(provider, runner, ambient_env):
    account = ProviderAccount.ambient("lightning")
    record, result = await provision(provider, account)
    _, _, runtime = recipe()

    assert runner.ops() == ["whoami", "start"]
    start = runner.of("start")[0]
    assert start.deployment_id == record.id and start.account_id == "ambient"
    payload = start.payload
    assert payload["name"] == record.resource.name == record.id
    assert payload["teamspace"] == record.resource.scope == "owner/tests"
    assert payload["machine"] == "L4"
    assert payload["image"] == runtime.docker_image
    assert payload["port"] == runtime.port == 8080
    assert payload["healthcheck_path"] == "/v1/health"
    assert payload["env"] == runtime.env_vars
    assert "--revision" in payload["command"]
    assert (payload["min_replicas"], payload["max_replicas"]) == (0, 1)
    assert payload["idle_threshold_seconds"] == 300
    assert payload["discovery_timeout"] == lp.ENDPOINT_DISCOVERY_TIMEOUT_SECONDS

    assert result.state == DeploymentState.STARTING
    assert result.endpoint_url == f"https://8080-{record.id}{HOST_SUFFIX}"
    assert result.resource_id == "dep_" + record.id
    assert record.resource.owned is True


@pytest.mark.asyncio
async def test_typed_autoscale_options_reach_the_worker(provider, runner, ambient_env):
    await provision(provider, ProviderAccount.ambient("lightning"), custom_args={
        "lightning": {"min_replicas": 0, "max_replicas": 2, "idle_threshold_seconds": 90},
    })
    payload = runner.of("start")[0].payload
    assert (payload["min_replicas"], payload["max_replicas"]) == (0, 2)
    assert payload["idle_threshold_seconds"] == 90


@pytest.mark.asyncio
async def test_image_model_ships_inferweave_worker_sources(provider, runner, ambient_env):
    await provision(
        provider, ProviderAccount.ambient("lightning"), model=IMAGE_MODEL, gpu_type="L40S"
    )
    assert "/tmp/iw-runtime" in runner.of("start")[0].payload["command"]


@pytest.mark.asyncio
async def test_unrelated_runtime_env_reaches_the_container(provider, runner, ambient_env):
    await provision(
        provider, ProviderAccount.ambient("lightning"), env={"MODEL_TOKEN": "unrelated-model-secret"}
    )
    assert runner.of("start")[0].payload["env"]["MODEL_TOKEN"] == "unrelated-model-secret"


@pytest.mark.asyncio
async def test_collision_is_definitive_and_not_owned(provider, runner, ambient_env):
    runner.collide = True
    account = ProviderAccount.ambient("lightning")
    record, request, profile, runtime = new_record(provider, account)
    with pytest.raises(ProviderOperationError, match="collision") as caught:
        await provider.provision(record, request, profile, runtime, account)
    assert caught.value.kind is FailureKind.INVALID_REQUEST
    assert caught.value.resource_may_exist is False
    assert record.resource.owned is False
    assert "delete" not in runner.ops()


@pytest.mark.parametrize("kind,may_exist", [
    (FailureKind.TRANSIENT, True),
    (FailureKind.PERMISSION, True),
    (FailureKind.RATE_LIMIT, True),
])
@pytest.mark.asyncio
async def test_uncertain_start_failure_marks_resource_owned(provider, runner, ambient_env, kind, may_exist):
    runner.start_error = ProviderOperationError("start failed", kind, resource_may_exist=may_exist)
    account = ProviderAccount.ambient("lightning")
    record, request, profile, runtime = new_record(provider, account)
    assert record.resource.owned is False
    with pytest.raises(ProviderOperationError) as caught:
        await provider.provision(record, request, profile, runtime, account)
    assert caught.value.kind is kind
    assert record.resource.owned is True


@pytest.mark.asyncio
async def test_definitive_start_rejection_leaves_resource_unowned(provider, runner, ambient_env):
    runner.start_error = ProviderOperationError(
        "forbidden", FailureKind.PERMISSION, status_code=403, resource_may_exist=False
    )
    account = ProviderAccount.ambient("lightning")
    record, request, profile, runtime = new_record(provider, account)
    with pytest.raises(ProviderOperationError):
        await provider.provision(record, request, profile, runtime, account)
    assert record.resource.owned is False


@pytest.mark.parametrize("urls", [[], None, ["https://a.cloudspaces.litng.ai", "https://b.cloudspaces.litng.ai"]])
@pytest.mark.asyncio
async def test_missing_or_ambiguous_endpoint_is_transient_and_owned(provider, runner, ambient_env, urls):
    class Odd(FakeRunner):
        async def _start(self, call):
            await super()._start(call)
            return {"resource_id": "dep_x", "urls": urls}

    odd = Odd()
    provider.runner = odd
    account = ProviderAccount.ambient("lightning")
    record, request, profile, runtime = new_record(provider, account)
    with pytest.raises(ProviderOperationError, match="single runtime endpoint") as caught:
        await provider.provision(record, request, profile, runtime, account)
    assert caught.value.kind is FailureKind.TRANSIENT
    assert caught.value.resource_may_exist is True
    assert record.resource.owned is True


@pytest.mark.parametrize("url", [
    "https://foreign.example",
    "http://8080-dep-test-d.cloudspaces.litng.ai",
    "https://evil.cloudspaces.litng.ai.attacker.test",
])
@pytest.mark.asyncio
async def test_unsafe_endpoint_is_refused_without_leaking_credentials(provider, runner, ambient_env, url):
    runner.urls = [url]
    account = ProviderAccount.ambient("lightning")
    record, request, profile, runtime = new_record(provider, account)
    with pytest.raises(ProviderOperationError, match="HTTPS auth resolver") as caught:
        await provider.provision(record, request, profile, runtime, account)
    assert caught.value.kind is FailureKind.INVALID_REQUEST
    assert caught.value.resource_may_exist is True
    assert KEY not in str(caught.value)
    assert record.resource.owned is True


# --- status mapping ------------------------------------------------------------------------------


@pytest.mark.parametrize("state,pending,failing,expected", [
    ("RUNNING", 0, 0, DeploymentState.STARTING),
    ("RUNNING", 1, 0, DeploymentState.PROVISIONING),
    ("RUNNING", 0, 1, DeploymentState.FAILED),
    ("STOPPED", 0, 0, DeploymentState.STOPPED),
    ("DELETED", 0, 0, DeploymentState.STOPPED),
    ("DEPLOYMENT_STATE_RUNNING", 0, 0, DeploymentState.STARTING),
    ("DEPLOYMENT_STATE_STOPPED", 0, 0, DeploymentState.STOPPED),
    ("DEPLOYMENT_STATE_DELETED", 0, 0, DeploymentState.STOPPED),
    ("DEPLOYMENT_STATE_PENDING", 0, 0, DeploymentState.PROVISIONING),
    ("DEPLOYMENT_STATE_FAILED", 0, 0, DeploymentState.FAILED),
    ("DEPLOYMENT_STATE_SCALED_TO_0", 0, 0, DeploymentState.STARTING),
    ("DEPLOYMENT_STATE_FROZEN", 0, 0, DeploymentState.STOPPED),
    ("DEPLOYMENT_STATE_BALANCE_STOPPED", 0, 0, DeploymentState.STOPPED),
    ("DEPLOYMENT_STATE_SHADOW_BANNED", 0, 0, DeploymentState.FAILED),
])
@pytest.mark.asyncio
async def test_status_never_claims_application_healthy(provider, runner, ambient_env, state, pending, failing, expected):
    account = ProviderAccount.ambient("lightning")
    record, _ = await provision(provider, account)
    data = runner.resources[("owner/tests", record.id)]
    data["desired_state"] = state
    data["status"].update(pending_replicas=pending, failing_replicas=failing, ready_replicas=0)
    status = await provider.status(record, account)
    assert status.state == expected
    assert status.endpoint_url == record.endpoint_url


@pytest.mark.asyncio
async def test_deleted_at_timestamp_means_stopped(provider, runner, ambient_env):
    account = ProviderAccount.ambient("lightning")
    record, _ = await provision(provider, account)
    runner.resources[("owner/tests", record.id)]["deleted_at"] = "2026-01-01T00:00:00Z"
    assert (await provider.status(record, account)).state == DeploymentState.STOPPED


@pytest.mark.parametrize("status", [{"failing_replicas": "many"}, "not-a-dict"])
@pytest.mark.asyncio
async def test_invalid_status_payload_is_transient(provider, runner, ambient_env, status):
    account = ProviderAccount.ambient("lightning")
    record, _ = await provision(provider, account)
    runner.resources[("owner/tests", record.id)]["status"] = status
    with pytest.raises(ProviderOperationError, match="invalid deployment status") as caught:
        await provider.status(record, account)
    assert caught.value.kind is FailureKind.TRANSIENT


@pytest.mark.asyncio
async def test_externally_deleted_is_stopped_and_stop_is_safe(provider, runner, ambient_env):
    account = ProviderAccount.ambient("lightning")
    record, _ = await provision(provider, account)
    runner.resources.clear()
    record.endpoint_url = "https://x" + HOST_SUFFIX
    status = await provider.status(record, account)
    assert status.state == DeploymentState.STOPPED
    assert status.endpoint_url == record.endpoint_url
    assert not await provider.resource_exists(record, account)
    await provider.stop(record, account)


@pytest.mark.asyncio
async def test_external_scaledown_still_allows_full_delete(provider, runner, ambient_env):
    account = ProviderAccount.ambient("lightning")
    record, _ = await provision(provider, account)
    runner.resources[("owner/tests", record.id)]["desired_state"] = "STOPPED"
    assert (await provider.status(record, account)).state == DeploymentState.STOPPED
    assert runner.resources
    await provider.stop(record, account)
    assert not runner.resources


@pytest.mark.asyncio
async def test_resource_snapshot_prefers_resource_id_and_scope(provider, runner, ambient_env):
    account = ProviderAccount.ambient("lightning")
    record, _ = await provision(provider, account)
    record.resource.resource_id = "dep_" + record.id
    snapshot = await provider.resource_snapshot(record, account)
    assert snapshot["name"] == record.id
    call = runner.of("inspect")[0]
    assert call.payload == {"target": "dep_" + record.id, "teamspace": "owner/tests"}


@pytest.mark.parametrize("resource", [None, "no-scope"])
@pytest.mark.asyncio
async def test_record_without_resource_identity_is_rejected(provider, runner, ambient_env, resource):
    from inferweave.domain.deployment_record import ResourceRef

    record = DeploymentRecord(
        id="x", model="fish-s2-pro", provider="lightning",
        resource=ResourceRef(name="n") if resource else None,
    )
    with pytest.raises(ProviderOperationError, match="no Lightning resource identity") as caught:
        await provider.status(record, ProviderAccount.ambient("lightning"))
    assert caught.value.kind is FailureKind.INVALID_REQUEST
    assert not runner.calls


# --- stop / ownership ----------------------------------------------------------------------------


@pytest.mark.parametrize("action", [AutostopAction.STOP, AutostopAction.DOWN])
@pytest.mark.asyncio
async def test_stop_and_down_both_fully_delete(provider, runner, ambient_env, action):
    account = ProviderAccount.ambient("lightning")
    record, _ = await provision(provider, account)
    await provider.stop(record, account, action)
    assert not runner.resources
    delete = runner.of("delete")[0]
    assert delete.payload == {"target": record.id, "teamspace": "owner/tests"}
    assert delete.deployment_id == record.id
    assert runner.ops()[-1] == "inspect"  # absence was confirmed


@pytest.mark.asyncio
async def test_delete_confirmation_polls_until_resource_disappears(
    provider, runner, ambient_env, monkeypatch
):
    sleeps = []
    real_sleep = asyncio.sleep

    async def fast_sleep(delay):
        sleeps.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(lp.asyncio, "sleep", fast_sleep)
    account = ProviderAccount.ambient("lightning")
    record, _ = await provision(provider, account)
    runner.delete_lag = 2
    await provider.stop(record, account)
    assert not runner.resources
    assert len(runner.of("inspect")) == 3  # two sightings, then confirmed absent
    assert sleeps == [5, 5]


@pytest.mark.asyncio
async def test_delete_not_confirmed_in_time_is_retryable_transient(
    provider, runner, ambient_env, monkeypatch
):
    monkeypatch.setattr(lp, "DELETE_CONFIRM_TIMEOUT_SECONDS", 0)
    account = ProviderAccount.ambient("lightning")
    record, _ = await provision(provider, account)
    runner.delete_lag = 100
    with pytest.raises(ProviderOperationError, match="not confirmed") as caught:
        await provider.stop(record, account)
    assert caught.value.kind is FailureKind.TRANSIENT
    assert caught.value.resource_may_exist is True
    assert runner.resources


@pytest.mark.asyncio
async def test_unowned_resource_cannot_be_deleted(provider, runner, ambient_env):
    account = ProviderAccount.ambient("lightning")
    record, *_ = new_record(provider, account)
    assert record.resource.owned is False
    runner.add_resource("owner/tests", record.id, foreign=True)
    with pytest.raises(ProviderOperationError, match="not owned") as caught:
        await provider.stop(record, account)
    assert caught.value.kind is FailureKind.INVALID_REQUEST
    assert caught.value.resource_may_exist is True
    assert "delete" not in runner.ops()
    assert runner.resources


@pytest.mark.asyncio
async def test_unowned_absent_resource_stops_without_delete(provider, runner, ambient_env):
    account = ProviderAccount.ambient("lightning")
    record, *_ = new_record(provider, account)
    await provider.stop(record, account)
    assert runner.ops() == ["inspect"]


@pytest.mark.asyncio
async def test_delete_failure_is_retryable(provider, runner, ambient_env):
    account = ProviderAccount.ambient("lightning")
    record, _ = await provision(provider, account)
    runner.delete_error = ProviderOperationError("cleanup denied", FailureKind.TRANSIENT)
    with pytest.raises(ProviderOperationError, match="cleanup denied"):
        await provider.stop(record, account)
    assert runner.resources
    runner.delete_error = None
    await provider.stop(record, account)
    assert not runner.resources


@pytest.mark.asyncio
async def test_resource_exists_reflects_remote_state(provider, runner, ambient_env):
    account = ProviderAccount.ambient("lightning")
    record, _ = await provision(provider, account)
    assert await provider.resource_exists(record, account)
    await provider.stop(record, account)
    assert not await provider.resource_exists(record, account)


# --- credential and account isolation ------------------------------------------------------------


def secret_free(call: Call, *sentinels: str) -> None:
    blob = repr((call.env, call.payload))
    for value in sentinels:
        assert value not in blob


@pytest.mark.asyncio
async def test_pooled_worker_env_is_private_and_secrets_travel_on_stdin(
    provider, runner, tmp_path, monkeypatch
):
    for name, value in AMBIENT_POISON.items():
        monkeypatch.setenv(name, value)
    before = dict(os.environ)
    manager = pool_manager(tmp_path)
    account = account_of(manager, "acct-a")

    record, _ = await provision(provider, account)
    await provider.status(record, account)
    await provider.stop(record, account)

    assert {c.op for c in runner.calls} == {"whoami", "start", "inspect", "delete"}
    home = home_for(provider, "acct-a")
    assert home.is_dir()
    for call in runner.calls:
        assert call.account_id == "acct-a"
        assert call.secrets == ACCOUNT_SECRETS["acct-a"]
        # No env-supplied credentials at all: only the passthrough allowlist plus account-private paths.
        assert set(call.env) - set(_PASSTHROUGH_ENV) == {
            "HOME", "USERPROFILE", "LIGHTNING_CREDENTIAL_PATH",
        }
        assert call.env["HOME"] == call.env["USERPROFILE"] == str(home)
        # The SDK must not fall back to the user's own ~/.lightning/credentials.json.
        assert call.env["LIGHTNING_CREDENTIAL_PATH"] == str(home / ".lightning" / "none.json")
        blob = repr((call.env, call.payload))
        assert "AMBIENT-" not in blob
        for value in ACCOUNT_SECRETS["acct-a"].values():
            assert value not in blob
        # Account B's credentials are never present in account A's worker calls.
        assert "SENTINEL-key-b" not in repr(call) and "SENTINEL-user-b" not in repr(call)
    assert dict(os.environ) == before  # the parent environment is never modified


@pytest.mark.asyncio
async def test_pooled_secrets_never_enter_payloads_or_records(provider, runner, tmp_path):
    account = account_of(pool_manager(tmp_path), "acct-b")
    record, _ = await provision(provider, account)
    for call in runner.calls:
        secret_free(call, *ACCOUNT_SECRETS["acct-b"].values())
    dump = record.model_dump_json()
    for value in ACCOUNT_SECRETS["acct-b"].values():
        assert value not in dump


@pytest.mark.asyncio
async def test_ambient_account_passes_no_secrets(provider, runner, ambient_env):
    account = ProviderAccount.ambient("lightning")
    record, _ = await provision(provider, account)
    await provider.status(record, account)
    await provider.stop(record, account)
    for call in runner.calls:
        assert call.secrets == {}
        assert call.account_id == "ambient"
        # Ambient keeps the user's own environment (native SDK behaviour).
        assert call.env["LIGHTNING_API_KEY"] == KEY
        assert call.env["LIGHTNING_USER_ID"] == "test-user"
        assert "LIGHTNING_CREDENTIAL_PATH" not in call.env or (
            call.env["LIGHTNING_CREDENTIAL_PATH"] == os.environ.get("LIGHTNING_CREDENTIAL_PATH")
        )
    assert not (provider.state_dir / "lightning").exists()


@pytest.mark.asyncio
async def test_two_accounts_each_use_only_their_own_credentials_and_teamspace(
    provider, runner, tmp_path
):
    manager = pool_manager(tmp_path)
    records = {}
    for account_id in ("acct-a", "acct-b"):
        account = account_of(manager, account_id)
        records[account_id], _ = await provision(provider, account)
        await provider.status(records[account_id], account)

    for account_id, record in records.items():
        mine = [c for c in runner.calls if c.account_id == account_id]
        assert {c.op for c in mine} == {"whoami", "start", "inspect"}
        for call in mine:
            assert call.secrets == ACCOUNT_SECRETS[account_id]
            assert call.env["HOME"] == str(home_for(provider, account_id))
        start = next(c for c in mine if c.op == "start")
        assert start.payload["teamspace"] == ACCOUNT_TEAMSPACE[account_id]
        assert record.resource.scope == ACCOUNT_TEAMSPACE[account_id]
        assert next(c for c in mine if c.op == "inspect").payload["teamspace"] == (
            ACCOUNT_TEAMSPACE[account_id]
        )
    assert home_for(provider, "acct-a") != home_for(provider, "acct-b")


@pytest.mark.asyncio
async def test_per_deploy_teamspace_overrides_account_teamspace_in_worker(provider, runner, tmp_path):
    account = account_of(pool_manager(tmp_path), "acct-a")
    record, _ = await provision(
        provider, account, custom_args={"lightning": {"teamspace": "other/space"}}
    )
    assert runner.of("start")[0].payload["teamspace"] == "other/space"
    assert record.resource.scope == "other/space"
    await provider.status(record, account)
    assert runner.of("inspect")[0].payload["teamspace"] == "other/space"


# --- endpoint auth -------------------------------------------------------------------------------


@pytest.mark.parametrize("provider_name,url,applied", [
    ("lightning", ENDPOINT, True), ("modal", ENDPOINT, False),
    ("lightning", "https://evil.cloudspaces.litng.ai.attacker.test", False),
    ("lightning", "http://8080-dep-test-d.cloudspaces.litng.ai", False),
    ("lightning", "https://example.test", False),
])
def test_endpoint_auth_is_scoped_and_repr_safe(ambient_env, provider_name, url, applied):
    auth = LightningEndpointAuth()
    assert bool(auth.headers_for(provider_name, url)) == applied
    assert KEY not in repr(auth)
    if applied:
        assert auth.headers_for(provider_name, url) == {"Authorization": f"Bearer {KEY}"}


def test_pooled_endpoint_auth_uses_the_owning_accounts_key(tmp_path, monkeypatch):
    monkeypatch.setenv("LIGHTNING_API_KEY", KEY)
    auth = LightningEndpointAuth()
    manager = pool_manager(tmp_path)
    for account_id, secrets in ACCOUNT_SECRETS.items():
        headers = auth.headers_for("lightning", ENDPOINT, account_of(manager, account_id))
        assert headers == {"Authorization": f"Bearer {secrets['LIGHTNING_API_KEY']}"}
    assert KEY not in repr(auth.headers_for("lightning", ENDPOINT, account_of(manager, "acct-a")))


def test_typed_autoscale_is_independent_of_destroy_timer():
    request, _, _ = recipe(autostop_mins=1440, custom_args={
        "lightning": {"min_replicas": 0, "max_replicas": 2, "idle_threshold_seconds": 90},
    })
    assert request.options.autostop.idle_minutes == 1440
    assert request.options.provider.lightning.idle_threshold_seconds == 90
    with pytest.raises(ValueError, match="must not exceed"):
        LightningOptions(min_replicas=3, max_replicas=1)


# --- runtime command -----------------------------------------------------------------------------


def test_launch_arguments_stay_literal():
    argument = "hello; echo bad; $(echo expanded)"
    runtime = RuntimeSpec(name="test", docker_image="python", run_command="",
                          run_args=[sys.executable, "-c", "import sys;print(sys.argv[1])", argument])
    command = shlex.split(runtime_command(runtime))
    # Tokenization is portable; the generated program is only executed on POSIX elsewhere.
    assert command[0] == "-c"
    assert shlex.quote(argument) in command[1]


def test_runtime_command_bootstraps_worker_sources_only_when_needed():
    plain = RuntimeSpec(name="t", docker_image="python", run_command="", run_args=["echo", "hi"])
    assert "/tmp/iw-runtime" not in runtime_command(plain)
    workers = RuntimeSpec(
        name="t", docker_image="python", run_command="",
        run_args=["python", "-m", "inferweave.workers.image_server"],
    )
    assert "/tmp/iw-runtime" in runtime_command(workers)


# --- installed SDK contract ----------------------------------------------------------------------


@pytest.mark.parametrize("gpu,count", [
    (gpu, count)
    for gpu in ("T4", "L4", "L40S", "A100-40GB", "A100-80GB", "H100", "H200", "B200")
    for count in ((1, 8) if gpu == "B200" else (1, 2, 4, 8))
])
def test_installed_sdk_public_contract(monkeypatch, gpu, count):
    import importlib.util
    import inspect

    if importlib.util.find_spec("lightning_sdk") is None:
        pytest.skip("Install lightning extra for the installed SDK contract check.")
    monkeypatch.setenv("LIGHTNING_DISABLE_VERSION_CHECK", "1")
    import lightning_sdk as sdk
    from lightning_sdk import deployment as config

    assert sdk.__version__ == "2026.10.1"
    parameters = inspect.signature(sdk.Deployment.start).parameters
    assert {"image", "machine", "ports", "entrypoint", "command", "include_credentials",
            "env", "auth", "health_check", "autoscale", "replicas", "spot"} <= set(parameters)
    assert {"teamspace", "name"} <= set(inspect.signature(sdk.Deployment).parameters)
    assert hasattr(config, "ApiKeyAuth")
    assert config.HttpHealthCheck(path="/health", port=8000).port == 8000
    request, profile, _ = recipe(gpu_type=gpu, num_gpus=count)
    # Use a small model requirement so T4 mapping is tested independently of Fish VRAM.
    profile = profile.model_copy(deep=True)
    profile.hardware.min_vram_gb = 1
    machine = getattr(sdk.Machine, LightningProvider.machine_name(request, profile))
    assert machine.accelerator_count == count
    assert machine.family == gpu.split("-")[0]


def test_paid_integration_requires_explicit_opt_in(monkeypatch):
    from test_lightning_integration import _skip_reason

    monkeypatch.delenv("INFERWEAVE_LIGHTNING_INTEGRATION", raising=False)
    monkeypatch.setenv("LIGHTNING_USER_ID", "user")
    monkeypatch.setenv("LIGHTNING_API_KEY", KEY)
    monkeypatch.setenv("LIGHTNING_TEAMSPACE", "owner/tests")
    assert "explicitly enable" in _skip_reason()
    monkeypatch.setenv("INFERWEAVE_LIGHTNING_INTEGRATION", "1")
    assert _skip_reason() is None
    monkeypatch.delenv("LIGHTNING_API_KEY")
    assert "LIGHTNING_API_KEY" in _skip_reason()


# --- through the SDK -----------------------------------------------------------------------------


def wav_bytes():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\x01\x00" * 24)
    return buffer.getvalue()


def make_lightning_weave(
    tmp_path: Path,
    runner: FakeRunner,
    *,
    accounts: AccountManager | None = None,
    handler=None,
    probe=None,
    state_name: str = "state.db",
) -> InferWeave:
    """An InferWeave "process" whose Lightning provider talks to ``runner``."""
    manager = accounts or AccountManager()
    probe = probe or MockHealthcheckProbeAdapter(default_healthy=True)
    healthcheck = HealthcheckService(probe_port=probe)
    endpoint_auth = default_endpoint_auth()
    router = ProviderRouter(endpoint_auth=endpoint_auth, state_dir=tmp_path / "accounts")
    router.register(LightningProvider(runner=runner, state_dir=tmp_path / "accounts"))
    lifecycle = LifecycleService(
        watchdog_port=MockWatchdogAdapter(),
        repository=SqliteDeploymentRepository(tmp_path / state_name),
        healthcheck_service=healthcheck,
        provider_resolver=router.get,
        endpoint_auth=endpoint_auth,
        activity_persist_interval_seconds=0.0,
        accounts=manager,
    )
    weave = InferWeave(
        router=router,
        healthcheck_service=healthcheck,
        lifecycle_service=lifecycle,
        endpoint_auth=endpoint_auth,
        inference_config=FAST_RETRIES,
        inference_http_client=(
            httpx.AsyncClient(transport=httpx.MockTransport(handler)) if handler else None
        ),
        accounts=manager,
    )
    for model_id in (AUDIO_MODEL, IMAGE_MODEL):
        profile = weave.registry.get(model_id).model_copy(deep=True)
        profile.healthcheck = HealthcheckConfig(
            port=profile.healthcheck.port,
            path=profile.healthcheck.path,
            initial_delay_seconds=0,
            timeout_seconds=0.2,
            probe_interval_seconds=0.01,
        )
        weave.register_model(profile)
    return weave


@pytest.fixture
def weave_factory(tmp_path):
    created = []

    def build(runner, **kwargs):
        weave = make_lightning_weave(tmp_path, runner, **kwargs)
        created.append(weave)
        return weave

    yield build
    # Closed in a sync fixture teardown via a fresh loop; tests that need an explicit close do it.
    for weave in created:
        asyncio.run(weave.close())


@pytest.mark.asyncio
async def test_ambient_deploy_records_owner_and_stops(runner, weave_factory, ambient_env):
    weave = weave_factory(runner)
    dep = await weave.deploy(AUDIO_MODEL, provider="lightning", wait_for_ready=False)
    assert dep.state == DeploymentState.STARTING
    assert dep.account == "ambient"
    record = await weave.lifecycle_service.get_record(dep.id)
    assert record.account == "ambient"
    assert record.resource.name == dep.id and record.resource.owned
    assert record.resource.resource_id == "dep_" + dep.id
    assert record.resource.scope == "owner/tests"
    assert not record.needs_reconciliation
    serialized = record.model_dump_json()
    assert KEY not in serialized and "test-user" not in serialized

    await dep.wait_for_ready()
    assert dep.state == DeploymentState.HEALTHY
    await dep.stop()
    assert not runner.resources
    assert (await weave.lifecycle_service.get_record(dep.id)).state == DeploymentState.STOPPED
    assert all(c.secrets == {} for c in runner.calls)


@pytest.mark.asyncio
async def test_missing_credentials_fail_before_resources(runner, weave_factory, ambient_env, monkeypatch):
    monkeypatch.delenv("LIGHTNING_API_KEY")
    weave = weave_factory(runner)
    with pytest.raises(ProviderAuthError, match="LIGHTNING_USER_ID"):
        await weave.deploy(AUDIO_MODEL, provider="lightning", wait_for_ready=False)
    assert not runner.calls and not runner.resources
    assert await weave.list_records() == []


@pytest.mark.parametrize("custom_args", [
    {"engine_args": {"LIGHTNING_API_KEY": KEY}},
    {"extra_cli_args": ["--token", KEY]},
    {"extra_env": {"OTHER_NAME": KEY}},
])
@pytest.mark.asyncio
async def test_platform_credentials_rejected_before_any_persistence(
    runner, weave_factory, ambient_env, custom_args
):
    weave = weave_factory(runner)
    with pytest.raises(ProviderAuthError, match="must not be injected") as error:
        await weave.deploy(AUDIO_MODEL, provider="lightning", custom_args=custom_args)
    assert KEY not in str(error.value)
    assert not runner.calls
    assert await weave.list_records() == []


@pytest.mark.asyncio
async def test_platform_credentials_cannot_be_copied_into_runtime_env(runner, weave_factory, ambient_env):
    weave = weave_factory(runner)
    with pytest.raises(ProviderAuthError, match="must not be injected"):
        await weave.deploy(AUDIO_MODEL, provider="lightning", env={"LIGHTNING_API_KEY": KEY})
    assert not runner.calls


@pytest.mark.asyncio
async def test_unrelated_runtime_secrets_are_allowed_through_the_sdk(runner, weave_factory, ambient_env):
    weave = weave_factory(runner)
    dep = await weave.deploy(
        AUDIO_MODEL, provider="lightning", wait_for_ready=False,
        env={"MODEL_TOKEN": "unrelated-model-secret"},
    )
    assert runner.of("start")[0].payload["env"]["MODEL_TOKEN"] == "unrelated-model-secret"
    await dep.stop()


@pytest.mark.asyncio
async def test_collision_preserves_user_resource_and_never_fails_over(runner, weave_factory, ambient_env):
    runner.collide = True
    weave = weave_factory(runner)
    with pytest.raises(ProviderOperationError, match="collision") as caught:
        await weave.deploy(AUDIO_MODEL, provider="lightning")
    assert caught.value.kind is FailureKind.INVALID_REQUEST
    assert "delete" not in runner.ops()
    [foreign] = runner.resources.values()
    assert foreign["foreign"] is True
    [record] = await weave.list_records()
    assert record.state == DeploymentState.FAILED
    assert record.resource.owned is False
    assert not record.needs_reconciliation


@pytest.mark.asyncio
async def test_unclassified_sdk_error_text_never_leaks(runner, weave_factory, ambient_env, caplog):
    runner.whoami_error = RuntimeError(f"SDK body has {KEY}")
    weave = weave_factory(runner)
    with pytest.raises(ProviderOperationError) as caught:
        await weave.deploy(AUDIO_MODEL, provider="lightning")
    assert caught.value.kind is FailureKind.TRANSIENT
    assert KEY not in str(caught.value) + repr(caught.value) + caplog.text
    assert caught.value.__suppress_context__
    assert not runner.resources and "start" not in runner.ops()
    [record] = await weave.list_records()
    assert record.state == DeploymentState.FAILED and not record.needs_reconciliation


@pytest.mark.asyncio
async def test_partial_create_after_uncertain_start_failure_is_removed(
    runner, weave_factory, ambient_env
):
    runner.start_error = ProviderOperationError("start timed out", FailureKind.TRANSIENT)
    weave = weave_factory(runner)
    with pytest.raises(ProviderOperationError) as caught:
        await weave.deploy(AUDIO_MODEL, provider="lightning")
    assert caught.value.kind is FailureKind.TRANSIENT
    assert not runner.resources  # the partially created Deployment was reconciled away
    assert runner.of("delete")
    [record] = await weave.list_records()
    assert record.state == DeploymentState.FAILED and not record.needs_reconciliation
    assert record.resource.owned is True


@pytest.mark.asyncio
async def test_unclassified_start_exception_cleans_created_resource(runner, weave_factory, ambient_env):
    runner.start_error = RuntimeError(f"SDK body has {KEY}")
    weave = weave_factory(runner)
    with pytest.raises(ProvisioningUncertainError) as caught:
        await weave.deploy(AUDIO_MODEL, provider="lightning")
    assert runner.resources
    await weave.reconcile(min_age_seconds=0, confirmed_settled=(caught.value.deployment_id,))
    assert not runner.resources


@pytest.mark.parametrize("stage", ["endpoint", "foreign_endpoint", "identity"])
@pytest.mark.asyncio
async def test_failure_cleanup(runner, weave_factory, ambient_env, stage):
    if stage == "endpoint":
        runner.urls = []
    elif stage == "foreign_endpoint":
        runner.urls = ["https://foreign.example"]
    else:
        runner.identity["ambient"] = "scoped-api-key"
    weave = weave_factory(runner)
    with pytest.raises(ProviderOperationError):
        await weave.deploy(AUDIO_MODEL, provider="lightning")
    assert not runner.resources
    assert ("start" in runner.ops()) == (stage != "identity")


@pytest.mark.asyncio
async def test_uncertain_cleanup_stops_without_retrying(runner, weave_factory, ambient_env):
    from inferweave.core.exceptions import ProvisioningUncertainError

    runner.start_error = ProviderOperationError("timed out", FailureKind.TRANSIENT)
    runner.delete_error = ProviderOperationError("cleanup denied", FailureKind.TRANSIENT)
    weave = weave_factory(runner)
    with pytest.raises(ProvisioningUncertainError):
        await weave.deploy(AUDIO_MODEL, provider="lightning")
    [record] = await weave.list_records()
    assert record.needs_reconciliation and record.account == "ambient"
    assert len(runner.of("start")) == 1
    runner.delete_error = None
    [fixed] = await weave.reconcile()
    assert fixed.id == record.id and not fixed.needs_reconciliation
    assert not runner.resources


@pytest.mark.asyncio
async def test_cancellation_mid_start_cleans_created_resource(runner, weave_factory, ambient_env):
    runner.start_gate = asyncio.Event()  # never set: the start call hangs until cancelled
    weave = weave_factory(runner)
    task = asyncio.create_task(weave.deploy(AUDIO_MODEL, provider="lightning"))
    await asyncio.wait_for(runner.start_entered.wait(), 3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    [pending] = await weave.list_records()
    assert pending.needs_reconciliation
    await weave.reconcile(min_age_seconds=0, confirmed_settled=(pending.id,))
    assert not runner.resources


@pytest.mark.asyncio
async def test_stop_failure_keeps_retryable_state(runner, weave_factory, ambient_env):
    weave = weave_factory(runner)
    dep = await weave.deploy(AUDIO_MODEL, provider="lightning")
    runner.delete_error = ProviderOperationError("cleanup denied", FailureKind.TRANSIENT)
    with pytest.raises(ProviderOperationError):
        await dep.stop()
    assert (await weave.lifecycle_service.get_record(dep.id)).state != DeploymentState.STOPPED
    assert runner.resources
    runner.delete_error = None
    await dep.stop()
    assert not runner.resources


@pytest.mark.asyncio
async def test_destroy_protects_active_request_and_restores_long_idle(runner, weave_factory, ambient_env):
    weave = weave_factory(runner)
    dep = await weave.deploy(AUDIO_MODEL, provider="lightning")
    lifecycle = weave.lifecycle_service
    state = lifecycle.get_state(dep.id)
    state.last_activity_at = datetime.now(UTC) - timedelta(minutes=120)
    lifecycle.begin_request(dep.id)
    assert not await lifecycle.check_and_autostop(dep.id)
    assert runner.resources
    lifecycle.end_request(dep.id)
    state.last_activity_at = datetime.now(UTC) - timedelta(minutes=120)
    assert await lifecycle.check_and_autostop(dep.id)
    assert not runner.resources


@pytest.mark.asyncio
async def test_cold_start_retry_keeps_auth_and_activity(runner, weave_factory, ambient_env):
    attempts = []

    def handler(request):
        attempts.append(request)
        assert request.headers["Authorization"] == f"Bearer {KEY}"
        if len(attempts) == 1:
            return httpx.Response(503)
        return httpx.Response(200, content=wav_bytes())

    weave = weave_factory(runner, handler=handler)
    dep = await weave.deploy(AUDIO_MODEL, provider="lightning")
    assert await dep.synthesize("retry") == wav_bytes()
    assert len(attempts) == 2
    assert not weave.lifecycle_service.get_state(dep.id).in_flight_requests
    await dep.stop()


@pytest.mark.asyncio
async def test_image_workload_uses_existing_transport_and_worker(runner, weave_factory, ambient_env):
    import base64

    png = b"\x89PNG\r\n\x1a\n" + b"image"

    def handler(request):
        assert request.headers["Authorization"] == f"Bearer {KEY}"
        assert str(request.url).endswith("/v1/images/generations")
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(png).decode()}]})

    weave = weave_factory(runner, handler=handler)
    dep = await weave.deploy(IMAGE_MODEL, provider="lightning", gpu_type="L40S")
    assert await dep.render("fox") == [png]
    assert "/tmp/iw-runtime" in runner.of("start")[0].payload["command"]
    await dep.stop()


# readiness / post-provision failures clean the paid resource

async def assert_failed_deployment_cleaned(weave, runner):
    assert not runner.resources
    assert weave.list_deployments() == []
    [record] = await weave.list_records()
    assert record.state == DeploymentState.STOPPED
    state = weave.lifecycle_service.get_state(record.id)
    assert state.is_stopped and state.stopped_at is not None
    watchdog = weave.lifecycle_service._watchdog
    assert record.id not in watchdog.scheduled_checks
    assert record.id in watchdog.cancelled_checks
    assert not await weave.lifecycle_service.check_and_autostop(
        record.id, now=datetime.now(UTC) + timedelta(hours=2),
    )
    assert runner.of("delete")


@pytest.mark.asyncio
async def test_readiness_failure_cleans_resources(runner, weave_factory, ambient_env):
    weave = weave_factory(runner, probe=MockHealthcheckProbeAdapter(default_healthy=False))
    with pytest.raises(HealthcheckTimeoutError):
        await weave.deploy(AUDIO_MODEL, provider="lightning")
    await assert_failed_deployment_cleaned(weave, runner)


@pytest.mark.asyncio
async def test_health_auth_failure_cleans_resources(runner, weave_factory, ambient_env, monkeypatch):
    weave = weave_factory(runner)

    async def fail(*args, **kwargs):
        raise ProviderAuthError("endpoint denied")

    monkeypatch.setattr(weave.healthcheck_service, "wait_for_ready", fail)
    with pytest.raises(ProviderAuthError):
        await weave.deploy(AUDIO_MODEL, provider="lightning")
    await assert_failed_deployment_cleaned(weave, runner)


@pytest.mark.parametrize("stage", ["cancel", "registration", "wiring"])
@pytest.mark.asyncio
async def test_post_provision_failure_cleans_lifecycle(runner, weave_factory, ambient_env, monkeypatch, stage):
    weave = weave_factory(runner)

    async def fail(*args, **kwargs):
        if stage == "cancel":
            raise asyncio.CancelledError
        raise RuntimeError("registration failed")

    if stage == "registration":
        monkeypatch.setattr(weave.lifecycle_service._watchdog, "schedule_check", fail)
    elif stage == "wiring":
        def fail_wiring(*args, **kwargs):
            raise RuntimeError("registration failed")
        monkeypatch.setattr(weave, "_wire_deployment", fail_wiring)
    else:
        monkeypatch.setattr(weave.healthcheck_service, "wait_for_ready", fail)
    with pytest.raises(asyncio.CancelledError if stage == "cancel" else RuntimeError):
        await weave.deploy(AUDIO_MODEL, provider="lightning")
    await assert_failed_deployment_cleaned(weave, runner)


@pytest.mark.asyncio
async def test_readiness_cleanup_failure_preserves_error_and_retry(
    runner, weave_factory, ambient_env, monkeypatch
):
    weave = weave_factory(runner)
    original_error = ProviderAuthError("endpoint denied")

    async def fail(*args, **kwargs):
        raise original_error

    monkeypatch.setattr(weave.healthcheck_service, "wait_for_ready", fail)
    runner.delete_error = ProviderOperationError("cleanup denied", FailureKind.TRANSIENT)
    with pytest.raises(ProviderAuthError) as caught:
        await weave.deploy(AUDIO_MODEL, provider="lightning")
    assert caught.value is original_error
    [deployment] = weave.list_deployments()
    assert runner.resources
    assert not weave.lifecycle_service.get_state(deployment.id).is_stopped
    assert (await weave.lifecycle_service.get_record(deployment.id)).state != DeploymentState.STOPPED
    runner.delete_error = None
    await weave.stop(deployment.id)
    assert not runner.resources
    assert weave.lifecycle_service.get_state(deployment.id).is_stopped


# two-account pool: deploy -> status -> stop with account affinity


def expect_affinity(runner: FakeRunner, dep, record) -> None:
    """Every worker call about ``dep`` carried exactly its owner's credentials and teamspace."""
    mine = [c for c in runner.calls if c.deployment_id == dep.id]
    assert {c.op for c in mine} >= {"start"}
    for call in mine:
        assert call.account_id == dep.account
        assert call.secrets == ACCOUNT_SECRETS[dep.account]
        assert call.payload["teamspace"] == ACCOUNT_TEAMSPACE[dep.account]
    assert record.account == dep.account
    assert record.resource.scope == ACCOUNT_TEAMSPACE[dep.account]


@pytest.mark.asyncio
async def test_two_account_pool_deploy_status_stop_keeps_account_affinity(
    runner, weave_factory, tmp_path, monkeypatch
):
    for name, value in AMBIENT_POISON.items():
        monkeypatch.setenv(name, value)
    manager = pool_manager(tmp_path)
    keys = {a: s["LIGHTNING_API_KEY"] for a, s in ACCOUNT_SECRETS.items()}
    seen: list[tuple[str, str]] = []

    def handler(request):
        seen.append((request.url.host, request.headers["Authorization"]))
        return httpx.Response(200, content=wav_bytes(), headers={"content-type": "audio/wav"})

    weave = weave_factory(runner, accounts=manager, handler=handler)
    first = await weave.deploy(AUDIO_MODEL, provider="lightning", wait_for_ready=False)
    second = await weave.deploy(AUDIO_MODEL, provider="lightning", wait_for_ready=False)
    assert {first.account, second.account} == {"acct-a", "acct-b"}  # round robin spread the load

    for dep in (first, second):
        record = await weave.lifecycle_service.get_record(dep.id)
        expect_affinity(runner, dep, record)
        assert (await dep.refresh()).account == dep.account
        assert await dep.synthesize("hi") == wav_bytes()
        assert (urlhost(dep), f"Bearer {keys[dep.account]}") in seen
        assert runner.of("inspect")  # status went through the worker as the owner

    # The health/inference traffic of each deployment carried only its own account's key.
    assert {h for _, h in seen} <= {f"Bearer {k}" for k in keys.values()}
    assert len({h for _, h in seen}) == 2

    for dep in (first, second):
        await dep.stop()
        record = await weave.lifecycle_service.get_record(dep.id)
        expect_affinity(runner, dep, record)
        assert record.state == DeploymentState.STOPPED
    assert not runner.resources
    for call in runner.calls:  # nothing ever ran as the ambient account or leaked ambient values
        assert call.account_id in ACCOUNT_SECRETS
        assert "AMBIENT-" not in repr((call.env, call.payload))
    health = {h.account_id: h for h in weave.account_health()}
    assert set(health) == {"acct-a", "acct-b"}


def urlhost(dep) -> str:
    from urllib.parse import urlsplit

    return urlsplit(dep.endpoint_url).hostname


@pytest.mark.asyncio
async def test_pinned_account_is_used_without_failover(runner, weave_factory, tmp_path):
    weave = weave_factory(runner, accounts=pool_manager(tmp_path))
    dep = await weave.deploy(
        AUDIO_MODEL, provider="lightning", account="acct-b", wait_for_ready=False
    )
    assert dep.account == "acct-b"
    assert {c.account_id for c in runner.calls} == {"acct-b"}
    assert runner.of("start")[0].payload["teamspace"] == ACCOUNT_TEAMSPACE["acct-b"]
    await dep.stop()


@pytest.mark.asyncio
async def test_unknown_pinned_account_is_refused(runner, weave_factory, tmp_path):
    weave = weave_factory(runner, accounts=pool_manager(tmp_path))
    with pytest.raises(AccountUnavailableError):
        await weave.deploy(AUDIO_MODEL, provider="lightning", account="nope")
    assert not runner.calls


@pytest.mark.asyncio
async def test_scoped_key_account_fails_over_to_the_next_account(runner, weave_factory, tmp_path):
    runner.identity["acct-a"] = "scoped-api-key"
    manager = pool_manager(tmp_path, strategy="failover")
    weave = weave_factory(runner, accounts=manager)
    dep = await weave.deploy(AUDIO_MODEL, provider="lightning", wait_for_ready=False)
    assert dep.account == "acct-b"
    # acct-a only ever got the identity check, with its own credentials.
    a_calls = [c for c in runner.calls if c.account_id == "acct-a"]
    assert [c.op for c in a_calls] == ["whoami"]
    assert a_calls[0].secrets == ACCOUNT_SECRETS["acct-a"]
    assert all(c.secrets == ACCOUNT_SECRETS["acct-b"] for c in runner.calls if c.account_id == "acct-b")
    records = {r.id: r for r in await weave.list_records()}
    assert len(records) == 2
    failed = next(r for r in records.values() if r.account == "acct-a")
    assert failed.state == DeploymentState.FAILED and not failed.needs_reconciliation
    assert records[dep.id].account == "acct-b"
    await dep.stop()


@pytest.mark.asyncio
async def test_removed_owner_account_is_never_replaced_by_another(runner, weave_factory, tmp_path):
    manager = pool_manager(tmp_path)
    weave = weave_factory(runner, accounts=manager)
    dep = await weave.deploy(
        AUDIO_MODEL, provider="lightning", account="acct-a", wait_for_ready=False
    )
    only_b = AccountsConfig({"lightning": ProviderAccounts(accounts=[
        lightning_account("acct-b", user_id_env="ACCT_B_USER", api_key_env="ACCT_B_KEY",
                          teamspace=ACCOUNT_TEAMSPACE["acct-b"]),
    ])}, state_dir=tmp_path / "accounts")
    replacement = AccountManager(only_b, environ=ACCOUNT_ENV)
    weave.accounts = replacement
    weave.lifecycle_service.accounts = replacement
    before = len(runner.calls)
    with pytest.raises(AccountUnavailableError):
        await dep.stop()
    assert len(runner.calls) == before
    assert runner.resources  # the resource is untouched rather than deleted as another account


@pytest.mark.asyncio
async def test_restart_attach_refresh_stop_with_legacy_lightning_record(
    runner, weave_factory, ambient_env, tmp_path, monkeypatch
):
    first = weave_factory(runner)
    dep = await first.deploy(AUDIO_MODEL, provider="lightning")
    record = await first.lifecycle_service.get_record(dep.id)
    legacy = record.model_dump(mode="json")
    resource = legacy.pop("resource")
    legacy.pop("account")
    legacy.pop("owner_fingerprint")
    legacy["lightning"] = {
        "name": resource["name"], "teamspace": resource["scope"],
        "resource_id": resource["resource_id"], "owned": resource["owned"],
    }
    await first.close()
    previous_calls = len(runner.calls)
    with sqlite3.connect(tmp_path / "state.db") as conn:
        conn.execute("UPDATE deployments SET data_json = ? WHERE id = ?", (json.dumps(legacy), dep.id))

    # Ambient config may change after an upgrade; operations must use the persisted scope.
    monkeypatch.setenv("LIGHTNING_TEAMSPACE", "other/space")
    second = weave_factory(runner)
    attached = await second.attach(dep.id)
    assert attached.account == "ambient"
    assert (await attached.refresh()).state == DeploymentState.HEALTHY
    await attached.stop()
    assert not runner.resources
    calls = [c for c in runner.calls[previous_calls:] if c.op in {"inspect", "delete"}]
    assert calls
    assert all(c.payload["teamspace"] == "owner/tests" for c in calls)
    assert all(c.payload["target"] == resource["resource_id"] for c in calls)


@pytest.mark.asyncio
async def test_restart_attach_find_health_inference_stop_with_pooled_account(
    runner, weave_factory, tmp_path
):
    from test_endpoint_auth import HeaderRecordingProbe

    requests = []

    def handler(request):
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer SENTINEL-key-b"
        return httpx.Response(200, content=wav_bytes(), headers={"content-type": "audio/wav"})

    first = weave_factory(runner, accounts=pool_manager(tmp_path), handler=handler)
    dep = await first.deploy(AUDIO_MODEL, provider="lightning", account="acct-b")
    assert await dep.synthesize("first") == wav_bytes()
    record = await first.lifecycle_service.get_record(dep.id)
    record.last_activity_at = datetime.now(UTC) - timedelta(minutes=60)
    await first.lifecycle_service.repository.save(record)
    await first.close()

    probe = HeaderRecordingProbe()
    second = weave_factory(
        runner, accounts=pool_manager(tmp_path), handler=handler, probe=probe
    )
    attached = await second.attach(dep.id)
    assert attached is not dep and attached.account == "acct-b"
    assert attached.last_activity_at == record.last_activity_at
    assert await second.find(model=dep.model, provider="lightning") is attached
    assert (await attached.refresh()).state == DeploymentState.HEALTHY
    assert (await attached.check_health()).is_healthy
    await attached.wait_for_ready()
    assert await attached.synthesize("recovered") == wav_bytes()
    assert len(probe.headers_seen) == 3
    assert all(h.get("Authorization") == "Bearer SENTINEL-key-b" for h in probe.headers_seen)
    assert all(str(r.url).endswith("/v1/tts") for r in requests)
    assert not second.lifecycle_service.get_state(dep.id).in_flight_requests
    await attached.stop()
    assert not runner.resources
    assert (await second.lifecycle_service.get_record(dep.id)).state == DeploymentState.STOPPED
    assert {c.account_id for c in runner.calls} == {"acct-b"}
