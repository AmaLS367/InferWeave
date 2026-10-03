"""Offline Lightning SDK doubles; no test here may contact Lightning."""

import asyncio
import io
import json
import subprocess
import sys
import threading
import time
import wave
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import httpx
import pytest
from support import make_weave

from inferweave import DeploymentState, LightningEndpointAuth, LightningOptions
from inferweave.core.exceptions import (
    DeploymentError,
    InsufficientVramError,
    ProviderAuthError,
    ProviderPlatformError,
)
from inferweave.domain.deployment_record import (
    DeploymentRecord,
    LightningDeploymentMetadata,
)
from inferweave.models.deployment import DeploymentRequest
from inferweave.providers.lightning_provider import (
    LightningProvider,
    _blocking,
    runtime_command,
)
from inferweave.registry.base import ModelRegistry
from inferweave.runtimes.templates import get_runtime_template

ENDPOINT = "https://8080-dep-test-d.cloudspaces.litng.ai"
KEY = "test-lightning-secret"


class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@pytest.fixture
def cloud(monkeypatch):
    monkeypatch.setattr("inferweave.providers.lightning_provider.ENDPOINT_DISCOVERY_TIMEOUT_SECONDS", 0)
    monkeypatch.setenv("LIGHTNING_USER_ID", "test-user")
    monkeypatch.setenv("LIGHTNING_API_KEY", KEY)
    monkeypatch.setenv("LIGHTNING_TEAMSPACE", "owner/tests")
    resources = {}
    calls = []
    switches = SimpleNamespace(
        start_error=None, urls=[ENDPOINT], identity="user", collision=False,
        delete_error=False, delay=0,
    )

    class Remote:
        def __init__(self, name, teamspace):
            calls.append(("lookup", name, teamspace))
            self.name = name
            self.id = "dep_" + name
            self.is_started = switches.collision

        def start(self, **kwargs):
            if switches.delay:
                time.sleep(switches.delay)
            calls.append(("start", kwargs))
            resources[self.id] = {
                "desired_state": "RUNNING",
                "status": {"ready_replicas": 1, "pending_replicas": 0, "failing_replicas": 0},
                "name": self.name,
            }
            if switches.start_error:
                raise switches.start_error

        @property
        def urls(self):
            return switches.urls

    def cli(*args, missing_ok=False):
        calls.append(("cli", args))
        if args[:2] == ("auth", "whoami"):
            return json.dumps({"auth_type": switches.identity})
        resource = next(
            (key for key, item in resources.items() if args[2] in (key, item["name"])), None
        )
        if args[1] == "delete":
            if switches.delete_error:
                raise DeploymentError("cleanup denied")
            if resource:
                del resources[resource]
            return ""
        return json.dumps(resources[resource]) if resource else None

    machine = SimpleNamespace(**{
        name + (f"_X_{count}" if count > 1 else ""): name + (f"_X_{count}" if count > 1 else "")
        for name in ("T4", "L4", "L40S", "A100_40GB", "A100_80GB", "H100", "H200", "B200")
        for count in ((1, 8) if name == "B200" else (1, 2, 4, 8))
    })
    sdk = SimpleNamespace(Machine=machine, Deployment=Remote)
    config = SimpleNamespace(ApiKeyAuth=Config, HttpHealthCheck=Config, AutoScaleConfig=Config)
    monkeypatch.setattr(LightningProvider, "_sdk", lambda self: (sdk, config))
    monkeypatch.setattr(LightningProvider, "_cli_sync", staticmethod(cli))
    return SimpleNamespace(resources=resources, calls=calls, switches=switches, sdk=sdk)


def recipe(model="fish-s2-pro", **kwargs):
    profile = ModelRegistry().get(model)
    request = DeploymentRequest(model=model, provider="lightning", **kwargs)
    return request, profile, get_runtime_template(profile.default_runtime).render(profile, request)


@pytest.mark.asyncio
async def test_deploy_shared_runtime_auth_metadata_and_stop(cloud, tmp_path):
    weave = make_weave(tmp_path / "state.db")
    dep = await weave.deploy("fish-s2-pro", provider="lightning", wait_for_ready=False)
    try:
        assert dep.state == DeploymentState.STARTING
        record = await weave.lifecycle_service.get_record(dep.id)
        assert record.lightning.name == dep.id and record.lightning.owned
        assert record.lightning.resource_id.startswith("dep_")
        assert record.lightning.teamspace == "owner/tests"
        serialized = record.model_dump_json()
        assert KEY not in serialized and "test-user" not in serialized
        start = next(call[1] for call in cloud.calls if call[0] == "start")
        _, _, runtime = recipe()
        assert start["image"] == runtime.docker_image
        assert start["machine"] == "L4"
        assert start["include_credentials"] is False
        assert start["spot"] is False
        assert start["entrypoint"] == "/bin/sh" and start["ports"] == [8080]
        assert start["health_check"].path == "/v1/health"
        assert "--revision" in start["command"]
        assert start["autoscale"].min_replicas == 0
        assert start["autoscale"].max_replicas == 1
        assert start["autoscale"].idle_threshold_seconds == "300"
        assert start["env"] == runtime.env_vars
        await dep.wait_for_ready()
        assert dep.state == DeploymentState.HEALTHY
        await dep.stop()
        await dep.stop()
        assert not cloud.resources
        assert (await weave.lifecycle_service.get_record(dep.id)).state == DeploymentState.STOPPED
        assert len([c for c in cloud.calls if c[0] == "cli" and c[1][1] == "delete"]) == 1
    finally:
        await weave.close()


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
async def test_status_never_claims_application_healthy(cloud, state, pending, failing, expected):
    provider = LightningProvider()
    dep = await provider.deploy(*recipe())
    resource = next(iter(cloud.resources.values()))
    resource["desired_state"] = state
    resource["status"].update(pending_replicas=pending, failing_replicas=failing, ready_replicas=0)
    assert (await provider.get_status(dep.id)).state == expected


@pytest.mark.asyncio
async def test_externally_deleted_is_stopped(cloud):
    provider = LightningProvider()
    dep = await provider.deploy(*recipe())
    cloud.resources.clear()
    assert (await provider.get_status(dep.id)).state == DeploymentState.STOPPED
    await provider.stop(dep.id)


@pytest.mark.asyncio
async def test_external_scaledown_still_allows_full_delete(cloud):
    provider = LightningProvider()
    dep = await provider.deploy(*recipe())
    resource = next(iter(cloud.resources.values()))
    resource["desired_state"] = "STOPPED"
    assert (await provider.get_status(dep.id)).state == DeploymentState.STOPPED
    assert cloud.resources
    await provider.stop(dep.id)
    assert not cloud.resources


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


@pytest.mark.parametrize("stage", ["start", "endpoint", "foreign_endpoint", "auth", "identity"])
@pytest.mark.asyncio
async def test_failure_cleanup_and_secret_safe_error(cloud, caplog, stage):
    if stage == "start":
        cloud.switches.start_error = RuntimeError(f"SDK body has {KEY}")
    elif stage == "endpoint":
        cloud.switches.urls = []
    elif stage == "foreign_endpoint":
        cloud.switches.urls = ["https://foreign.example"]
    elif stage == "auth":
        cloud.switches.start_error = SimpleSdkError(403, KEY)
    else:
        cloud.switches.identity = "scoped-api-key"
    provider = LightningProvider()
    with pytest.raises((DeploymentError, ProviderAuthError)) as caught:
        await provider.deploy(*recipe())
    assert not cloud.resources
    assert KEY not in str(caught.value) + repr(caught.value) + caplog.text
    assert caught.value.__suppress_context__


class SimpleSdkError(Exception):
    def __init__(self, status, secret):
        super().__init__(secret)
        self.status = status


@pytest.mark.asyncio
async def test_cleanup_error_preserves_original(cloud, caplog):
    cloud.switches.start_error = RuntimeError(KEY)
    cloud.switches.delete_error = True
    with pytest.raises(DeploymentError, match="deploy failed"):
        await LightningProvider().deploy(*recipe())
    assert "cleanup failed" in caplog.text and KEY not in caplog.text


@pytest.mark.asyncio
async def test_collision_preserves_user_resource(cloud):
    cloud.switches.collision = True
    with pytest.raises(DeploymentError, match="collision"):
        await LightningProvider().deploy(*recipe())
    assert not any(c[0] == "start" or c[0] == "cli" and c[1][1] == "delete" for c in cloud.calls)


@pytest.mark.asyncio
async def test_unowned_resource_cannot_be_deleted(cloud):
    provider = LightningProvider()
    record = DeploymentRecord(
        id="user-resource", model="fish-s2-pro", provider="lightning",
        lightning=LightningDeploymentMetadata(name="user", teamspace="owner/tests", owned=False),
    )
    await provider._save(record)
    with pytest.raises(DeploymentError, match="not owned"):
        await provider.stop(record.id)
    assert not any(c[0] == "cli" for c in cloud.calls)


@pytest.mark.parametrize("key", ["LIGHTNING_API_KEY", "LIGHTNING_USER_ID"])
@pytest.mark.asyncio
async def test_missing_credentials_fail_before_resources(cloud, monkeypatch, key):
    monkeypatch.delenv(key)
    with pytest.raises(ProviderAuthError, match="LIGHTNING_USER_ID"):
        await LightningProvider().deploy(*recipe())
    assert not cloud.resources and not cloud.calls


@pytest.mark.asyncio
async def test_org_and_teamspace_configuration(cloud, monkeypatch):
    monkeypatch.setenv("LIGHTNING_TEAMSPACE", "tests")
    monkeypatch.setenv("LIGHTNING_ORG", "org")
    provider = LightningProvider()
    dep = await provider.deploy(*recipe())
    assert (await provider._record(dep.id)).lightning.teamspace == "org/tests"


@pytest.mark.asyncio
async def test_explicit_teamspace_overrides_environment(cloud):
    provider = LightningProvider()
    dep = await provider.deploy(*recipe(custom_args={"lightning": {"teamspace": "other/space"}}))
    assert (await provider._record(dep.id)).lightning.teamspace == "other/space"


@pytest.mark.asyncio
async def test_platform_credentials_cannot_be_copied_into_runtime(cloud):
    with pytest.raises(ProviderAuthError, match="injected"):
        await LightningProvider().deploy(*recipe(env={"LIGHTNING_API_KEY": KEY}))
    assert not cloud.calls


@pytest.mark.parametrize("custom_args", [
    {"engine_args": {"LIGHTNING_API_KEY": KEY}},
    {"extra_cli_args": ["--token", KEY]},
    {"extra_env": {"OTHER_NAME": KEY}},
])
@pytest.mark.asyncio
async def test_platform_credentials_rejected_before_option_persistence(cloud, custom_args):
    provider = LightningProvider()
    with pytest.raises(ProviderAuthError, match="persisted options") as error:
        await provider.deploy(*recipe(custom_args=custom_args))
    assert KEY not in str(error.value)
    assert not provider._records and not cloud.calls


@pytest.mark.parametrize("credential_name", [
    "LIGHTNING_API_KEY", "LIGHTNING_USER_ID", "LIGHTNING_AUTH_TOKEN",
])
@pytest.mark.parametrize("location", ["cli", "engine", "options_env", "runtime_env", "setup"])
@pytest.mark.asyncio
async def test_embedded_credentials_rejected_before_persistence(
    cloud, monkeypatch, credential_name, location,
):
    monkeypatch.setenv("LIGHTNING_AUTH_TOKEN", "test-auth-token")
    credential = {"LIGHTNING_API_KEY": KEY, "LIGHTNING_USER_ID": "test-user",
                  "LIGHTNING_AUTH_TOKEN": "test-auth-token"}[credential_name]
    value = f"prefix-{credential}-suffix"
    custom_args = {
        "cli": {"extra_cli_args": [f"--token={value}"]},
        "engine": {"engine_args": {"token": f"Bearer {value}"}},
        "options_env": {"extra_env": {"OTHER_NAME": f"Bearer {value}"}},
    }.get(location, {})
    request, profile, runtime = recipe(custom_args=custom_args)
    if location == "runtime_env":
        runtime.env_vars["OTHER_NAME"] = f"Bearer {value}"
    elif location == "setup":
        runtime.setup_commands.append(f"echo {value}")
    provider = LightningProvider()
    with pytest.raises(ProviderAuthError, match="persisted options") as error:
        await provider.deploy(request, profile, runtime)
    assert credential not in str(error.value)
    assert not provider._records and not cloud.calls


@pytest.mark.asyncio
async def test_unrelated_runtime_secrets_are_allowed(cloud):
    provider = LightningProvider()
    dep = await provider.deploy(*recipe(env={"MODEL_TOKEN": "unrelated-model-secret"}))
    assert dep.id in provider._records
    assert next(c[1] for c in cloud.calls if c[0] == "start")["env"]["MODEL_TOKEN"] == "unrelated-model-secret"
    await provider.stop(dep.id)


@pytest.mark.parametrize("provider,url,applied", [
    ("lightning", ENDPOINT, True), ("modal", ENDPOINT, False),
    ("lightning", "https://evil.cloudspaces.litng.ai.attacker.test", False),
    ("lightning", "http://8080-dep-test-d.cloudspaces.litng.ai", False),
    ("lightning", "https://example.test", False),
])
def test_endpoint_auth_is_scoped_and_repr_safe(cloud, provider, url, applied):
    auth = LightningEndpointAuth()
    assert bool(auth.headers_for(provider, url)) == applied
    assert KEY not in repr(auth)
    if applied:
        assert auth.headers_for(provider, url) == {"Authorization": f"Bearer {KEY}"}


def test_typed_autoscale_is_independent_of_destroy_timer():
    request, _, _ = recipe(autostop_mins=1440, custom_args={
        "lightning": {"min_replicas": 0, "max_replicas": 2, "idle_threshold_seconds": 90},
    })
    assert request.options.autostop.idle_minutes == 1440
    assert request.options.provider.lightning.idle_threshold_seconds == 90
    with pytest.raises(ValueError, match="must not exceed"):
        LightningOptions(min_replicas=3, max_replicas=1)


@pytest.mark.asyncio
async def test_sync_sdk_does_not_block_loop(cloud):
    cloud.switches.delay = 0.15
    provider = LightningProvider()
    task = asyncio.create_task(provider.deploy(*recipe()))
    ticks = 0
    while not task.done():
        await asyncio.sleep(0.01)
        ticks += 1
    await task
    assert ticks >= 5


@pytest.mark.asyncio
async def test_cancellation_drains_creation_then_cleans(cloud):
    started = threading.Event()
    original = cloud.sdk.Deployment.start

    def slow_start(self, **kwargs):
        started.set()
        time.sleep(0.05)
        original(self, **kwargs)

    cloud.sdk.Deployment.start = slow_start
    task = asyncio.create_task(LightningProvider().deploy(*recipe()))
    await asyncio.to_thread(started.wait, 3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not cloud.resources


def wav_bytes():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\x01\x00" * 24)
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_restart_attach_find_health_inference_stop(cloud, tmp_path):
    from test_endpoint_auth import HeaderRecordingProbe

    requests = []

    def handler(request):
        requests.append(request)
        assert request.headers["Authorization"] == f"Bearer {KEY}"
        return httpx.Response(200, content=wav_bytes(), headers={"content-type": "audio/wav"})

    db = tmp_path / "state.db"
    first = make_weave(db, handler=handler)
    dep = await first.deploy("fish-s2-pro", provider="lightning")
    assert await dep.synthesize("first") == wav_bytes()
    record = await first.lifecycle_service.get_record(dep.id)
    record.last_activity_at = datetime.now(UTC) - timedelta(minutes=60)
    await first.lifecycle_service.repository.save(record)
    await first.close()
    probe = HeaderRecordingProbe()
    second = make_weave(db, handler=handler, probe=probe)
    try:
        attached = await second.attach(dep.id)
        assert attached is not dep
        assert attached.last_activity_at == record.last_activity_at
        assert await second.find(model=dep.model, provider="lightning") is attached
        assert (await attached.refresh()).state == DeploymentState.HEALTHY
        assert (await attached.check_health()).is_healthy
        await attached.wait_for_ready()
        assert await attached.synthesize("recovered") == wav_bytes()
        assert len(probe.headers_seen) == 3
        assert all(headers.get("Authorization") == f"Bearer {KEY}" for headers in probe.headers_seen)
        assert all(str(r.url).endswith("/v1/tts") for r in requests)
        assert not second.lifecycle_service.get_state(dep.id).in_flight_requests
        await attached.stop()
        assert not cloud.resources
        assert (await second.lifecycle_service.get_record(dep.id)).state == DeploymentState.STOPPED
    finally:
        await second.close()


@pytest.mark.asyncio
async def test_destroy_protects_active_request_and_restores_long_idle(cloud, tmp_path):
    weave = make_weave(tmp_path / "state.db")
    dep = await weave.deploy("fish-s2-pro", provider="lightning")
    try:
        lifecycle = weave.lifecycle_service
        state = lifecycle.get_state(dep.id)
        state.last_activity_at = datetime.now(UTC) - timedelta(minutes=120)
        lifecycle.begin_request(dep.id)
        assert not await lifecycle.check_and_autostop(dep.id)
        assert cloud.resources
        lifecycle.end_request(dep.id)
        state.last_activity_at = datetime.now(UTC) - timedelta(minutes=120)
        assert await lifecycle.check_and_autostop(dep.id)
        assert not cloud.resources
    finally:
        await weave.close()


@pytest.mark.asyncio
async def test_stop_failure_keeps_retryable_state(cloud):
    provider = LightningProvider()
    dep = await provider.deploy(*recipe())
    cloud.switches.delete_error = True
    with pytest.raises(DeploymentError):
        await provider.stop(dep.id)
    assert (await provider._record(dep.id)).state != DeploymentState.STOPPED
    assert cloud.resources
    cloud.switches.delete_error = False
    await provider.stop(dep.id)
    assert not cloud.resources


@pytest.mark.asyncio
async def test_cold_start_retry_keeps_auth_and_activity(cloud, tmp_path):
    attempts = []

    def handler(request):
        attempts.append(request)
        assert request.headers["Authorization"] == f"Bearer {KEY}"
        if len(attempts) == 1:
            return httpx.Response(503)
        return httpx.Response(200, content=wav_bytes())

    weave = make_weave(tmp_path / "state.db", handler=handler)
    try:
        dep = await weave.deploy("fish-s2-pro", provider="lightning")
        assert await dep.synthesize("retry") == wav_bytes()
        assert len(attempts) == 2
        assert not weave.lifecycle_service.get_state(dep.id).in_flight_requests
        await dep.stop()
    finally:
        await weave.close()


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
    sdk, config = LightningProvider()._sdk()
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


@pytest.mark.asyncio
async def test_readiness_failure_cleans_resources(cloud, tmp_path):
    from inferweave.adapters.healthcheck.mock_probe import MockHealthcheckProbeAdapter
    from inferweave.core.exceptions import HealthcheckTimeoutError

    weave = make_weave(tmp_path / "state.db", probe=MockHealthcheckProbeAdapter(default_healthy=False))
    try:
        with pytest.raises(HealthcheckTimeoutError):
            await weave.deploy("fish-s2-pro", provider="lightning")
        await assert_failed_deployment_cleaned(weave, cloud)
    finally:
        await weave.close()


@pytest.mark.asyncio
async def test_health_auth_failure_cleans_resources(cloud, tmp_path, monkeypatch):
    weave = make_weave(tmp_path / "state.db")

    async def fail(*args, **kwargs):
        raise ProviderAuthError("endpoint denied")

    monkeypatch.setattr(weave.healthcheck_service, "wait_for_ready", fail)
    try:
        with pytest.raises(ProviderAuthError):
            await weave.deploy("fish-s2-pro", provider="lightning")
        await assert_failed_deployment_cleaned(weave, cloud)
    finally:
        await weave.close()


async def assert_failed_deployment_cleaned(weave, cloud):
    assert not cloud.resources
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
    assert len([c for c in cloud.calls if c[0] == "cli" and c[1][1] == "delete"]) == 1


@pytest.mark.parametrize("stage", ["cancel", "registration", "wiring"])
@pytest.mark.asyncio
async def test_post_provision_failure_cleans_lifecycle(cloud, tmp_path, monkeypatch, stage):
    weave = make_weave(tmp_path / "state.db")

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
    try:
        with pytest.raises(asyncio.CancelledError if stage == "cancel" else RuntimeError):
            await weave.deploy("fish-s2-pro", provider="lightning")
        await assert_failed_deployment_cleaned(weave, cloud)
    finally:
        await weave.close()


@pytest.mark.asyncio
async def test_readiness_cleanup_failure_preserves_error_and_retry(cloud, tmp_path, monkeypatch):
    weave = make_weave(tmp_path / "state.db")
    original_error = ProviderAuthError("endpoint denied")

    async def fail(*args, **kwargs):
        raise original_error

    monkeypatch.setattr(weave.healthcheck_service, "wait_for_ready", fail)
    cloud.switches.delete_error = True
    try:
        with pytest.raises(ProviderAuthError) as caught:
            await weave.deploy("fish-s2-pro", provider="lightning")
        assert caught.value is original_error
        [deployment] = weave.list_deployments()
        assert cloud.resources
        assert not weave.lifecycle_service.get_state(deployment.id).is_stopped
        assert (await weave.lifecycle_service.get_record(deployment.id)).state != DeploymentState.STOPPED
        cloud.switches.delete_error = False
        await weave.stop(deployment.id)
        assert not cloud.resources
        assert weave.lifecycle_service.get_state(deployment.id).is_stopped
    finally:
        await weave.close()


@pytest.mark.asyncio
async def test_image_uses_existing_transport_and_worker(cloud, tmp_path):
    import base64

    png = b"\x89PNG\r\n\x1a\n" + b"image"

    def handler(request):
        assert request.headers["Authorization"] == f"Bearer {KEY}"
        assert str(request.url).endswith("/v1/images/generations")
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(png).decode()}]})

    weave = make_weave(tmp_path / "state.db", handler=handler)
    try:
        dep = await weave.deploy("black-forest-labs/FLUX.1-schnell", provider="lightning", gpu_type="L40S")
        assert await dep.render("fox") == [png]
        assert "/tmp/iw-runtime" in next(c[1] for c in cloud.calls if c[0] == "start")["command"]
        await dep.stop()
    finally:
        await weave.close()


def test_launch_arguments_stay_literal(tmp_path):
    from inferweave.runtimes.base import RuntimeSpec

    argument = "hello; echo bad; $(echo expanded)"
    runtime = RuntimeSpec(name="test", docker_image="python", run_command="",
                          run_args=[sys.executable, "-c", "import sys;print(sys.argv[1])", argument])
    import shlex

    command = shlex.split(runtime_command(runtime))
    # Execute the generated shell program only on POSIX; tokenization is portable.
    assert command[0] == "-c"
    assert shlex.quote(argument) in command[1]


@pytest.mark.asyncio
async def test_cli_errors_and_timeouts_never_leak_secrets(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=1, stdout=KEY, stderr=KEY,
    ))
    with pytest.raises(DeploymentError) as caught:
        await _blocking(LightningProvider._cli_sync, "deployment", "delete", "name")
    assert KEY not in str(caught.value)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: (
        (_ for _ in ()).throw(subprocess.TimeoutExpired(KEY, 1, output=KEY))
    ))
    with pytest.raises(DeploymentError) as caught:
        LightningProvider._cli_sync("deployment", "delete", "name")
    assert KEY not in str(caught.value)
