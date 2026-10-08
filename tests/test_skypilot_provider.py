"""Unit tests for SkyPilotProvider: task spec, account isolation and lifecycle operations.

The provider is stateless and talks to a SkyPilot worker through a ``WorkerRunner``; these tests
inject a fake runner and assert on the operation name, the JSON payload, the worker environment
and the secrets handed to it. No SkyPilot, subprocess or network is involved.
"""

import copy
import hashlib
import json
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from inferweave.accounts import (
    AccountManager,
    AccountsConfig,
    ProviderAccount,
    ProviderAccounts,
    runpod_account,
    vast_account,
)
from inferweave.adapters.lifecycle.memory_repository import InMemoryDeploymentRepository
from inferweave.core.exceptions import ProviderOperationError, ProviderPlatformError
from inferweave.core.failures import FailureKind
from inferweave.domain.deployment_record import DeploymentRecord
from inferweave.domain.lifecycle import AutostopAction, AutostopPolicy
from inferweave.domain.options import DeploymentOptions
from inferweave.isolation import WorkerRunner
from inferweave.isolation import skypilot_worker as worker
from inferweave.models.deployment import DeploymentRequest
from inferweave.models.enums import DeploymentState, ProviderType, WorkloadType
from inferweave.models.profile import (
    HardwareRequirements,
    HealthcheckConfig,
    ModelProfile,
)
from inferweave.providers.skypilot import SkyPilotProvider, resolve_worker_workdir
from inferweave.runtimes.base import RuntimeSpec
from inferweave.services.provisioning_service import (
    ProvisionCandidate,
    ProvisioningService,
)

ENDPOINT = "http://1.2.3.4:8000"
AMBIENT_SENTINEL = "SENTINEL-ambient-runpod-key"


@dataclass
class Call:
    op: str
    payload: dict[str, Any]
    env: dict[str, str]
    secrets: dict[str, str]
    timeout: float
    deployment_id: str | None
    account_id: str


class FakeRunner:
    """Records every worker call and returns canned results (or raises canned errors)."""

    def __init__(self, **responses: Any) -> None:
        self.calls: list[Call] = []
        self.responses: dict[str, Any] = {
            "launch": {"endpoint": ENDPOINT},
            "status": {"exists": True, "status": "UP", "endpoint": ENDPOINT},
            "stop": {"existed": True},
            "down": {"existed": True},
        }
        self.responses.update(responses)

    async def run(
        self,
        operation: str,
        payload: Any,
        *,
        env: Any,
        secrets: Any = None,
        timeout: float,
        deployment_id: str | None = None,
        account_id: str = "",
    ) -> Any:
        self.calls.append(
            Call(
                operation,
                copy.deepcopy(dict(payload)),
                dict(env),
                dict(secrets or {}),
                timeout,
                deployment_id,
                account_id,
            )
        )
        response = self.responses[operation]
        if isinstance(response, BaseException):
            raise response
        return copy.deepcopy(response)


@pytest.fixture(autouse=True)
def clean_platform_env(monkeypatch):
    """No ambient platform credentials unless a test sets them explicitly."""
    for name in ("RUNPOD_API_KEY", "VAST_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def linux_host(monkeypatch):
    """Neutralizes the native-Windows guard so provider operations can run in tests."""
    monkeypatch.setattr(SkyPilotProvider, "_ensure_supported_platform", lambda self: None)


@pytest.fixture
def sample_profile() -> ModelProfile:
    return ModelProfile(
        id="meta-llama/Meta-Llama-3-8B-Instruct",
        name="Llama 3 8B Instruct",
        workload_type=WorkloadType.LLM,
        default_runtime="vllm",
        hardware=HardwareRequirements(
            min_vram_gb=16,
            gpu_count=1,
            recommended_gpus=["A10G"],
        ),
        healthcheck=HealthcheckConfig(port=8000),
    )


@pytest.fixture
def sample_runtime() -> RuntimeSpec:
    return RuntimeSpec(
        name="vllm",
        docker_image="vllm/vllm-openai:latest",
        setup_commands=["echo 'setup'"],
        run_command="python3 -m vllm.entrypoints.openai.api_server",
        port=8000,
        env_vars={"MODEL": "meta-llama/Meta-Llama-3-8B-Instruct"},
    )


def make_provider(
    tmp_path: Path, cloud: str = "runpod", runner: FakeRunner | None = None
) -> SkyPilotProvider:
    return SkyPilotProvider(cloud, state_dir=tmp_path, runner=runner or FakeRunner())  # type: ignore[arg-type]


def make_record(
    provider: SkyPilotProvider,
    request: DeploymentRequest,
    account: ProviderAccount,
    deployment_id: str = "iw-test-cluster",
) -> DeploymentRecord:
    return DeploymentRecord(
        id=deployment_id,
        model=request.model,
        provider=provider.name,
        account=account.id,
        resource=provider.resource_ref(deployment_id, request, account),
        options=request.options or DeploymentOptions(),
    )


def pooled_manager(tmp_path: Path, cloud: str = "runpod", ids: tuple[str, ...] = ("a", "b")):
    make = runpod_account if cloud == "runpod" else vast_account
    env_names = {i: f"IW_TEST_{cloud.upper()}_{i.upper()}" for i in ids}
    config = AccountsConfig(
        {cloud: ProviderAccounts(accounts=[make(i, api_key_env=env_names[i]) for i in ids])},
        state_dir=tmp_path,
    )
    environ = {env_names[i]: f"SENTINEL-{cloud}-{i}-secret" for i in ids}
    return AccountManager(config, environ=environ)


def ambient(cloud: str = "runpod") -> ProviderAccount:
    return ProviderAccount.ambient(cloud)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["capacity", "auth"])
async def test_settled_worker_launch_failure_reconciles_and_fails_over(
    tmp_path, sample_profile, sample_runtime, linux_host, kind
):
    from test_skypilot_worker import (
        FakeSky,
        InvalidCloudCredentials,
        ResourcesUnavailableError,
    )

    class FailureRunner(FakeRunner):
        exists = kind == "capacity"

        async def run(self, operation, payload, **kwargs):
            if operation == "status":
                self.responses["status"] = {"exists": self.exists, "status": "UP", "endpoint": ENDPOINT}
            result = await super().run(operation, payload, **kwargs)
            if operation == "down":
                self.exists = False
            if operation == "launch":
                sky = FakeSky()
                if kwargs["account_id"] == "a":
                    error_class = ResourcesUnavailableError if kind == "capacity" else InvalidCloudCredentials
                    sky.results["launch"] = error_class("SENTINEL-failure-secret")
                try:
                    result = worker.op_launch(sky, payload)
                except worker.WorkerError as error:
                    envelope = {"iw_worker": 1, "ok": False, "error": {
                        "kind": error.kind, "message": str(error),
                        "resource_may_exist": error.resource_may_exist,
                        "operation_may_continue": error.operation_may_continue,
                    }}
                    return WorkerRunner._parse(
                        json.dumps(envelope).encode(), 1, "runpod launch", kwargs["deployment_id"],
                    )
            return result

    manager = pooled_manager(tmp_path)
    runner = FailureRunner()
    provider = make_provider(tmp_path, runner=runner)
    repository = InMemoryDeploymentRepository()
    request = DeploymentRequest(model=sample_profile.id, provider="runpod")
    candidate = ProvisionCandidate(provider=provider, request=request, runtime=sample_runtime)
    record, _ = await ProvisioningService(manager, repository).provision([candidate], sample_profile)
    assert record.account == "b"
    assert [(c.op, c.account_id) for c in runner.calls] == (
        [("launch", "a"), ("status", "a"), ("down", "a"), ("status", "a"), ("launch", "b")]
        if kind == "capacity" else [("launch", "a"), ("launch", "b")]
    )
    [failed] = [r for r in await repository.list_all() if r.state == DeploymentState.FAILED]
    assert failed.account == "a"
    assert not failed.creation_may_continue and not failed.needs_reconciliation
    health = {h.account_id: h for h in manager.health()}
    assert health["a"].state == ("available" if kind == "capacity" else "revoked")
    assert all(h.in_flight == 0 for h in health.values())


# --- task spec (launch_payload) -----------------------------------------------------------


def test_launch_payload_maps_request_runtime_and_resources(tmp_path, sample_profile, sample_runtime):
    provider = make_provider(tmp_path)
    request = DeploymentRequest(
        model=sample_profile.id,
        provider="runpod",
        gpu_type="A100",
        num_gpus=2,
        autostop_mins=20,
    )
    record = make_record(provider, request, ambient())

    payload = provider.launch_payload(record, request, sample_profile, sample_runtime)

    assert payload["cluster"] == "iw-test-cluster"
    assert payload["setup"] == "echo 'setup'"
    assert payload["run"] == shlex.join(sample_runtime.run_args)
    assert payload["envs"] == {"MODEL": "meta-llama/Meta-Llama-3-8B-Instruct"}
    assert payload["port"] == 8000
    assert payload["idle_minutes"] == 20
    assert payload["down"] is False
    assert payload["workdir"] is None  # run script does not use the inferweave package
    assert payload["resources"] == {
        "cloud": "runpod",
        "accelerators": "A100:2",
        "ports": [8000],
        "use_spot": True,
        "image_id": "docker:vllm/vllm-openai:latest",
    }
    json.dumps(payload)  # the payload is sent to the worker as JSON


def test_launch_payload_defaults_to_recommended_gpu_then_a10g(tmp_path, sample_profile, sample_runtime):
    provider = make_provider(tmp_path)
    request = DeploymentRequest(model=sample_profile.id, provider="runpod")
    record = make_record(provider, request, ambient())

    payload = provider.launch_payload(record, request, sample_profile, sample_runtime)
    assert payload["resources"]["accelerators"] == "A10G:1"

    bare = sample_profile.model_copy(deep=True)
    bare.hardware.recommended_gpus = []
    payload = provider.launch_payload(record, request, bare, sample_runtime)
    assert payload["resources"]["accelerators"] == "A10G:1"


def test_launch_payload_setup_is_none_or_newline_joined(tmp_path, sample_profile, sample_runtime):
    provider = make_provider(tmp_path)
    request = DeploymentRequest(model=sample_profile.id, provider="runpod")
    record = make_record(provider, request, ambient())

    none_runtime = sample_runtime.model_copy(update={"setup_commands": []})
    assert provider.launch_payload(record, request, sample_profile, none_runtime)["setup"] is None

    multi = sample_runtime.model_copy(update={"setup_commands": ["pip install a", "pip install b"]})
    setup = provider.launch_payload(record, request, sample_profile, multi)["setup"]
    assert setup == "pip install a\npip install b"


def test_launch_payload_resource_options(tmp_path, sample_profile, sample_runtime):
    provider = make_provider(tmp_path)
    request = DeploymentRequest(
        model=sample_profile.id,
        provider="runpod",
        custom_args={
            "allow_spot": False,
            "disk_size_gb": 120,
            "preferred_regions": ["us-central-1", "eu-west-1"],
            "zone": "us-central-1a",
            "provider_args": {"image_id": "docker:custom/image:1"},
        },
    )
    record = make_record(provider, request, ambient())

    resources = provider.launch_payload(record, request, sample_profile, sample_runtime)["resources"]

    assert resources["use_spot"] is False
    assert resources["disk_size"] == 120
    assert resources["region"] == "us-central-1"
    assert resources["zone"] == "us-central-1a"
    # An explicit image_id overrides the runtime's docker image.
    assert resources["image_id"] == "docker:custom/image:1"


def test_launch_payload_skips_docker_image_id_without_runtime_image(
    tmp_path, sample_profile, sample_runtime
):
    provider = make_provider(tmp_path)
    request = DeploymentRequest(model=sample_profile.id, provider="runpod")
    record = make_record(provider, request, ambient())
    runtime = sample_runtime.model_copy(update={"docker_image": ""})

    resources = provider.launch_payload(record, request, sample_profile, runtime)["resources"]
    assert "image_id" not in resources


def test_launch_payload_explicit_workdir_gets_pythonpath(tmp_path, sample_profile, sample_runtime):
    provider = make_provider(tmp_path)
    request = DeploymentRequest(
        model=sample_profile.id,
        provider="runpod",
        custom_args={"provider_args": {"workdir": "/path/to/workdir"}},
    )
    record = make_record(provider, request, ambient())

    payload = provider.launch_payload(record, request, sample_profile, sample_runtime)

    assert payload["workdir"] == "/path/to/workdir"
    assert payload["envs"]["PYTHONPATH"] == ".:$PYTHONPATH"
    assert payload["resources"]["image_id"] == "docker:vllm/vllm-openai:latest"


def test_launch_payload_keeps_user_pythonpath(tmp_path, sample_profile, sample_runtime):
    provider = make_provider(tmp_path)
    request = DeploymentRequest(
        model=sample_profile.id,
        provider="runpod",
        custom_args={"provider_args": {"workdir": "/w"}},
    )
    record = make_record(provider, request, ambient())
    runtime = sample_runtime.model_copy(update={"env_vars": {"PYTHONPATH": "/custom"}})

    envs = provider.launch_payload(record, request, sample_profile, runtime)["envs"]
    assert envs["PYTHONPATH"] == "/custom"


def test_launch_payload_auto_workdir_for_inferweave_workers(tmp_path, sample_profile):
    flux_runtime = RuntimeSpec(
        name="flux-diffusers",
        docker_image="pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime",
        run_command="python3 -m inferweave.workers.flux --model black-forest-labs/FLUX.1-schnell --port 8000",
        port=8000,
        env_vars={"MODEL": "black-forest-labs/FLUX.1-schnell"},
    )
    provider = make_provider(tmp_path)
    request = DeploymentRequest(model="black-forest-labs/FLUX.1-schnell", provider="runpod")
    record = make_record(provider, request, ambient())

    payload = provider.launch_payload(record, request, sample_profile, flux_runtime)

    # The workdir is auto-resolved to a local directory that contains the inferweave package.
    assert payload["workdir"] is not None
    assert (Path(payload["workdir"]) / "inferweave").is_dir()
    assert ".:$PYTHONPATH" in payload["envs"]["PYTHONPATH"]


@pytest.mark.parametrize(
    ("custom_args", "idle_minutes", "down"),
    [
        ({}, 25, False),
        ({"autodown": True}, 25, True),
    ],
)
def test_launch_payload_autostop_mapping_from_request(
    tmp_path, sample_profile, sample_runtime, custom_args, idle_minutes, down
):
    provider = make_provider(tmp_path)
    request = DeploymentRequest(
        model=sample_profile.id, provider="runpod", autostop_mins=25, custom_args=custom_args
    )
    record = make_record(provider, request, ambient())

    payload = provider.launch_payload(record, request, sample_profile, sample_runtime)

    assert payload["idle_minutes"] == idle_minutes
    assert payload["down"] is down


def test_launch_payload_autostop_action_down_uses_policy_idle_minutes(
    tmp_path, sample_profile, sample_runtime
):
    provider = make_provider(tmp_path)
    request = DeploymentRequest(
        model=sample_profile.id,
        provider="runpod",
        options=DeploymentOptions(
            autostop=AutostopPolicy(action=AutostopAction.DOWN, idle_minutes=15, enabled=True)
        ),
    )
    record = make_record(provider, request, ambient())

    payload = provider.launch_payload(record, request, sample_profile, sample_runtime)

    assert payload["down"] is True
    assert payload["idle_minutes"] == 15


def test_launch_payload_autostop_disabled_clears_idle_minutes(tmp_path, sample_profile, sample_runtime):
    provider = make_provider(tmp_path)
    request = DeploymentRequest(
        model=sample_profile.id,
        provider="runpod",
        autostop_mins=20,
        options=DeploymentOptions(autostop=AutostopPolicy(enabled=False, idle_minutes=None)),
    )
    record = make_record(provider, request, ambient())

    payload = provider.launch_payload(record, request, sample_profile, sample_runtime)

    assert payload["idle_minutes"] is None
    assert payload["down"] is False


# --- workdir staging ----------------------------------------------------------------------


def test_resolve_worker_workdir_from_source_tree():
    workdir = resolve_worker_workdir()
    assert workdir is not None
    p = Path(workdir)
    assert p.name == "src"
    assert (p / "inferweave").is_dir()


def test_resolve_worker_workdir_from_site_packages_never_syncs_site_packages(tmp_path):
    # Simulate installed environment in site-packages
    site_packages = tmp_path / ".venv" / "lib" / "python3.12" / "site-packages"
    pkg_dir = site_packages / "inferweave"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text("# inferweave init", encoding="utf-8")
    (pkg_dir / "workers").mkdir()
    (pkg_dir / "workers" / "flux.py").write_text("# flux worker", encoding="utf-8")

    # Add huge dummy packages alongside inferweave
    torch_dir = site_packages / "torch"
    torch_dir.mkdir()
    (torch_dir / "libtorch.so").write_bytes(b"huge_binary")

    staging_base = tmp_path / "custom_staging"

    staged_workdir = resolve_worker_workdir(pkg_path=pkg_dir, staging_base=staging_base)

    assert staged_workdir is not None
    # 1. CRITICAL: Never return the entire site-packages!
    assert staged_workdir != str(site_packages)
    assert Path(staged_workdir).name != "site-packages"

    # 2. Returned workdir must be the staged worker_pkg directory
    staged_path = Path(staged_workdir)
    assert staged_path == staging_base / "worker_pkg"

    # 3. Only inferweave was staged, NOT torch or other dependencies
    assert (staged_path / "inferweave").is_dir()
    assert (staged_path / "inferweave" / "__init__.py").exists()
    assert (staged_path / "inferweave" / "workers" / "flux.py").exists()
    assert not (staged_path / "torch").exists()


# --- identity, dry run, platform ----------------------------------------------------------


def test_identity_and_dry_run_helpers(tmp_path, sample_profile, sample_runtime):
    provider = make_provider(tmp_path, "aws")
    request = DeploymentRequest(model=sample_profile.id, provider="aws", dry_run=True)

    assert provider.name == "aws"
    assert provider.provider_type == ProviderType.SKYPILOT
    deployment_id = provider.new_deployment_id(sample_profile)
    assert deployment_id.startswith("iw-meta-llama-meta-llama-3-8b-instruct-")
    assert provider.new_deployment_id(sample_profile) != deployment_id
    assert provider.resource_ref(deployment_id, request, ambient("aws")).name == deployment_id
    assert "dryrun" in provider.dry_run_endpoint(deployment_id, sample_runtime)
    # A dry run never needs SkyPilot, so it works on any OS (the Windows guard is not consulted).
    provider.preflight(request, sample_profile, sample_runtime, ambient("aws"))


def test_windows_guard_blocks_live_operations(monkeypatch, tmp_path, sample_profile, sample_runtime):
    provider = make_provider(tmp_path, "aws")
    request = DeploymentRequest(model=sample_profile.id, provider="aws")

    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(ProviderPlatformError, match="WSL2"):
        provider.preflight(request, sample_profile, sample_runtime, ambient("aws"))

    monkeypatch.setattr(sys, "platform", "linux")
    provider.preflight(request, sample_profile, sample_runtime, ambient("aws"))


@pytest.mark.asyncio
async def test_operations_refuse_native_windows(monkeypatch, tmp_path):
    runner = FakeRunner()
    provider = make_provider(tmp_path, "aws", runner)
    record = DeploymentRecord(id="iw-x", model="m", provider="aws")

    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(ProviderPlatformError):
        await provider.status(record, ambient("aws"))
    assert runner.calls == []


def test_preflight_accepts_complete_pooled_account(
    linux_host, tmp_path, sample_profile, sample_runtime
):
    provider = make_provider(tmp_path)
    manager = pooled_manager(tmp_path)
    request = DeploymentRequest(model=sample_profile.id, provider="runpod")
    provider.preflight(request, sample_profile, sample_runtime, manager.resolve("runpod", "a"))


# --- provision / endpoint resolution --------------------------------------------------------


@pytest.mark.asyncio
async def test_provision_sends_launch_and_reports_starting_with_endpoint(
    linux_host, tmp_path, sample_profile, sample_runtime
):
    runner = FakeRunner()
    provider = make_provider(tmp_path, runner=runner)
    request = DeploymentRequest(model=sample_profile.id, provider="runpod", gpu_type="A100")
    record = make_record(provider, request, ambient())

    result = await provider.provision(record, request, sample_profile, sample_runtime, ambient())

    (call,) = runner.calls
    assert call.op == "launch"
    assert call.timeout == 3600
    assert call.deployment_id == record.id
    assert call.account_id == "ambient"
    assert call.payload["cloud"] == "runpod"
    assert call.payload["cluster"] == record.resource.name
    assert call.payload["resources"]["accelerators"] == "A100:1"
    # A launched cluster is infrastructure only; HEALTHY needs a readiness probe.
    assert result.state == DeploymentState.STARTING
    assert result.endpoint_url == ENDPOINT


@pytest.mark.asyncio
@pytest.mark.parametrize("launch_result", [{"endpoint": None}, {}, None])
async def test_provision_without_endpoint_stays_provisioning(
    linux_host, tmp_path, sample_profile, sample_runtime, launch_result
):
    provider = make_provider(tmp_path, runner=FakeRunner(launch=launch_result))
    request = DeploymentRequest(model=sample_profile.id, provider="runpod")
    record = make_record(provider, request, ambient())

    result = await provider.provision(record, request, sample_profile, sample_runtime, ambient())

    assert result.state == DeploymentState.PROVISIONING
    assert result.endpoint_url is None


@pytest.mark.asyncio
async def test_provision_surfaces_launch_failure(
    linux_host, tmp_path, sample_profile, sample_runtime
):
    """A failed launch is a classified error from the worker; it must not be swallowed."""
    failure = ProviderOperationError("no capacity", FailureKind.CAPACITY, resource_may_exist=True)
    provider = make_provider(tmp_path, runner=FakeRunner(launch=failure))
    request = DeploymentRequest(model=sample_profile.id, provider="runpod", gpu_type="A100")
    record = make_record(provider, request, ambient())

    with pytest.raises(ProviderOperationError, match="no capacity") as excinfo:
        await provider.provision(record, request, sample_profile, sample_runtime, ambient())

    assert excinfo.value.kind is FailureKind.CAPACITY
    assert excinfo.value.resource_may_exist is True


# --- account isolation ------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("cloud", ["runpod", "vast"])
async def test_pooled_account_runs_in_private_home_with_secret_only_in_secrets(
    linux_host, monkeypatch, tmp_path, sample_profile, sample_runtime, cloud
):
    monkeypatch.setenv("RUNPOD_API_KEY", AMBIENT_SENTINEL)
    monkeypatch.setenv("VAST_API_KEY", AMBIENT_SENTINEL)
    manager = pooled_manager(tmp_path, cloud)
    account = manager.resolve(cloud, "a")
    runner = FakeRunner()
    provider = make_provider(tmp_path, cloud, runner)
    request = DeploymentRequest(model=sample_profile.id, provider=cloud, gpu_type="A100")
    record = make_record(provider, request, account)

    await provider.provision(record, request, sample_profile, sample_runtime, account)

    (call,) = runner.calls
    digest = hashlib.sha256(f"a/{account.owner_fingerprint}".encode()).hexdigest()[:16]
    home = tmp_path / "skypilot" / cloud / digest / "home"
    assert call.env["HOME"] == str(home)
    assert call.env["USERPROFILE"] == str(home)
    assert home.is_dir()
    # No platform credential is inherited from the parent process.
    assert "RUNPOD_API_KEY" not in call.env
    assert "VAST_API_KEY" not in call.env
    assert not any(AMBIENT_SENTINEL in value for value in call.env.values())

    secret = f"SENTINEL-{cloud}-a-secret"
    assert call.secrets == {"api_key": secret}
    assert secret not in json.dumps(call.payload)
    assert not any(secret in value for value in call.env.values())

    assert call.account_id == "a"
    assert call.payload["mode"] == "account"
    assert call.payload["cloud"] == cloud
    assert call.payload["server_port"] == provider.account_port(account)
    assert call.payload["fingerprint"] == account.secret_fingerprint
    assert call.payload["fingerprint"]
    assert secret not in call.payload["fingerprint"]


@pytest.mark.asyncio
async def test_each_pooled_account_gets_its_own_home_secret_and_port(
    linux_host, tmp_path, sample_profile, sample_runtime
):
    manager = pooled_manager(tmp_path)
    runner = FakeRunner()
    provider = make_provider(tmp_path, "runpod", runner)
    request = DeploymentRequest(model=sample_profile.id, provider="runpod")

    for account_id in ("a", "b"):
        account = manager.resolve("runpod", account_id)
        record = make_record(provider, request, account, f"iw-{account_id}")
        await provider.status(record, account)

    first, second = runner.calls
    assert first.env["HOME"] != second.env["HOME"]
    assert first.secrets == {"api_key": "SENTINEL-runpod-a-secret"}
    assert second.secrets == {"api_key": "SENTINEL-runpod-b-secret"}
    assert first.payload["server_port"] != second.payload["server_port"]
    assert first.payload["fingerprint"] != second.payload["fingerprint"]


@pytest.mark.asyncio
async def test_ambient_account_uses_parent_environment_and_no_secrets(
    linux_host, monkeypatch, tmp_path, sample_profile, sample_runtime
):
    monkeypatch.setenv("RUNPOD_API_KEY", AMBIENT_SENTINEL)
    runner = FakeRunner()
    provider = make_provider(tmp_path, "runpod", runner)
    request = DeploymentRequest(model=sample_profile.id, provider="runpod")
    record = make_record(provider, request, ambient())

    await provider.provision(record, request, sample_profile, sample_runtime, ambient())

    (call,) = runner.calls
    assert call.payload["mode"] == "ambient"
    assert "server_port" not in call.payload
    assert "fingerprint" not in call.payload
    assert call.secrets == {}
    # The user's own SkyPilot setup (and environment) is used unchanged.
    assert call.env["RUNPOD_API_KEY"] == AMBIENT_SENTINEL
    assert call.env.get("HOME") == os.environ.get("HOME")
    assert not (tmp_path / "skypilot").exists()


@pytest.mark.asyncio
async def test_pooled_account_on_non_pooled_cloud_is_invalid_request(
    linux_host, tmp_path, sample_profile, sample_runtime
):
    pooled = pooled_manager(tmp_path).resolve("runpod", "a")
    runner = FakeRunner()
    provider = make_provider(tmp_path, "aws", runner)
    request = DeploymentRequest(model=sample_profile.id, provider="aws")
    record = make_record(provider, request, pooled)

    with pytest.raises(ProviderOperationError) as excinfo:
        await provider.provision(record, request, sample_profile, sample_runtime, pooled)

    assert excinfo.value.kind is FailureKind.INVALID_REQUEST
    assert excinfo.value.resource_may_exist is False
    assert "aws" in str(excinfo.value)
    assert runner.calls == []  # nothing reached a worker
    assert "SENTINEL" not in str(excinfo.value)


def test_account_port_is_stable_persisted_and_distinct(tmp_path):
    manager = pooled_manager(tmp_path, ids=("a", "b", "c"))
    provider = make_provider(tmp_path)
    accounts = [manager.resolve("runpod", i) for i in ("a", "b", "c")]

    ports = [provider.account_port(account) for account in accounts]

    assert len(set(ports)) == 3
    assert all(47000 <= port < 50000 for port in ports)
    assert ports == [provider.account_port(account) for account in accounts]  # stable

    registry = tmp_path / "skypilot" / "ports.json"
    assert json.loads(registry.read_text(encoding="utf-8")) == {
        f"runpod/a/{manager.resolve('runpod', 'a').owner_fingerprint}": ports[0],
        f"runpod/b/{manager.resolve('runpod', 'b').owner_fingerprint}": ports[1],
        f"runpod/c/{manager.resolve('runpod', 'c').owner_fingerprint}": ports[2],
    }
    # A fresh provider (new process) reads the persisted assignment.
    assert make_provider(tmp_path).account_port(accounts[1]) == ports[1]


def test_account_port_registry_is_shared_across_clouds(tmp_path):
    runpod_a = pooled_manager(tmp_path, "runpod", ("a",)).resolve("runpod", "a")
    vast_a = pooled_manager(tmp_path, "vast", ("a",)).resolve("vast", "a")

    runpod_port = make_provider(tmp_path, "runpod").account_port(runpod_a)
    vast_port = make_provider(tmp_path, "vast").account_port(vast_a)

    assert runpod_port != vast_port
    registry = json.loads((tmp_path / "skypilot" / "ports.json").read_text(encoding="utf-8"))
    assert {"/".join(k.split("/")[:2]) for k in registry} == {"runpod/a", "vast/a"}


def test_account_port_recovers_from_corrupt_registry(tmp_path):
    account = pooled_manager(tmp_path).resolve("runpod", "a")
    registry = tmp_path / "skypilot" / "ports.json"
    registry.parent.mkdir(parents=True)
    registry.write_text("{not json", encoding="utf-8")

    port = make_provider(tmp_path).account_port(account)

    assert json.loads(registry.read_text(encoding="utf-8")) == {f"runpod/a/{account.owner_fingerprint}": port}


# --- status / exists / stop ------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("worker_result", "expected"),
    [
        ({"exists": True, "status": "UP", "endpoint": ENDPOINT}, DeploymentState.STARTING),
        ({"exists": True, "status": "CLUSTERSTATUS.UP", "endpoint": ENDPOINT}, DeploymentState.STARTING),
        ({"exists": True, "status": "INIT", "endpoint": None}, DeploymentState.PROVISIONING),
        ({"exists": True, "status": "STOPPED", "endpoint": None}, DeploymentState.STOPPED),
        ({"exists": True, "status": "WEIRD", "endpoint": None}, DeploymentState.PENDING),
        ({"exists": False, "status": None, "endpoint": None}, DeploymentState.STOPPED),
        ({"exists": False}, DeploymentState.STOPPED),
    ],
)
async def test_status_maps_cluster_state_and_never_reports_healthy(
    linux_host, tmp_path, worker_result, expected
):
    runner = FakeRunner(status=worker_result)
    provider = make_provider(tmp_path, runner=runner)
    record = DeploymentRecord(
        id="iw-test-cluster",
        model="m",
        provider="runpod",
        resource=provider.resource_ref("iw-test-cluster", DeploymentRequest(model="m"), ambient()),
    )

    status = await provider.status(record, ambient())

    assert status.state == expected
    assert status.state != DeploymentState.HEALTHY
    (call,) = runner.calls
    assert call.op == "status"
    assert call.payload["cluster"] == "iw-test-cluster"
    assert call.timeout == 180


@pytest.mark.asyncio
async def test_status_endpoint_comes_from_worker_else_record(linux_host, tmp_path):
    provider = make_provider(tmp_path, runner=FakeRunner())
    record = DeploymentRecord(
        id="iw-x", model="m", provider="runpod", endpoint_url="http://old:8000"
    )

    assert (await provider.status(record, ambient())).endpoint_url == ENDPOINT

    provider = make_provider(
        tmp_path, runner=FakeRunner(status={"exists": True, "status": "UP", "endpoint": None})
    )
    assert (await provider.status(record, ambient())).endpoint_url == "http://old:8000"

    provider = make_provider(tmp_path, runner=FakeRunner(status={"exists": False}))
    stopped = await provider.status(record, ambient())
    assert stopped.state == DeploymentState.STOPPED
    assert stopped.endpoint_url == "http://old:8000"


@pytest.mark.asyncio
async def test_status_falls_back_to_record_id_without_resource(linux_host, tmp_path):
    runner = FakeRunner()
    provider = make_provider(tmp_path, runner=runner)

    await provider.status(DeploymentRecord(id="iw-legacy", model="m", provider="runpod"), ambient())

    assert runner.calls[0].payload["cluster"] == "iw-legacy"


@pytest.mark.asyncio
async def test_status_runs_under_the_owner_account(linux_host, tmp_path):
    manager = pooled_manager(tmp_path)
    runner = FakeRunner()
    provider = make_provider(tmp_path, runner=runner)
    owner = manager.resolve('runpod', 'b')
    record = DeploymentRecord(id="iw-x", model="m", provider="runpod", account="b")

    await provider.status(record, owner)

    (call,) = runner.calls
    assert call.account_id == "b"
    assert call.secrets == {"api_key": "SENTINEL-runpod-b-secret"}
    assert call.env["HOME"].endswith(str(Path("skypilot") / "runpod" / hashlib.sha256(f"b/{manager.resolve('runpod', 'b').owner_fingerprint}".encode()).hexdigest()[:16] / "home"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("worker_result", "expected"),
    [({"exists": True, "status": "STOPPED"}, True), ({"exists": False}, False)],
)
async def test_resource_exists_reflects_cluster_presence(
    linux_host, tmp_path, worker_result, expected
):
    provider = make_provider(tmp_path, runner=FakeRunner(status=worker_result))
    record = DeploymentRecord(id="iw-x", model="m", provider="runpod")

    assert await provider.resource_exists(record, ambient()) is expected


@pytest.mark.asyncio
async def test_status_error_is_not_swallowed(linux_host, tmp_path):
    error = ProviderOperationError("boom", FailureKind.TRANSIENT)
    provider = make_provider(tmp_path, runner=FakeRunner(status=error))
    record = DeploymentRecord(id="iw-x", model="m", provider="runpod")

    with pytest.raises(ProviderOperationError):
        await provider.status(record, ambient())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "operation"),
    [
        (AutostopAction.STOP, "stop"),
        (AutostopAction.DOWN, "down"),
        ("down", "down"),
        ("stop", "stop"),
    ],
)
async def test_stop_maps_action_to_worker_operation(linux_host, tmp_path, action, operation):
    runner = FakeRunner()
    provider = make_provider(tmp_path, runner=runner)
    record = DeploymentRecord(
        id="iw-test-cluster",
        model="m",
        provider="runpod",
        resource=provider.resource_ref("iw-test-cluster", DeploymentRequest(model="m"), ambient()),
    )

    await provider.stop(record, ambient(), action=action)

    (call,) = runner.calls
    assert call.op == operation
    assert call.payload["cluster"] == "iw-test-cluster"
    assert call.timeout == 900
    assert call.deployment_id == "iw-test-cluster"


@pytest.mark.asyncio
async def test_stop_defaults_to_stop(linux_host, tmp_path):
    runner = FakeRunner()
    provider = make_provider(tmp_path, runner=runner)

    await provider.stop(DeploymentRecord(id="iw-x", model="m", provider="runpod"), ambient())

    assert [call.op for call in runner.calls] == ["stop"]


@pytest.mark.asyncio
async def test_stop_error_is_not_swallowed(linux_host, tmp_path):
    error = ProviderOperationError("denied", FailureKind.PERMISSION)
    provider = make_provider(tmp_path, runner=FakeRunner(down=error))

    with pytest.raises(ProviderOperationError):
        await provider.stop(
            DeploymentRecord(id="iw-x", model="m", provider="runpod"),
            ambient(),
            action=AutostopAction.DOWN,
        )
