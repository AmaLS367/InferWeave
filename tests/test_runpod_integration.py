"""Live RunPod integration test (opt-in, spends real money).

Drives the public InferWeave SDK through a full RunPod lifecycle via SkyPilot:

    deploy -> endpoint -> application readiness -> HEALTHY -> persisted status read
           -> teardown -> verify non-running state

It never runs in normal CI (`pytest -m "not integration"`) and skips cleanly unless every
requirement below is met. Run it deliberately with:

    INFERWEAVE_RUNPOD_INTEGRATION=1 RUNPOD_API_KEY=... \
        uv run --extra runpod pytest -m integration tests/test_runpod_integration.py -s

Required:
    INFERWEAVE_RUNPOD_INTEGRATION=1   Explicit opt-in; without it the test is skipped.
    RUNPOD_API_KEY                    RunPod credentials. Alternatively an existing
                                      ~/.runpod/config.toml (written by `runpod config`).
    SkyPilot with RunPod support      `pip install 'inferweave[runpod]'`, on Linux/macOS/WSL2
                                      (SkyPilot cannot run on native Windows).

Optional (cheapest-reasonable defaults shown):
    INFERWEAVE_RUNPOD_GPU=L4                        GPU type to request.
    INFERWEAVE_RUNPOD_MODEL=facebook/opt-125m       Hugging Face model served with vLLM.
    INFERWEAVE_RUNPOD_READY_TIMEOUT_SECS=1500       Max wait for the model server to be ready.
    INFERWEAVE_RUNPOD_AUTOSTOP_MINS=15              Provider-side idle autodown safety net.
    INFERWEAVE_RUNPOD_ALLOW_SPOT=0                  Set to 1 to allow cheaper, preemptible capacity.

Cost safety: the cluster is torn down in a `finally` block (via a fresh SDK instance reading the
persisted record, then a SkyPilot name sweep as a backstop), and the deployment also carries a
provider-side autodown timer in case this process is killed. If cleanup itself fails the test
fails loudly and names the cluster so it can be removed manually (`sky down <name>`).
"""

import asyncio
import importlib.util
import os
import sys
import uuid
from pathlib import Path

import pytest

from inferweave import (
    AutostopAction,
    DeploymentState,
    HardwareRequirements,
    InferWeave,
    ModelProfile,
    WorkloadType,
)
from inferweave.models.profile import HealthcheckConfig

GPU = os.getenv("INFERWEAVE_RUNPOD_GPU", "L4")
HF_MODEL = os.getenv("INFERWEAVE_RUNPOD_MODEL", "facebook/opt-125m")
READY_TIMEOUT_SECS = int(os.getenv("INFERWEAVE_RUNPOD_READY_TIMEOUT_SECS", "1500"))
AUTOSTOP_MINS = int(os.getenv("INFERWEAVE_RUNPOD_AUTOSTOP_MINS", "15"))
ALLOW_SPOT = os.getenv("INFERWEAVE_RUNPOD_ALLOW_SPOT", "0") == "1"


def _skip_reason() -> str | None:
    """Returns why the live test cannot run here, or None when it is enabled and configured."""
    if os.getenv("INFERWEAVE_RUNPOD_INTEGRATION") != "1":
        return "live RunPod test is opt-in: set INFERWEAVE_RUNPOD_INTEGRATION=1 (spends real money)"
    if sys.platform == "win32":
        return "SkyPilot requires POSIX; run under Linux, macOS, or WSL2"
    if importlib.util.find_spec("sky") is None:
        return "SkyPilot is not installed: pip install 'inferweave[runpod]'"
    if not (os.getenv("RUNPOD_API_KEY") or (Path.home() / ".runpod" / "config.toml").is_file()):
        return "RunPod credentials missing: set RUNPOD_API_KEY or run `runpod config`"
    return None


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or ""),
]


def _sweep_clusters(token: str) -> list[str]:
    """Backstop: tears down any SkyPilot cluster whose name carries this run's unique token."""
    import sky

    cleaned: list[str] = []
    for cluster in sky.get(sky.status()):
        name = getattr(cluster, "name", None) or (
            cluster.get("name") if isinstance(cluster, dict) else None
        )
        if name and token in name:
            sky.get(sky.down(cluster_name=name))
            cleaned.append(name)
    return cleaned


async def _teardown(token: str, profile: ModelProfile) -> list[str]:
    """Stops everything this run created. Returns error strings (empty on clean teardown)."""
    errors: list[str] = []

    # Fresh SDK instance: proves teardown works from persisted records, and still works if
    # the deploying instance failed before returning a handle.
    fresh = InferWeave()
    fresh.register_model(profile)
    for record in await fresh.list_records():
        if token in record.id and record.state != DeploymentState.STOPPED:
            try:
                # RunPod pods cannot be "stopped" by SkyPilot, only torn down.
                await fresh.stop(record.id, action=AutostopAction.DOWN)
            except Exception as err:  # noqa: BLE001
                errors.append(f"SDK stop of '{record.id}' failed: {err}")

    try:
        swept = await asyncio.to_thread(_sweep_clusters, token)
        if swept:
            errors.append(f"SkyPilot sweep had to tear down leaked clusters: {swept}")
    except Exception as err:  # noqa: BLE001
        errors.append(f"SkyPilot cluster sweep failed (check `sky status` for '{token}'): {err}")
    return errors


@pytest.mark.asyncio
async def test_live_runpod_lifecycle(monkeypatch, tmp_path):
    """deploy -> endpoint -> readiness -> HEALTHY -> persisted read -> down -> non-running."""
    monkeypatch.setenv("INFERWEAVE_DEPLOYMENTS_PATH", str(tmp_path / "deployments.db"))

    token = uuid.uuid4().hex[:8]
    profile = ModelProfile(
        id=f"iwit-{token}",
        name="InferWeave RunPod integration probe",
        workload_type=WorkloadType.LLM,
        default_runtime="vllm",
        artifact_id=HF_MODEL,
        hardware=HardwareRequirements(min_vram_gb=8, recommended_gpus=[GPU]),
        healthcheck=HealthcheckConfig(
            path="/health",
            port=8000,
            initial_delay_seconds=30,
            timeout_seconds=READY_TIMEOUT_SECS,
            probe_interval_seconds=10,
            request_timeout_seconds=10,
        ),
    )
    weave = InferWeave()
    weave.register_model(profile)

    failure: BaseException | None = None
    try:
        # 1. Deploy without blocking on readiness so each lifecycle stage is observable.
        deployment = await weave.deploy(
            model=profile.id,
            provider="runpod",
            gpu_type=GPU,
            autostop_mins=AUTOSTOP_MINS,
            custom_args={"autodown": True, "allow_spot": ALLOW_SPOT},
            wait_for_ready=False,
        )
        assert deployment.provider == "runpod"
        # A launched cluster is infrastructure only: it must not already read as HEALTHY.
        assert deployment.state in {DeploymentState.PROVISIONING, DeploymentState.STARTING}

        # 2. Receive an endpoint (it can lag the launch; refresh until RunPod exposes it).
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 300
        while not deployment.endpoint_url and loop.time() < deadline:
            await asyncio.sleep(10)
            await deployment.refresh()
        assert deployment.endpoint_url, "RunPod never exposed an endpoint for the deployment"

        # 3. Wait for application readiness: only a real probe may produce HEALTHY.
        status = await deployment.wait_for_ready(timeout_seconds=READY_TIMEOUT_SECS)
        assert status.state == DeploymentState.HEALTHY
        assert deployment.is_healthy
        assert deployment.status.ready_at is not None
        assert (await deployment.check_health()).is_healthy

        # 4. Read the persisted deployment from a fresh SDK instance (cross-process view).
        reader = InferWeave()
        reader.register_model(profile)
        (persisted,) = [r for r in await reader.list_records() if r.id == deployment.id]
        assert persisted.provider == "runpod"
        assert persisted.state == DeploymentState.HEALTHY
        assert persisted.endpoint_url == deployment.endpoint_url
        assert persisted.ready_at is not None
        reread = await reader.get_status(deployment.id)
        assert reread.state == DeploymentState.HEALTHY
        assert reread.endpoint_url == deployment.endpoint_url

        # 5. Tear down from the fresh instance, then verify against the cloud, not just records.
        await reader.stop(deployment.id, action=AutostopAction.DOWN)
        stopped = await reader.get_status(deployment.id)
        assert stopped.state == DeploymentState.STOPPED
        # SDK status short-circuits on a STOPPED record, so also ask the provider what
        # SkyPilot reports: a torn-down cluster is no longer listed and maps to STOPPED.
        (record,) = [r for r in await reader.list_records() if r.id == deployment.id]
        owner = reader.accounts.resolve(record.provider, record.account, record.id)
        live = await reader.router.get("runpod").status(record, owner)
        assert live.state == DeploymentState.STOPPED
    except BaseException as err:
        failure = err
        raise
    finally:
        cleanup_errors = await _teardown(token, profile)
        await weave.lifecycle_service.close()  # cancel the idle watchdog of this process
        if cleanup_errors and failure is None:
            pytest.fail("RunPod cleanup problems: " + "; ".join(cleanup_errors))
        for message in cleanup_errors:
            print(f"[cleanup] {message}", file=sys.stderr)
