"""Deployment recovery across process restarts: InferWeave.attach() and InferWeave.find()."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from support import (
    AUDIO_MODEL,
    ENDPOINT,
    IMAGE_MODEL,
    deploy_on_modal,
    make_weave,
    modal_pool,
    patched_modal,
)

from inferweave.adapters.healthcheck.mock_probe import MockHealthcheckProbeAdapter
from inferweave.adapters.lifecycle.sqlite_repository import SqliteDeploymentRepository
from inferweave.core.exceptions import (
    AccountUnavailableError,
    AmbiguousDeploymentError,
    DeploymentNotActiveError,
    DeploymentNotFoundError,
    ModelNotFoundError,
)
from inferweave.models.enums import DeploymentState, WorkloadType


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "deployments.db"


async def _read_record(db_path: Path, deployment_id: str):
    return await SqliteDeploymentRepository(db_path).get(deployment_id)


@pytest.mark.asyncio
async def test_attach_after_restart_preserves_persisted_metadata(db_path: Path):
    # Process A deploys and is "killed".
    weave_a = make_weave(db_path)
    original = await deploy_on_modal(
        weave_a,
        destroy_after_idle_mins=1440,
        scaledown_window_seconds=600,
        custom_args={"cleanup_on_failure": True},
    )
    await original.wait_for_ready()
    persisted = await _read_record(db_path, original.id)
    assert persisted is not None
    assert persisted.state == DeploymentState.HEALTHY

    # Process B starts with an empty in-memory state.
    weave_b = make_weave(db_path)
    assert weave_b.list_deployments() == []
    attached = await weave_b.attach(original.id)

    assert attached is not original
    assert attached.id == original.id
    assert attached.model == AUDIO_MODEL
    assert attached.provider == "modal"
    assert attached.endpoint_url == ENDPOINT
    assert attached.state == DeploymentState.HEALTHY
    assert attached.workload_type == WorkloadType.AUDIO
    assert attached.status.created_at == persisted.created_at
    assert attached.status.ready_at == persisted.ready_at
    assert attached.autostop_mins == 1440

    # Attaching must not rewrite the persisted record.
    after = await _read_record(db_path, original.id)
    assert after is not None
    assert after.created_at == persisted.created_at
    assert after.ready_at == persisted.ready_at
    assert after.options == persisted.options
    assert after.options.autostop.idle_minutes == 1440
    assert after.options.provider.scaledown_window_seconds == 600
    assert after.options.cleanup_on_failure is True
    assert after.state == DeploymentState.HEALTHY
    assert weave_b.list_deployments() == [attached]


@pytest.mark.asyncio
async def test_attached_handle_refresh_stop_and_health_closures_work(db_path: Path):
    weave_a = make_weave(db_path)
    dep_a = await deploy_on_modal(weave_a)

    probe = MockHealthcheckProbeAdapter(default_healthy=True)
    weave_b = make_weave(db_path, probe=probe)
    attached = await weave_b.attach(dep_a.id)

    # check_health / wait_for_ready use the recovered closures and reach the probe
    result = await attached.check_health()
    assert result.is_healthy is True
    assert probe.probed_urls[-1] == f"{ENDPOINT}/v1/health"
    status = await attached.wait_for_ready()
    assert status.state == DeploymentState.HEALTHY
    assert (await _read_record(db_path, dep_a.id)).state == DeploymentState.HEALTHY

    # refresh() reconciles with the provider and the readiness probe
    with patched_modal() as mocks:
        refreshed = await attached.refresh()
    assert refreshed.state == DeploymentState.HEALTHY
    assert refreshed.model == AUDIO_MODEL
    assert refreshed.endpoint_url == ENDPOINT
    mocks.lookup_app.assert_awaited()

    # stop() really stops the provider app and persists STOPPED
    with patched_modal() as mocks:
        await attached.stop()
    mocks.stop_app.assert_awaited_once()
    assert mocks.stop_app.await_args.args[0].id == dep_a.id
    assert attached.state == DeploymentState.STOPPED
    stopped = await _read_record(db_path, dep_a.id)
    assert stopped is not None and stopped.state == DeploymentState.STOPPED


@pytest.mark.asyncio
async def test_attach_nonexistent_deployment_raises(db_path: Path):
    weave = make_weave(db_path)
    with pytest.raises(DeploymentNotFoundError):
        await weave.attach("iw-modal-does-not-exist")


@pytest.mark.asyncio
async def test_attach_stopped_deployment_is_rejected(db_path: Path):
    weave_a = make_weave(db_path)
    dep = await deploy_on_modal(weave_a)
    with patched_modal():
        await dep.stop()

    weave_b = make_weave(db_path)
    with pytest.raises(DeploymentNotActiveError, match="stopped"):
        await weave_b.attach(dep.id)
    assert await weave_b.find(model=AUDIO_MODEL, provider="modal") is None


@pytest.mark.asyncio
async def test_attach_failed_and_dry_run_deployments_are_rejected(db_path: Path):
    weave_a = make_weave(db_path)
    failed = await deploy_on_modal(weave_a)
    repo = SqliteDeploymentRepository(db_path)
    rec = await repo.get(failed.id)
    rec.mark_failed("readiness timed out")
    await repo.save(rec)

    dry = await weave_a.deploy(model=AUDIO_MODEL, provider="modal", dry_run=True)

    weave_b = make_weave(db_path)
    with pytest.raises(DeploymentNotActiveError, match="failed"):
        await weave_b.attach(failed.id)
    with pytest.raises(DeploymentNotActiveError, match="dry-run"):
        await weave_b.attach(dry.id)


@pytest.mark.asyncio
async def test_attach_unhealthy_scaled_to_zero_deployment_is_allowed(db_path: Path):
    weave_a = make_weave(db_path)
    dep = await deploy_on_modal(weave_a)
    repo = SqliteDeploymentRepository(db_path)
    rec = await repo.get(dep.id)
    rec.state = DeploymentState.UNHEALTHY  # cold app: probes time out until it wakes
    await repo.save(rec)

    attached = await make_weave(db_path).attach(dep.id)
    assert attached.state == DeploymentState.UNHEALTHY


@pytest.mark.asyncio
async def test_attach_with_unregistered_model_raises_model_not_found(db_path: Path):
    weave_a = make_weave(db_path)
    dep = await deploy_on_modal(weave_a)
    repo = SqliteDeploymentRepository(db_path)
    rec = await repo.get(dep.id)
    rec.model = "custom/unregistered-model"
    await repo.save(rec)

    with pytest.raises(ModelNotFoundError):
        await make_weave(db_path).attach(dep.id)


@pytest.mark.asyncio
async def test_find_by_model_and_provider_returns_attached_handle(db_path: Path):
    weave_a = make_weave(db_path)
    audio = await deploy_on_modal(weave_a, model=AUDIO_MODEL)
    await deploy_on_modal(weave_a, model=IMAGE_MODEL)

    weave_b = make_weave(db_path)
    found = await weave_b.find(model=AUDIO_MODEL, provider="modal")
    assert found is not None
    assert found.id == audio.id
    assert found.workload_type == WorkloadType.AUDIO
    assert await weave_b.find(model="fish-s2-pro", provider="runpod") is None
    assert await weave_b.find(model="meta-llama/Meta-Llama-3-8B-Instruct") is None
    # find() reuses the handle that attach() already tracks
    assert await weave_b.attach(audio.id) is found
    assert await weave_b.find(model=AUDIO_MODEL, provider="MODAL") is found


@pytest.mark.asyncio
async def test_find_with_multiple_matches_raises_ambiguity_error(db_path: Path):
    weave_a = make_weave(db_path)
    first = await deploy_on_modal(weave_a)
    second = await deploy_on_modal(weave_a)

    weave_b = make_weave(db_path)
    with pytest.raises(AmbiguousDeploymentError) as excinfo:
        await weave_b.find(model=AUDIO_MODEL, provider="modal")
    assert set(excinfo.value.candidate_ids) == {first.id, second.id}
    assert first.id in str(excinfo.value)

    # Explicit attach resolves the ambiguity; stopping one makes find() unambiguous.
    with patched_modal():
        await (await weave_b.attach(first.id)).stop()
    found = await weave_b.find(model=AUDIO_MODEL, provider="modal")
    assert found is not None and found.id == second.id


@pytest.mark.asyncio
async def test_attach_returns_tracked_handle_from_deploying_process(db_path: Path):
    weave = make_weave(db_path)
    deployed = await deploy_on_modal(weave)
    assert await weave.attach(deployed.id) is deployed
    assert await weave.find(model=AUDIO_MODEL, provider="modal") is deployed


@pytest.mark.asyncio
async def test_attach_after_local_stop_rejects_stale_tracked_handle(db_path: Path):
    weave = make_weave(db_path)
    deployed = await deploy_on_modal(weave)
    with patched_modal():
        await deployed.stop()
    with pytest.raises(DeploymentNotActiveError):
        await weave.attach(deployed.id)


@pytest.mark.asyncio
async def test_legacy_record_without_activity_gets_fresh_timer_and_backfill(
    db_path: Path,
):
    weave_a = make_weave(db_path)
    dep = await deploy_on_modal(weave_a)
    repo = SqliteDeploymentRepository(db_path)
    rec = await repo.get(dep.id)
    rec.last_activity_at = None  # record written by an older InferWeave version
    rec.workload_type = None
    await repo.save(rec)

    before = datetime.now(UTC)
    attached = await make_weave(db_path).attach(dep.id)
    assert attached.last_activity_at is not None
    assert attached.last_activity_at >= before
    assert attached.workload_type == WorkloadType.AUDIO  # taken from the model profile
    backfilled = await repo.get(dep.id)
    assert backfilled.last_activity_at == attached.last_activity_at


@pytest.mark.asyncio
async def test_attach_restores_idle_timer_and_destroy_policy(db_path: Path):
    weave_a = make_weave(db_path)
    dep = await deploy_on_modal(weave_a, destroy_after_idle_mins=60)
    long_ago = datetime.now(UTC) - timedelta(minutes=50)
    await weave_a.lifecycle_service.touch_activity(dep.id, now=long_ago)

    weave_b = make_weave(db_path)
    attached = await weave_b.attach(dep.id)
    assert attached.last_activity_at == long_ago
    assert attached.is_idle() is False
    scheduled = weave_b.lifecycle_service._watchdog.scheduled_checks  # type: ignore[attr-defined]
    assert dep.id in scheduled  # watchdog is re-armed for the recovered deployment

    # 50 minutes of idleness survived the restart; 15 more minutes crosses the 60 minute limit.
    with patched_modal() as mocks:
        assert await weave_b.lifecycle_service.check_and_autostop(
            dep.id, now=long_ago + timedelta(minutes=59)
        ) is False
        assert await weave_b.lifecycle_service.check_and_autostop(
            dep.id, now=long_ago + timedelta(minutes=61)
        ) is True
    mocks.stop_app.assert_awaited_once()
    assert mocks.stop_app.await_args.args[0].id == dep.id
    assert (await _read_record(db_path, dep.id)).state == DeploymentState.STOPPED


@pytest.mark.asyncio
async def test_attach_restores_the_owning_account_and_stops_under_it(db_path: Path):
    """A restarted process recovers the pooled account that owns the deployment."""
    pool = modal_pool("a", "b")
    weave_a = make_weave(db_path, accounts=pool)
    dep = await deploy_on_modal(weave_a, account="b")
    assert dep.account == "b"
    persisted = await _read_record(db_path, dep.id)
    assert persisted.account == "b" and persisted.resource.name == dep.id

    weave_b = make_weave(db_path, accounts=modal_pool("a", "b"))
    attached = await weave_b.attach(dep.id)
    assert attached.account == "b"
    assert (await weave_b.find(model=AUDIO_MODEL, provider="modal")) is attached

    with patched_modal() as mocks:
        await attached.stop()
    stopped_record, stopped_account = mocks.stop_app.await_args.args
    assert stopped_record.id == dep.id
    assert stopped_account.id == "b"  # never another account of the pool


@pytest.mark.asyncio
async def test_attach_fails_when_the_owning_account_is_no_longer_configured(db_path: Path):
    weave_a = make_weave(db_path, accounts=modal_pool("a", "b"))
    dep = await deploy_on_modal(weave_a, account="b")

    # account "b" was removed from the pool: never silently adopt account "a"
    weave_b = make_weave(db_path, accounts=modal_pool("a"))
    with pytest.raises(AccountUnavailableError, match="'b'"):
        await weave_b.attach(dep.id)
    assert weave_b.list_deployments() == []


def _custom_container_profile(
    model_id: str = "custom/vllm-container",
    *,
    port: int = 8088,
    health_path: str = "/readyz",
    args: list[str] | None = None,
    setup_commands: list[str] | None = None,
    env: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    metadata: dict | None = None,
):
    from inferweave.models.profile import HealthcheckConfig, ModelProfile

    hc = HealthcheckConfig(
        port=port,
        path=health_path,
        initial_delay_seconds=0.0,
        timeout_seconds=0.2,
        probe_interval_seconds=0.01,
        headers=headers or {},
    )
    return ModelProfile.for_container(
        id=model_id,
        name="Custom Container Service",
        image="vllm/vllm-openai:v0.6.0",
        args=args if args is not None else ["--model", model_id, "--port", str(port)],
        port=port,
        setup_commands=setup_commands or ["echo warmup"],
        env=env or {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
        healthcheck=hc,
        min_vram_gb=16.0,
        metadata=metadata or {"team": "inference", "tier": 1},
    )


@pytest.mark.asyncio
async def test_custom_container_deploy_persist_attach_refresh_healthcheck_stop(
    db_path: Path,
):
    """Custom container deploy -> persist -> new SDK instance -> attach -> refresh/healthcheck/stop."""
    weave_a = make_weave(db_path)
    custom_profile = _custom_container_profile()
    weave_a.register_model(custom_profile)

    dep_a = await deploy_on_modal(weave_a, model=custom_profile.id)
    persisted = await _read_record(db_path, dep_a.id)
    assert persisted is not None
    assert persisted.profile_snapshot is not None
    assert persisted.profile_snapshot.id == custom_profile.id
    assert persisted.profile_snapshot.container is not None
    assert persisted.profile_snapshot.container.image == "vllm/vllm-openai:v0.6.0"
    assert persisted.profile_snapshot.container.port == 8088
    assert persisted.profile_snapshot.healthcheck.path == "/readyz"

    # New SDK instance without registering the custom model
    probe = MockHealthcheckProbeAdapter(default_healthy=True)
    weave_b = make_weave(db_path, probe=probe)
    with pytest.raises(ModelNotFoundError):
        weave_b.registry.get(custom_profile.id)

    attached = await weave_b.attach(dep_a.id)
    assert attached.id == dep_a.id
    assert attached.model == custom_profile.id
    assert attached.workload_type == WorkloadType.CUSTOM
    assert attached.endpoint_url == ENDPOINT

    # Snapshot recovery must not mutate the global/instance registry
    with pytest.raises(ModelNotFoundError):
        weave_b.registry.get(custom_profile.id)

    # Healthcheck uses recovered custom port/path
    probe_res = await attached.check_health()
    assert probe_res.is_healthy is True
    assert probe.probed_urls[-1] == f"{ENDPOINT}/readyz"

    ready_status = await attached.wait_for_ready()
    assert ready_status.state == DeploymentState.HEALTHY

    # Refresh and get_status use recovered profile healthcheck and preserve snapshot
    with patched_modal() as mocks:
        refreshed = await attached.refresh()
        status_via_sdk = await weave_b.get_status(dep_a.id)
    assert refreshed.state == DeploymentState.HEALTHY
    assert status_via_sdk.state == DeploymentState.HEALTHY
    assert mocks.lookup_app.await_count == 2

    # Stop tears down remote resource and preserves snapshot on STOPPED record
    with patched_modal() as mocks:
        await attached.stop()
    mocks.stop_app.assert_awaited_once()
    stopped_rec = await _read_record(db_path, dep_a.id)
    assert stopped_rec is not None
    assert stopped_rec.state == DeploymentState.STOPPED
    assert stopped_rec.profile_snapshot is not None
    assert stopped_rec.profile_snapshot.id == custom_profile.id


@pytest.mark.asyncio
async def test_custom_container_recovery_via_find(db_path: Path):
    """Custom container deploy -> persist -> new SDK instance -> find() -> healthcheck/refresh/stop."""
    weave_a = make_weave(db_path)
    custom_profile = _custom_container_profile("custom/find-model", port=9000, health_path="/ping")
    weave_a.register_model(custom_profile)
    dep_a = await deploy_on_modal(weave_a, model=custom_profile.id)

    probe = MockHealthcheckProbeAdapter(default_healthy=True)
    weave_b = make_weave(db_path, probe=probe)
    found = await weave_b.find(model=custom_profile.id, provider="modal")
    assert found is not None
    assert found.id == dep_a.id
    assert found.model == custom_profile.id
    assert found.workload_type == WorkloadType.CUSTOM

    res = await found.check_health()
    assert res.is_healthy is True
    assert probe.probed_urls[-1] == f"{ENDPOINT}/ping"

    with patched_modal():
        await found.stop()
    assert found.state == DeploymentState.STOPPED


@pytest.mark.asyncio
async def test_custom_container_recovery_with_json_repository(tmp_path: Path):
    """Custom container deploy -> JSON persistence -> new SDK instance -> attach & find."""
    from inferweave import InferWeave
    from inferweave.accounts import AccountManager, AccountsConfig
    from inferweave.adapters.lifecycle.json_repository import JsonDeploymentRepository
    from inferweave.adapters.lifecycle.watchdog import MockWatchdogAdapter
    from inferweave.providers.router import ProviderRouter
    from inferweave.services.healthcheck_service import HealthcheckService
    from inferweave.services.lifecycle_service import LifecycleService

    json_path = tmp_path / "deployments.json"

    def make_json_weave(probe: MockHealthcheckProbeAdapter | None = None) -> InferWeave:
        probe_adapter = probe or MockHealthcheckProbeAdapter(default_healthy=True)
        hc = HealthcheckService(probe_port=probe_adapter)
        router = ProviderRouter()
        accounts = AccountManager(AccountsConfig())
        lifecycle = LifecycleService(
            watchdog_port=MockWatchdogAdapter(),
            repository=JsonDeploymentRepository(json_path),
            healthcheck_service=hc,
            provider_resolver=router.get,
            accounts=accounts,
        )
        return InferWeave(
            router=router,
            healthcheck_service=hc,
            lifecycle_service=lifecycle,
            accounts=accounts,
        )

    weave_a = make_json_weave()
    custom_profile = _custom_container_profile("custom/json-container", port=8123, health_path="/live")
    weave_a.register_model(custom_profile)
    dep_a = await deploy_on_modal(weave_a, model=custom_profile.id)

    probe_b = MockHealthcheckProbeAdapter(default_healthy=True)
    weave_b = make_json_weave(probe=probe_b)
    found = await weave_b.find(model=custom_profile.id, provider="modal")
    assert found is not None
    assert found.id == dep_a.id
    assert (await weave_b.attach(dep_a.id)) is found

    await found.check_health()
    assert probe_b.probed_urls[-1] == f"{ENDPOINT}/live"

    with patched_modal():
        await found.stop()
    rec = await JsonDeploymentRepository(json_path).get(dep_a.id)
    assert rec is not None and rec.state == DeploymentState.STOPPED
    assert rec.profile_snapshot is not None
    assert rec.profile_snapshot.id == custom_profile.id


@pytest.mark.asyncio
async def test_registered_profile_takes_precedence_over_snapshot(db_path: Path):
    """An explicitly registered ModelProfile in the new SDK instance takes precedence over snapshot."""
    weave_a = make_weave(db_path)
    original_profile = _custom_container_profile(
        "custom/precedence-model", port=8080, health_path="/v1/health"
    )
    weave_a.register_model(original_profile)
    dep = await deploy_on_modal(weave_a, model=original_profile.id)

    probe = MockHealthcheckProbeAdapter(default_healthy=True)
    weave_b = make_weave(db_path, probe=probe)
    updated_profile = _custom_container_profile(
        "custom/precedence-model", port=8080, health_path="/v2/override-health"
    )
    weave_b.register_model(updated_profile)

    attached = await weave_b.attach(dep.id)
    await attached.check_health()
    assert probe.probed_urls[-1] == f"{ENDPOINT}/v2/override-health"


@pytest.mark.asyncio
async def test_legacy_records_without_snapshot_preserve_behavior(db_path: Path):
    """Old records without profile_snapshot work for built-in models and raise ModelNotFoundError for custom models."""
    weave_a = make_weave(db_path)
    builtin_dep = await deploy_on_modal(weave_a, model=AUDIO_MODEL)

    custom_profile = _custom_container_profile("custom/legacy-no-snapshot")
    weave_a.register_model(custom_profile)
    custom_dep = await deploy_on_modal(weave_a, model=custom_profile.id)

    repo = SqliteDeploymentRepository(db_path)
    for dep_id in (builtin_dep.id, custom_dep.id):
        rec = await repo.get(dep_id)
        rec.profile_snapshot = None
        await repo.save(rec)

    weave_b = make_weave(db_path)
    # Built-in model attaches via registry even without snapshot
    attached_builtin = await weave_b.attach(builtin_dep.id)
    assert attached_builtin.id == builtin_dep.id

    # Custom model without snapshot raises ModelNotFoundError
    with pytest.raises(ModelNotFoundError):
        await weave_b.attach(custom_dep.id)
    with pytest.raises(ModelNotFoundError):
        await weave_b.find(model=custom_profile.id, provider="modal")


@pytest.mark.asyncio
async def test_invalid_or_mismatched_snapshots_raise_model_not_found(db_path: Path):
    """Mismatched or structurally invalid snapshots never substitute a model or crash record loading."""
    import json
    import sqlite3

    weave_a = make_weave(db_path)
    custom_profile = _custom_container_profile("custom/target-model")
    weave_a.register_model(custom_profile)
    dep = await deploy_on_modal(weave_a, model=custom_profile.id)
    repo = SqliteDeploymentRepository(db_path)

    # 1. Snapshot ID does not match record.model
    rec = await repo.get(dep.id)
    other_profile = _custom_container_profile("custom/different-model")
    rec.profile_snapshot = other_profile
    await repo.save(rec)

    weave_b = make_weave(db_path)
    with pytest.raises(ModelNotFoundError, match="does not match"):
        await weave_b.attach(dep.id)
    with pytest.raises(ModelNotFoundError, match="does not match"):
        await weave_b.find(model=custom_profile.id, provider="modal")

    # 2. Snapshot workload_type conflicts with record.workload_type
    rec = await repo.get(dep.id)
    rec.profile_snapshot = custom_profile
    rec.workload_type = WorkloadType.AUDIO
    await repo.save(rec)
    with pytest.raises(ModelNotFoundError, match="workload_type"):
        await make_weave(db_path).attach(dep.id)

    # 3. Corrupted raw JSON snapshot in SQLite (missing required fields, port mismatch, unknown runtime, primitive)
    bad_snapshots = [
        {"id": custom_profile.id},  # missing required fields
        "not-a-dict-snapshot",
        {
            **custom_profile.model_dump(mode="json"),
            "default_runtime": "unknown-runtime-xyz",
        },
        {
            **custom_profile.model_dump(mode="json"),
            "container": None,
        },
        {
            **custom_profile.model_dump(mode="json"),
            "healthcheck": {**custom_profile.healthcheck.model_dump(mode="json"), "port": 9999},
        },
    ]
    for bad_snap in bad_snapshots:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT data_json FROM deployments WHERE id = ?", (dep.id,)
            ).fetchone()
            payload = json.loads(row[0])
            payload["workload_type"] = WorkloadType.CUSTOM.value
            payload["profile_snapshot"] = bad_snap
            conn.execute(
                "UPDATE deployments SET data_json = ? WHERE id = ?",
                (json.dumps(payload), dep.id),
            )

        # Record is still readable for cleanup/reconcile
        loaded = await repo.get(dep.id)
        assert loaded is not None

        # Attach refuses invalid snapshot with ModelNotFoundError
        with pytest.raises(ModelNotFoundError):
            await make_weave(db_path).attach(dep.id)

        # Explicitly registering the valid model still allows attach() even when snapshot is corrupt
        weave_registered = make_weave(db_path)
        weave_registered.register_model(custom_profile)
        attached = await weave_registered.attach(dep.id)
        assert attached.id == dep.id

    # Cleanup via stop() still works under the owning account despite corrupt snapshot
    with patched_modal() as mocks:
        await make_weave(db_path).stop(dep.id)
    mocks.stop_app.assert_awaited_once()


@pytest.mark.asyncio
async def test_snapshot_sanitizes_secrets_and_fails_safely_on_unregistered_recovery(
    db_path: Path,
):
    """Secrets in headers, argv, setup_commands, or metadata are redacted on disk and refuse unsafe recovery."""
    import sqlite3

    from inferweave.adapters.healthcheck.mock_probe import MockHealthcheckProbeAdapter

    class HeaderRecordingProbe(MockHealthcheckProbeAdapter):
        def __init__(self) -> None:
            super().__init__(default_healthy=True)
            self.last_headers: dict[str, str] = {}

        async def probe(
            self,
            url: str,
            method: str = "GET",
            timeout_seconds: float = 5.0,
            headers=None,
            expected_status_codes=None,
        ):
            self.last_headers = dict(headers or {})
            return await super().probe(
                url, method, timeout_seconds, headers, expected_status_codes
            )

    probe_a = HeaderRecordingProbe()
    weave_a = make_weave(db_path, probe=probe_a)
    secret_profile = _custom_container_profile(
        "custom/secret-bearing-model",
        args=["--model", "my-model", "--hf-token", "hf_supersecretvalue123456"],
        setup_commands=["export HF_TOKEN=hf_supersecretvalue123456"],
        headers={"Authorization": "Bearer sk-live-health-secret-999999"},
        metadata={"nested": {"api_key": "sk-nested-metadata-secret-111111"}},
    )
    weave_a.register_model(secret_profile)
    dep_a = await deploy_on_modal(weave_a, model=secret_profile.id)

    # 1. In-memory deployment in weave_a retains working unredacted configuration
    await dep_a.check_health()
    assert probe_a.last_headers.get("Authorization") == "Bearer sk-live-health-secret-999999"
    assert secret_profile.healthcheck.headers["Authorization"] == "Bearer sk-live-health-secret-999999"
    assert secret_profile.container.args == [
        "--model",
        "my-model",
        "--hf-token",
        "hf_supersecretvalue123456",
    ]

    # 2. Raw SQLite bytes contain no secrets
    with sqlite3.connect(db_path) as conn:
        raw_json = conn.execute(
            "SELECT data_json FROM deployments WHERE id = ?", (dep_a.id,)
        ).fetchone()[0]
    assert "hf_supersecretvalue123456" not in raw_json
    assert "sk-live-health-secret-999999" not in raw_json
    assert "sk-nested-metadata-secret-111111" not in raw_json
    assert "[REDACTED]" in raw_json

    # 3. Recovery in a new SDK instance without registration fails safely (never uses [REDACTED])
    weave_b = make_weave(db_path)
    with pytest.raises(ModelNotFoundError, match="redacted"):
        await weave_b.attach(dep_a.id)
    with pytest.raises(ModelNotFoundError, match="redacted"):
        await weave_b.find(model=secret_profile.id, provider="modal")

    # 4. Re-registering the profile in weave_b allows attach() with full working config
    probe_b = HeaderRecordingProbe()
    weave_c = make_weave(db_path, probe=probe_b)
    weave_c.register_model(secret_profile)
    attached = await weave_c.attach(dep_a.id)
    await attached.check_health()
    assert probe_b.last_headers.get("Authorization") == "Bearer sk-live-health-secret-999999"


@pytest.mark.asyncio
async def test_write_ahead_and_lifecycle_preserve_snapshot(db_path: Path):
    """The initial write-ahead record and subsequent lifecycle updates preserve profile_snapshot."""
    from unittest.mock import patch

    weave_a = make_weave(db_path, accounts=modal_pool("a"))
    custom_profile = _custom_container_profile("custom/write-ahead-model")
    weave_a.register_model(custom_profile)

    write_ahead_snapshots = []
    orig_provision = weave_a.router.get("modal").provision

    async def spy_provision(record, request, profile, runtime, account):
        persisted_before = await _read_record(db_path, record.id)
        write_ahead_snapshots.append(
            persisted_before.profile_snapshot if persisted_before else None
        )
        return await orig_provision(record, request, profile, runtime, account)

    with patch.object(weave_a.router.get("modal"), "provision", side_effect=spy_provision):
        dep = await deploy_on_modal(weave_a, model=custom_profile.id, account="a")

    assert len(write_ahead_snapshots) == 1
    assert write_ahead_snapshots[0] is not None
    assert write_ahead_snapshots[0].id == custom_profile.id

    # Preserved through wait_for_ready, touch_activity, refresh, and stop
    await dep.wait_for_ready()
    await weave_a.lifecycle_service.touch_activity(dep.id)
    with patched_modal():
        await dep.refresh()
        await dep.stop()

    final_rec = await _read_record(db_path, dep.id)
    assert final_rec is not None
    assert final_rec.state == DeploymentState.STOPPED
    assert final_rec.profile_snapshot is not None
    assert final_rec.profile_snapshot.id == custom_profile.id
