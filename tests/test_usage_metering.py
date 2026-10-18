"""Logical metering, independent of retries, provider billing and process lifetimes."""

import asyncio
import base64
import json
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime, timedelta

import httpx
import pytest
import pytest_asyncio
from support import deploy_on_modal, make_weave, modal_pool, patched_modal

from inferweave import DeploymentUsage, UsageReport
from inferweave.accounts import AccountManager, AccountsConfig
from inferweave.adapters.lifecycle import (
    InMemoryDeploymentRepository,
    JsonDeploymentRepository,
    SqliteDeploymentRepository,
)
from inferweave.clients import InferenceClient, InferenceConfig, InferenceTransport
from inferweave.clients.metering import UsageDelta
from inferweave.core.exceptions import (
    AccountUnavailableError,
    DeploymentNotFoundError,
    InferenceError,
    InvalidInferenceResponseError,
    UnsupportedWorkloadError,
)
from inferweave.domain.deployment_record import DeploymentRecord
from inferweave.models.enums import DeploymentState, WorkloadType
from inferweave.ports.deployment_repository import DeploymentRepositoryPort
from inferweave.services.usage_service import UsageService


@pytest.fixture(params=["memory", "json", "sqlite"])
def repository(request, tmp_path):
    if request.param == "memory":
        return InMemoryDeploymentRepository()
    if request.param == "json":
        return JsonDeploymentRepository(tmp_path / "d.json")
    return SqliteDeploymentRepository(tmp_path / "d.db")


@pytest_asyncio.fixture
async def clients(repository):
    await repository.save(DeploymentRecord(
        id="d", model="test", provider="modal", state=DeploymentState.HEALTHY,
        usage=DeploymentUsage(),
    ))
    service = UsageService(repository, AccountManager(AccountsConfig()))
    pools = []

    def make(handler, workload=WorkloadType.CUSTOM, clock=lambda: 10):
        pool = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        pools.append(pool)
        transport = InferenceTransport(
            "d", endpoint_fn=lambda: "https://test.example",
            http_client=pool, config=InferenceConfig(max_retries=2, backoff_base_seconds=0),
        )
        return InferenceClient(
            "d", workload, transport, on_usage=lambda delta: service.record("d", delta), clock=clock,
        )

    yield make
    for pool in pools:
        await pool.aclose()


@pytest.mark.asyncio
async def test_retry_redirect_bytes_and_latency_are_one_logical_call(repository, clients):
    seen = []
    responses = [
        httpx.Response(503, content=b"wait"),
        httpx.Response(303, headers={"location": "/result"}, content=b"redirect"),
        httpx.Response(200, content=b"answer"),
    ]

    def handler(request):
        seen.append(request)
        return responses.pop(0)

    times = iter([100.0, 102.5])
    client = clients(handler, WorkloadType.AUDIO, clock=lambda: next(times))
    # Audio POST policy retries; generic POST deliberately does not.
    audio = b"RIFF\x00\x00\x00\x00WAVEaudio"
    responses[-1] = httpx.Response(200, content=audio)
    assert await client.synthesize("test") == audio
    usage = (await repository.get("d")).usage
    assert usage.requests == usage.successes == 1
    assert usage.failures == usage.cancellations == usage.unfinished_requests == 0
    assert usage.total_latency_seconds == usage.average_latency_seconds == 2.5
    assert usage.request_body_bytes == sum(len(r.content) for r in seen)
    assert usage.response_body_bytes == len(b"waitredirect") + len(audio)
    assert usage.incomplete_transfers == 0
    assert len(seen) == 3


@pytest.mark.asyncio
async def test_generic_success_failure_and_validation_failures(repository, clients):
    client = clients(lambda _: httpx.Response(200, content=b"abc"))
    await client.request("POST", "/", content="λ")
    with pytest.raises(ValueError):
        await client.request("POST", "/", json={}, content=b"conflict")
    failing = clients(lambda _: httpx.Response(400, content=b"bad"))
    with pytest.raises(InferenceError):
        await failing.request("GET", "/")
    with pytest.raises(UnsupportedWorkloadError):
        await client.render("wrong workload")
    usage = (await repository.get("d")).usage
    assert usage.requests == 3 and usage.successes == 1 and usage.failures == 2
    assert usage.request_body_bytes == len("λ".encode())
    assert usage.response_body_bytes == 6 and usage.unfinished_requests == 0


@pytest.mark.asyncio
async def test_image_success_and_invalid_response_count_after_decode(repository, clients):
    body = {"data": [{"b64_json": base64.b64encode(b"png").decode()}]}
    client = clients(lambda _: httpx.Response(200, json=body), WorkloadType.IMAGE)
    assert await client.render("a cat") == [b"png"]
    invalid = clients(lambda _: httpx.Response(200, json={}), WorkloadType.IMAGE)
    with pytest.raises(InvalidInferenceResponseError):
        await invalid.render("a cat")
    usage = (await repository.get("d")).usage
    assert usage.requests == 2 and usage.successes == usage.failures == 1
    # HTTP response bytes (including JSON/base64), not decoded image bytes.
    assert usage.response_body_bytes == len(httpx.Response(200, json=body).content) + 2


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["cancel", "timeout", "connection"])
async def test_failures_and_cancellation_are_finished_once(repository, clients, failure):
    entered = asyncio.Event()
    release = asyncio.Event()

    async def handler(request):
        entered.set()
        if failure == "cancel":
            await release.wait()
        if failure == "timeout":
            raise httpx.ReadTimeout("dummy secret must not persist", request=request)
        raise httpx.ConnectError("dummy secret must not persist", request=request)

    client = clients(handler)
    task = asyncio.create_task(client.request("GET", "/"))
    if failure == "cancel":
        await entered.wait()
        mid = (await repository.get("d")).usage
        assert mid.requests == mid.unfinished_requests == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(InferenceError):
            await task
    usage = (await repository.get("d")).usage
    assert usage.requests == 1 and usage.successes == usage.unfinished_requests == 0
    assert usage.cancellations == int(failure == "cancel")
    assert usage.failures == int(failure != "cancel")
    assert usage.incomplete_transfers == 1 and usage.response_body_bytes == 0
    assert "dummy secret" not in (await repository.get("d")).model_dump_json()


@pytest.mark.asyncio
async def test_concurrent_calls_and_stale_lifecycle_save_preserve_all_fields(repository, clients):
    old = await repository.get("d")
    client = clients(lambda _: httpx.Response(200, content=b"payload"))
    await asyncio.gather(*(client.request("POST", "/", content=b"input") for _ in range(30)))
    old.mark_stopped()
    old.error_message = "lifecycle message"
    await repository.save(old)  # Simulates slow stop/refresh saving a snapshot from before requests.
    result = await repository.get("d")
    assert result.state == DeploymentState.STOPPED and result.error_message == "lifecycle message"
    assert result.usage.requests == result.usage.successes == 30
    assert result.usage.request_body_bytes == 150 and result.usage.response_body_bytes == 210
    assert result.usage.unfinished_requests == 0


@pytest.mark.asyncio
async def test_atomic_usage_never_resurrects_deleted_records(repository):
    await repository.save(DeploymentRecord(id="d", model="test", provider="modal"))
    service = UsageService(repository, AccountManager(AccountsConfig()))
    await repository.delete("d")
    with pytest.raises(DeploymentNotFoundError):
        await service.record("d", UsageDelta(requests=1))
    assert await repository.get("d") is None


@pytest.mark.asyncio
async def test_atomic_mutation_rollback(repository):
    await repository.save(DeploymentRecord(id="d", model="test", provider="modal"))

    def broken(record):
        record.mark_stopped()
        raise ValueError("abort")

    with pytest.raises(ValueError, match="abort"):
        await repository.update("d", broken)
    assert (await repository.get("d")).state == DeploymentState.PENDING


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["json", "sqlite"])
async def test_restart_attach_find_and_read_only_snapshots(tmp_path, backend):
    pool = modal_pool("a", "b")
    weaves = []

    def build():
        weave = make_weave(tmp_path / "ignored.db", handler=lambda _: httpx.Response(200, content=b"ok"), accounts=pool)
        if backend == "json":
            weave.lifecycle_service._repository = JsonDeploymentRepository(tmp_path / "d.json")
            weave.provisioning_service._repository = weave.lifecycle_service.repository
            weave._usage_service._repository = weave.lifecycle_service.repository
        else:
            weave.lifecycle_service._repository = SqliteDeploymentRepository(tmp_path / "d.db")
            weave.provisioning_service._repository = weave.lifecycle_service.repository
            weave._usage_service._repository = weave.lifecycle_service.repository
        weaves.append(weave)
        return weave

    try:
        first = build()
        handle = await deploy_on_modal(first, account="b")
        await handle.request("POST", "/", content=b"one")
        before = await handle.usage()
        assert isinstance(before, UsageReport)
        # Simulate a killed process after a durable start, with no terminal result.
        await first._usage_service.record(handle.id, UsageDelta(requests=1))
        await first.close()
        second = build()
        recovered = await second.attach(handle.id)
        assert await second.find(model=handle.model) is recovered
        await recovered.request("GET", "/")
        report = await recovered.usage()
        assert report.usage.requests == 3 and report.usage.successes == 2
        assert report.usage.unfinished_requests == 1  # no replay or fabricated failure
        assert report.pricing == before.pricing and recovered.account == "b"
        assert before.usage.requests == 1  # detached, immutable snapshots
        with pytest.raises(ValueError):
            report.usage.requests = 999
        with patched_modal():
            await recovered.stop()
        assert (await second.get_usage(handle.id)).usage == report.usage
    finally:
        for weave in weaves:
            await weave.close()
            await weave._inference_http_client.aclose()


@pytest.mark.asyncio
async def test_usage_read_enforces_owner_and_never_leaks_credentials(tmp_path):
    accounts = modal_pool("a", "b")
    weave = make_weave(tmp_path / "d.db", accounts=accounts, handler=lambda _: httpx.Response(200))
    try:
        a = await deploy_on_modal(weave, account="a")
        b = await deploy_on_modal(weave, account="b")
        await a.request("GET", "/")
        assert (await a.usage()).usage.requests == 1 and (await b.usage()).usage.requests == 0
        assert "SENTINEL" not in json.dumps([r.model_dump(mode="json") for r in await weave.list_records()])
        removed = make_weave(tmp_path / "d.db", accounts=modal_pool("a"))
        try:
            with pytest.raises(AccountUnavailableError):
                await removed.get_usage(b.id)
        finally:
            await removed.close()
    finally:
        await weave.close()
        await weave._inference_http_client.aclose()


@pytest.mark.asyncio
async def test_legacy_usage_unknown_until_observation(repository):
    record = DeploymentRecord.model_validate({
        "id": "old", "model": "test", "provider": "modal",
        "created_at": (datetime.now(UTC) - timedelta(days=365)).isoformat(),
    })
    await repository.save(record)
    service = UsageService(repository, AccountManager(AccountsConfig()))
    before = await service.report("old")
    assert before.usage is None and before.pricing is None
    assert before.cost.estimated_cost_usd is before.cost.measured_gpu_seconds is None
    await service.record("old", UsageDelta(requests=1))
    await service.record("old", UsageDelta(successes=1))
    after = await service.report("old")
    assert after.usage.requests == after.usage.successes == 1
    assert after.usage.tracking_started_at > record.created_at


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["json", "sqlite"])
async def test_real_processes_share_atomic_usage_without_clobbering_lifecycle(tmp_path, backend):
    path = tmp_path / ("d.json" if backend == "json" else "d.db")
    repo = JsonDeploymentRepository(path) if backend == "json" else SqliteDeploymentRepository(path)
    await repo.save(DeploymentRecord(id="d", model="test", provider="modal", usage=DeploymentUsage()))
    script = """
import asyncio, sys
from inferweave.accounts import AccountManager, AccountsConfig
from inferweave.adapters.lifecycle import JsonDeploymentRepository, SqliteDeploymentRepository
from inferweave.clients.metering import UsageDelta
from inferweave.services.usage_service import UsageService
async def main():
    repo = (JsonDeploymentRepository if sys.argv[1] == 'json' else SqliteDeploymentRepository)(sys.argv[2])
    service = UsageService(repo, AccountManager(AccountsConfig()))
    for _ in range(15):
        await service.record('d', UsageDelta(requests=1))
        await service.record('d', UsageDelta(successes=1, response_body_bytes=7))
asyncio.run(main())
"""
    processes = [await asyncio.to_thread(
        subprocess.Popen, [sys.executable, "-c", script, backend, str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ) for _ in range(2)]
    # Stale lifecycle save races cross-process metering.
    old = await repo.get("d")
    old.mark_stopped()
    await repo.save(old)
    for process in processes:
        _, stderr = await asyncio.to_thread(process.communicate, timeout=30)
        assert process.returncode == 0, stderr.decode()
    result = await repo.get("d")
    assert result.state == DeploymentState.STOPPED
    assert result.usage.requests == result.usage.successes == 30
    assert result.usage.response_body_bytes == 210


@pytest.mark.asyncio
async def test_meter_failure_preserves_inference_and_only_logs_error_type(clients, caplog):
    client = clients(lambda _: httpx.Response(200, content=b"ok"))

    async def broken(_):
        raise OSError("SENTINEL-storage-password")

    client._on_usage = broken
    assert (await client.request("GET", "/")).content == b"ok"
    assert "Usage persistence failed (OSError)" in caplog.text
    assert "SENTINEL" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["start", "finish"])
async def test_repeated_cancellation_during_write_drains_exactly_once(repository, clients, phase):
    client = clients(lambda _: httpx.Response(200, content=b"ok"))
    original = client._on_usage
    entered = asyncio.Event()
    release = asyncio.Event()

    async def paused(delta):
        if bool(delta.requests) == (phase == "start"):
            entered.set()
            await release.wait()
        await original(delta)

    client._on_usage = paused
    task = asyncio.create_task(client.request("GET", "/"))
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    usage = (await repository.get("d")).usage
    assert usage.requests == 1 and usage.unfinished_requests == 0
    # Once the response is complete, cancellation during bookkeeping does not undo success.
    assert usage.successes == int(phase == "finish")
    assert usage.cancellations == int(phase == "start")


@pytest.mark.asyncio
async def test_json_mutation_refuses_to_drop_corrupt_or_unparseable_records(tmp_path):
    path = tmp_path / "d.json"
    repo = JsonDeploymentRepository(path)
    service = UsageService(repo, AccountManager(AccountsConfig()))
    for raw in ["{invalid", json.dumps({"other": {"missing": "fields"}}), "[]"]:
        path.write_text(raw, encoding="utf-8")
        with pytest.raises(ValueError, match="Cannot mutate invalid deployment storage"):
            await service.record("d", UsageDelta(requests=1))
        assert path.read_text(encoding="utf-8") == raw


@pytest.mark.asyncio
async def test_concurrent_deployments_with_shared_http_client_keep_bytes_separate(repository):
    for dep_id in ("a", "b"):
        await repository.save(DeploymentRecord(id=dep_id, model="test", provider="modal", usage=DeploymentUsage()))
    service = UsageService(repository, AccountManager(AccountsConfig()))
    entered = {name: asyncio.Event() for name in ("a", "b")}

    async def handler(request):
        name = request.url.path[1:]
        entered[name].set()
        await entered["a" if name == "b" else "b"].wait()
        return httpx.Response(200, content=request.content * 2)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as pool:
        def make(name):
            return InferenceClient(name, WorkloadType.CUSTOM, InferenceTransport(
                name, endpoint_fn=lambda: "https://test.example", http_client=pool,
            ), on_usage=lambda delta: service.record(name, delta))
        await asyncio.gather(make("a").request("POST", "/a", content=b"aaa"),
                             make("b").request("POST", "/b", content=b"bbbbb"))
    a, b = (await service.report("a")).usage, (await service.report("b")).usage
    assert (a.requests, a.request_body_bytes, a.response_body_bytes) == (1, 3, 6)
    assert (b.requests, b.request_body_bytes, b.response_body_bytes) == (1, 5, 10)


@pytest.mark.asyncio
async def test_old_custom_repository_still_attaches_with_usage_unavailable(tmp_path):
    from fakes import FakeProvider
    from fakes import make_weave as fake_weave

    class OldRepository(DeploymentRepositoryPort):
        def __init__(self):
            self.inner = InMemoryDeploymentRepository()

        async def save(self, record):
            await self.inner.save(record)

        async def get(self, deployment_id):
            return await self.inner.get(deployment_id)

        async def list_all(self):
            return await self.inner.list_all()

        async def delete(self, deployment_id):
            await self.inner.delete(deployment_id)

    repo = OldRepository()
    fake = FakeProvider()
    weave = fake_weave(tmp_path / "unused.db", AccountsConfig(), providers=[fake], repository=repo)
    try:
        handle = await weave.deploy("fish-s2-pro", provider="modal", wait_for_ready=False)
        await weave.lifecycle_service.touch_activity(handle.id)
        record = await repo.get(handle.id)
        record.last_activity_at = None
        await repo.save(record)
        weave._active_deployments.clear()
        attached = await weave.attach(handle.id)
        assert attached.id == handle.id
        assert (await attached.usage()).usage is None
    finally:
        await weave.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["json", "sqlite"])
async def test_legacy_records_and_malformed_usage_do_not_leak_validation_input(tmp_path, backend, caplog):
    path = tmp_path / ("d.json" if backend == "json" else "d.db")
    repo = JsonDeploymentRepository(path) if backend == "json" else SqliteDeploymentRepository(path)
    legacy = {"id": "old", "model": "fish-s2-pro", "provider": "modal", "state": "healthy"}
    await repo.save(DeploymentRecord.model_validate(legacy))

    def write_raw(payload):
        if backend == "json":
            path.write_text(json.dumps({"old": payload}), encoding="utf-8")
        else:
            with sqlite3.connect(path) as conn:
                conn.execute("UPDATE deployments SET data_json = ? WHERE id = ?", (json.dumps(payload), "old"))

    write_raw(legacy)  # Actual old storage lacks the new fields entirely.
    report = await UsageService(repo, AccountManager(AccountsConfig())).report("old")
    assert report.usage is report.pricing is None
    assert report.cost.estimated_cost_usd is None
    write_raw({**legacy, "usage": {"requests": "SENTINEL-invalid-counter-secret"}})
    assert await repo.get("old") is None
    assert await repo.list_all() == []
    assert "ValidationError" in caplog.text
    assert "SENTINEL-invalid-counter-secret" not in caplog.text
